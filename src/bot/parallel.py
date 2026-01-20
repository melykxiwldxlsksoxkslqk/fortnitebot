"""
Multi-account parallelism - параллельный запуск нескольких ботов.

Особенности:
- Пул воркеров для ограничения одновременных ботов
- Очередь задач с приоритетами
- Мониторинг и статистика
- Автоматические рестарты при ошибках
- Graceful shutdown
"""

import asyncio
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime
import traceback

from ..core import get_logger
from .logic import BotLogic
from .runner import BotRunner

logger = get_logger(__name__)


class BotPriority(int, Enum):
    """Приоритет бота в очереди."""
    HIGH = 0
    NORMAL = 1
    LOW = 2


class BotStatus(str, Enum):
    """Статус бота."""
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"
    RESTARTING = "restarting"


@dataclass(order=True)
class BotTask:
    """Задача запуска бота."""
    priority: BotPriority = field(compare=True)
    account: Dict[str, str] = field(compare=False)
    proxy: Optional[Dict[str, str]] = field(default=None, compare=False)
    island_code: str = field(default="", compare=False)
    headless: bool = field(default=True, compare=False)
    retry_count: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    
    @property
    def login(self) -> str:
        return self.account.get('login', 'unknown')


@dataclass
class BotResult:
    """Результат работы бота."""
    login: str
    success: bool
    started_at: float
    finished_at: float
    error: Optional[str] = None
    retry_count: int = 0
    
    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at


@dataclass
class PoolStats:
    """Статистика пула."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_workers: int = 0
    queued_tasks: int = 0
    average_duration: float = 0.0
    total_retries: int = 0
    started_at: Optional[float] = None


class BotWorkerPool:
    """
    Пул воркеров для параллельного запуска ботов.
    
    Особенности:
    - Ограничение максимального количества одновременных ботов
    - Очередь с приоритетами
    - Автоматические рестарты
    - Мониторинг
    """
    
    def __init__(
        self,
        max_workers: int = 5,
        settings: Optional[Dict[str, Any]] = None,
        status_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Инициализация пула.
        
        Args:
            max_workers: Максимальное количество одновременных ботов
            settings: Общие настройки для всех ботов
            status_callback: Колбэк для обновления статуса
        """
        self.max_workers = max_workers
        self.settings = settings or {}
        self.status_callback = status_callback
        
        # Состояние
        self._executor: Optional[ThreadPoolExecutor] = None
        self._task_queue: queue.PriorityQueue[BotTask] = queue.PriorityQueue()
        self._futures: Dict[str, Future] = {}
        self._results: Dict[str, BotResult] = {}
        self._statuses: Dict[str, BotStatus] = {}
        self._bots: Dict[str, BotLogic] = {}
        
        # Синхронизация
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        
        # Статистика
        self._stats = PoolStats()
        self._durations: List[float] = []
        
        logger.info(f"BotWorkerPool создан: max_workers={max_workers}")
    
    def _log_status(self, login: str, message: str):
        """Логирование статуса."""
        logger.info(f"[{login}] {message}")
        if self.status_callback:
            try:
                self.status_callback(login, message)
            except Exception:
                pass
    
    def start(self):
        """Запуск пула."""
        if self._executor is not None:
            logger.warning("Пул уже запущен")
            return
        
        self._shutdown_event.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="BotWorker"
        )
        self._stats.started_at = time.time()
        
        # Запуск воркера очереди
        self._worker_thread = threading.Thread(
            target=self._queue_worker,
            daemon=True,
            name="QueueWorker"
        )
        self._worker_thread.start()
        
        logger.info("Пул запущен")
    
    def stop(self, wait: bool = True, timeout: float = 30.0):
        """
        Остановка пула.
        
        Args:
            wait: Ожидать завершения всех задач
            timeout: Таймаут ожидания
        """
        logger.info("Остановка пула...")
        self._shutdown_event.set()
        
        # Остановка всех ботов
        with self._lock:
            for login, bot in list(self._bots.items()):
                try:
                    bot.request_stop()
                    self._statuses[login] = BotStatus.STOPPED
                except Exception:
                    pass
        
        # Ожидание завершения
        if self._executor and wait:
            self._executor.shutdown(wait=True, cancel_futures=True)
        elif self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        
        self._executor = None
        
        # Ожидание воркера очереди
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        
        logger.info("Пул остановлен")
    
    def submit(
        self,
        account: Dict[str, str],
        proxy: Optional[Dict[str, str]] = None,
        island_code: Optional[str] = None,
        priority: BotPriority = BotPriority.NORMAL,
        headless: bool = True,
    ) -> str:
        """
        Добавление бота в очередь.
        
        Args:
            account: Данные аккаунта
            proxy: Прокси (опционально)
            island_code: Код острова (опционально, берется из settings)
            priority: Приоритет
            headless: Режим без UI
        
        Returns:
            Логин добавленного бота
        """
        login = account.get('login', 'unknown')
        code = island_code or self.settings.get('island_code', '')
        
        task = BotTask(
            priority=priority,
            account=account,
            proxy=proxy,
            island_code=code,
            headless=headless,
        )
        
        with self._lock:
            self._task_queue.put(task)
            self._statuses[login] = BotStatus.QUEUED
            self._stats.total_tasks += 1
            self._stats.queued_tasks = self._task_queue.qsize()
        
        self._log_status(login, "Добавлен в очередь")
        return login
    
    def submit_many(
        self,
        accounts: List[Dict[str, str]],
        proxies: Optional[List[Dict[str, str]]] = None,
        island_code: Optional[str] = None,
        priority: BotPriority = BotPriority.NORMAL,
        headless: bool = True,
    ) -> List[str]:
        """
        Добавление нескольких ботов в очередь.
        
        Args:
            accounts: Список аккаунтов
            proxies: Список прокси (по порядку или None)
            island_code: Код острова
            priority: Приоритет
            headless: Режим без UI
        
        Returns:
            Список логинов добавленных ботов
        """
        proxies = proxies or []
        logins = []
        
        for i, account in enumerate(accounts):
            proxy = proxies[i] if i < len(proxies) else None
            login = self.submit(
                account=account,
                proxy=proxy,
                island_code=island_code,
                priority=priority,
                headless=headless,
            )
            logins.append(login)
        
        logger.info(f"Добавлено {len(logins)} ботов в очередь")
        return logins
    
    def cancel(self, login: str) -> bool:
        """
        Отмена задачи бота.
        
        Args:
            login: Логин бота
        
        Returns:
            True если отменено
        """
        with self._lock:
            # Отмена запущенного бота
            if login in self._bots:
                self._bots[login].request_stop()
                self._statuses[login] = BotStatus.STOPPED
                return True
            
            # Отмена future
            if login in self._futures:
                future = self._futures[login]
                if future.cancel():
                    self._statuses[login] = BotStatus.STOPPED
                    return True
        
        return False
    
    def cancel_all(self):
        """Отмена всех задач."""
        with self._lock:
            # Очистка очереди
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Остановка всех ботов
            for login in list(self._bots.keys()):
                self.cancel(login)
        
        logger.info("Все задачи отменены")
    
    def get_status(self, login: str) -> Optional[BotStatus]:
        """Получение статуса бота."""
        with self._lock:
            return self._statuses.get(login)
    
    def get_result(self, login: str) -> Optional[BotResult]:
        """Получение результата бота."""
        with self._lock:
            return self._results.get(login)
    
    def get_stats(self) -> PoolStats:
        """Получение статистики пула."""
        with self._lock:
            self._stats.queued_tasks = self._task_queue.qsize()
            self._stats.active_workers = len([
                f for f in self._futures.values() 
                if not f.done()
            ])
            if self._durations:
                self._stats.average_duration = sum(self._durations) / len(self._durations)
            return self._stats
    
    def get_all_statuses(self) -> Dict[str, BotStatus]:
        """Получение всех статусов."""
        with self._lock:
            return dict(self._statuses)
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Ожидание завершения всех задач.
        
        Args:
            timeout: Таймаут в секундах
        
        Returns:
            True если все завершились до таймаута
        """
        start = time.time()
        
        while True:
            with self._lock:
                # Проверяем очередь
                if self._task_queue.empty() and not self._futures:
                    return True
                
                # Проверяем все futures
                all_done = all(f.done() for f in self._futures.values())
                if all_done and self._task_queue.empty():
                    return True
            
            # Проверяем таймаут
            if timeout and time.time() - start > timeout:
                return False
            
            time.sleep(0.5)
    
    def _queue_worker(self):
        """Воркер обработки очереди."""
        logger.info("Queue worker запущен")
        
        while not self._shutdown_event.is_set():
            try:
                # Получаем задачу с таймаутом
                try:
                    task = self._task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if self._shutdown_event.is_set():
                    break
                
                # Ждём свободный слот
                while True:
                    with self._lock:
                        active = len([f for f in self._futures.values() if not f.done()])
                        if active < self.max_workers:
                            break
                    
                    if self._shutdown_event.is_set():
                        break
                    time.sleep(0.5)
                
                if self._shutdown_event.is_set():
                    break
                
                # Запускаем задачу
                self._start_task(task)
                
            except Exception as e:
                logger.error(f"Queue worker error: {e}", exc_info=True)
        
        logger.info("Queue worker остановлен")
    
    def _start_task(self, task: BotTask):
        """Запуск задачи в пуле."""
        login = task.login
        
        with self._lock:
            self._statuses[login] = BotStatus.STARTING
            self._stats.queued_tasks = self._task_queue.qsize()
        
        self._log_status(login, "Запуск...")
        
        # Создаём future
        future = self._executor.submit(self._run_bot, task)
        
        with self._lock:
            self._futures[login] = future
        
        # Обработка результата в отдельном потоке
        def handle_result(f: Future):
            try:
                result = f.result()
                self._handle_task_result(task, result)
            except Exception as e:
                self._handle_task_error(task, e)
        
        future.add_done_callback(handle_result)
    
    def _run_bot(self, task: BotTask) -> BotResult:
        """Запуск бота (выполняется в воркере)."""
        login = task.login
        started_at = time.time()
        
        with self._lock:
            self._statuses[login] = BotStatus.RUNNING
        
        try:
            # Создаём колбэк для статуса
            def status_cb(msg: str):
                self._log_status(login, msg)
            
            # Создаём BotLogic
            bot = BotLogic(
                account=task.account,
                proxy=task.proxy,
                config={
                    **self.settings,
                    'island_code': task.island_code,
                    'headless': task.headless,
                },
                update_status_callback=lambda l, m: status_cb(m),
            )
            
            with self._lock:
                self._bots[login] = bot
            
            # Запускаем
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(bot.run())
                success = not bot.stop_requested
            finally:
                loop.close()
            
            return BotResult(
                login=login,
                success=success,
                started_at=started_at,
                finished_at=time.time(),
                retry_count=task.retry_count,
            )
            
        except Exception as e:
            logger.error(f"[{login}] Ошибка: {e}", exc_info=True)
            return BotResult(
                login=login,
                success=False,
                started_at=started_at,
                finished_at=time.time(),
                error=str(e),
                retry_count=task.retry_count,
            )
        finally:
            with self._lock:
                self._bots.pop(login, None)
    
    def _handle_task_result(self, task: BotTask, result: BotResult):
        """Обработка результата задачи."""
        login = task.login
        
        with self._lock:
            self._results[login] = result
            self._durations.append(result.duration)
            self._futures.pop(login, None)
            
            if result.success:
                self._statuses[login] = BotStatus.SUCCESS
                self._stats.completed_tasks += 1
                self._log_status(login, f"Успешно завершён ({result.duration:.1f}с)")
            else:
                # Проверяем нужен ли рестарт
                if task.retry_count < task.max_retries and not self._shutdown_event.is_set():
                    self._statuses[login] = BotStatus.RESTARTING
                    self._stats.total_retries += 1
                    self._log_status(login, f"Рестарт #{task.retry_count + 1}...")
                    
                    # Добавляем обратно в очередь
                    new_task = BotTask(
                        priority=task.priority,
                        account=task.account,
                        proxy=task.proxy,
                        island_code=task.island_code,
                        headless=task.headless,
                        retry_count=task.retry_count + 1,
                        max_retries=task.max_retries,
                    )
                    self._task_queue.put(new_task)
                else:
                    self._statuses[login] = BotStatus.FAILED
                    self._stats.failed_tasks += 1
                    self._log_status(login, f"Завершён с ошибкой: {result.error}")
    
    def _handle_task_error(self, task: BotTask, error: Exception):
        """Обработка ошибки задачи."""
        login = task.login
        
        result = BotResult(
            login=login,
            success=False,
            started_at=time.time(),
            finished_at=time.time(),
            error=str(error),
            retry_count=task.retry_count,
        )
        
        self._handle_task_result(task, result)


class AsyncBotPool:
    """
    Асинхронный пул ботов.
    
    Альтернативная реализация с использованием asyncio.
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        settings: Optional[Dict[str, Any]] = None,
        status_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.max_concurrent = max_concurrent
        self.settings = settings or {}
        self.status_callback = status_callback
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, BotResult] = {}
        self._shutdown = False
    
    async def submit(
        self,
        account: Dict[str, str],
        proxy: Optional[Dict[str, str]] = None,
        island_code: Optional[str] = None,
    ) -> str:
        """Добавление бота."""
        login = account.get('login', 'unknown')
        
        task = asyncio.create_task(
            self._run_bot(account, proxy, island_code),
            name=f"bot_{login}"
        )
        self._tasks[login] = task
        
        return login
    
    async def submit_many(
        self,
        accounts: List[Dict[str, str]],
        proxies: Optional[List[Dict[str, str]]] = None,
        island_code: Optional[str] = None,
    ) -> List[str]:
        """Добавление нескольких ботов."""
        proxies = proxies or []
        logins = []
        
        for i, account in enumerate(accounts):
            proxy = proxies[i] if i < len(proxies) else None
            login = await self.submit(account, proxy, island_code)
            logins.append(login)
        
        return logins
    
    async def wait_all(self, timeout: Optional[float] = None) -> Dict[str, BotResult]:
        """Ожидание всех задач."""
        if not self._tasks:
            return {}
        
        done, pending = await asyncio.wait(
            self._tasks.values(),
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )
        
        # Отмена незавершённых
        for task in pending:
            task.cancel()
        
        return self._results
    
    async def cancel_all(self):
        """Отмена всех задач."""
        self._shutdown = True
        
        for task in self._tasks.values():
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        
        self._tasks.clear()
    
    async def _run_bot(
        self,
        account: Dict[str, str],
        proxy: Optional[Dict[str, str]],
        island_code: Optional[str],
    ) -> BotResult:
        """Запуск бота."""
        login = account.get('login', 'unknown')
        started_at = time.time()
        
        async with self._semaphore:
            if self._shutdown:
                return BotResult(
                    login=login,
                    success=False,
                    started_at=started_at,
                    finished_at=time.time(),
                    error="Shutdown requested"
                )
            
            try:
                code = island_code or self.settings.get('island_code', '')
                
                bot = BotLogic(
                    account=account,
                    proxy=proxy,
                    config={
                        **self.settings,
                        'island_code': code,
                    },
                    update_status_callback=self.status_callback,
                )
                
                await bot.run()
                
                result = BotResult(
                    login=login,
                    success=True,
                    started_at=started_at,
                    finished_at=time.time(),
                )
                
            except asyncio.CancelledError:
                result = BotResult(
                    login=login,
                    success=False,
                    started_at=started_at,
                    finished_at=time.time(),
                    error="Cancelled"
                )
            except Exception as e:
                result = BotResult(
                    login=login,
                    success=False,
                    started_at=started_at,
                    finished_at=time.time(),
                    error=str(e)
                )
            
            self._results[login] = result
            return result


# === Удобные функции ===

def run_bots_parallel(
    accounts: List[Dict[str, str]],
    proxies: Optional[List[Dict[str, str]]] = None,
    island_code: str = "",
    max_workers: int = 5,
    headless: bool = True,
    status_callback: Optional[Callable[[str, str], None]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, BotResult]:
    """
    Запуск нескольких ботов параллельно.
    
    Args:
        accounts: Список аккаунтов
        proxies: Список прокси
        island_code: Код острова
        max_workers: Макс. одновременных ботов
        headless: Режим без UI
        status_callback: Колбэк статуса
        timeout: Таймаут
    
    Returns:
        Словарь результатов по логинам
    """
    pool = BotWorkerPool(
        max_workers=max_workers,
        settings={'island_code': island_code, 'headless': headless},
        status_callback=status_callback,
    )
    
    try:
        pool.start()
        pool.submit_many(accounts, proxies, island_code, headless=headless)
        pool.wait_for_completion(timeout=timeout)
        
        return {login: pool.get_result(login) for login in [a.get('login', 'unknown') for a in accounts]}
    finally:
        pool.stop()


async def run_bots_async(
    accounts: List[Dict[str, str]],
    proxies: Optional[List[Dict[str, str]]] = None,
    island_code: str = "",
    max_concurrent: int = 5,
    status_callback: Optional[Callable[[str, str], None]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, BotResult]:
    """
    Асинхронный запуск нескольких ботов.
    
    Args:
        accounts: Список аккаунтов
        proxies: Список прокси
        island_code: Код острова
        max_concurrent: Макс. одновременных ботов
        status_callback: Колбэк статуса
        timeout: Таймаут
    
    Returns:
        Словарь результатов по логинам
    """
    pool = AsyncBotPool(
        max_concurrent=max_concurrent,
        settings={'island_code': island_code},
        status_callback=status_callback,
    )
    
    await pool.submit_many(accounts, proxies, island_code)
    return await pool.wait_all(timeout=timeout)
