"""
Тесты для модуля параллельного запуска ботов.
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock


class TestBotPriority:
    """Тесты приоритетов."""
    
    def test_priority_values(self):
        """Тест значений приоритетов."""
        from src.bot.parallel import BotPriority
        
        assert BotPriority.HIGH == 0
        assert BotPriority.NORMAL == 1
        assert BotPriority.LOW == 2
    
    def test_priority_ordering(self):
        """Тест сортировки приоритетов."""
        from src.bot.parallel import BotPriority
        
        priorities = [BotPriority.LOW, BotPriority.HIGH, BotPriority.NORMAL]
        sorted_priorities = sorted(priorities)
        
        assert sorted_priorities == [BotPriority.HIGH, BotPriority.NORMAL, BotPriority.LOW]


class TestBotStatus:
    """Тесты статусов."""
    
    def test_status_values(self):
        """Тест значений статусов."""
        from src.bot.parallel import BotStatus
        
        assert BotStatus.QUEUED == "queued"
        assert BotStatus.RUNNING == "running"
        assert BotStatus.SUCCESS == "success"
        assert BotStatus.FAILED == "failed"
        assert BotStatus.STOPPED == "stopped"


class TestBotTask:
    """Тесты задачи бота."""
    
    def test_task_creation(self):
        """Тест создания задачи."""
        from src.bot.parallel import BotTask, BotPriority
        
        task = BotTask(
            priority=BotPriority.HIGH,
            account={'login': 'test_user', 'password': 'pass'},
            island_code='1234-5678-9012',
        )
        
        assert task.priority == BotPriority.HIGH
        assert task.login == 'test_user'
        assert task.island_code == '1234-5678-9012'
    
    def test_task_ordering(self):
        """Тест сортировки задач по приоритету."""
        from src.bot.parallel import BotTask, BotPriority
        
        task_low = BotTask(priority=BotPriority.LOW, account={'login': 'low'})
        task_high = BotTask(priority=BotPriority.HIGH, account={'login': 'high'})
        task_normal = BotTask(priority=BotPriority.NORMAL, account={'login': 'normal'})
        
        tasks = [task_low, task_high, task_normal]
        sorted_tasks = sorted(tasks)
        
        assert sorted_tasks[0].login == 'high'
        assert sorted_tasks[1].login == 'normal'
        assert sorted_tasks[2].login == 'low'
    
    def test_task_defaults(self):
        """Тест значений по умолчанию."""
        from src.bot.parallel import BotTask, BotPriority
        
        task = BotTask(
            priority=BotPriority.NORMAL,
            account={'login': 'test'},
        )
        
        assert task.proxy is None
        assert task.headless is True
        assert task.retry_count == 0
        assert task.max_retries == 3


class TestBotResult:
    """Тесты результата бота."""
    
    def test_result_creation(self):
        """Тест создания результата."""
        from src.bot.parallel import BotResult
        
        result = BotResult(
            login='test',
            success=True,
            started_at=100.0,
            finished_at=200.0,
        )
        
        assert result.login == 'test'
        assert result.success is True
        assert result.duration == 100.0
    
    def test_result_with_error(self):
        """Тест результата с ошибкой."""
        from src.bot.parallel import BotResult
        
        result = BotResult(
            login='test',
            success=False,
            started_at=100.0,
            finished_at=150.0,
            error="Connection failed",
        )
        
        assert result.success is False
        assert result.error == "Connection failed"
        assert result.duration == 50.0


class TestPoolStats:
    """Тесты статистики пула."""
    
    def test_stats_defaults(self):
        """Тест значений по умолчанию."""
        from src.bot.parallel import PoolStats
        
        stats = PoolStats()
        
        assert stats.total_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 0
        assert stats.active_workers == 0
        assert stats.average_duration == 0.0


class TestBotWorkerPool:
    """Тесты пула воркеров."""
    
    def test_pool_creation(self):
        """Тест создания пула."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=3)
        
        assert pool.max_workers == 3
        assert pool._executor is None
    
    def test_pool_start_stop(self):
        """Тест запуска и остановки пула."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=2)
        
        pool.start()
        assert pool._executor is not None
        
        pool.stop(wait=False)
        assert pool._executor is None
    
    def test_pool_submit(self):
        """Тест добавления задачи."""
        from src.bot.parallel import BotWorkerPool, BotStatus
        
        pool = BotWorkerPool(max_workers=2)
        
        login = pool.submit(
            account={'login': 'test_user', 'password': 'pass'},
            island_code='1234',
        )
        
        assert login == 'test_user'
        assert pool.get_status(login) == BotStatus.QUEUED
    
    def test_pool_submit_many(self):
        """Тест добавления нескольких задач."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=2)
        
        accounts = [
            {'login': 'user1', 'password': 'pass1'},
            {'login': 'user2', 'password': 'pass2'},
            {'login': 'user3', 'password': 'pass3'},
        ]
        
        logins = pool.submit_many(accounts, island_code='1234')
        
        assert len(logins) == 3
        assert 'user1' in logins
        assert 'user2' in logins
        assert 'user3' in logins
    
    def test_pool_get_stats(self):
        """Тест получения статистики."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=2)
        
        stats = pool.get_stats()
        
        assert stats.total_tasks == 0
        assert stats.queued_tasks == 0
    
    def test_pool_cancel(self):
        """Тест отмены задачи."""
        from src.bot.parallel import BotWorkerPool, BotStatus
        
        pool = BotWorkerPool(max_workers=2)
        
        login = pool.submit(
            account={'login': 'cancel_test', 'password': 'pass'},
        )
        
        # Отмена до запуска невозможна через cancel (нет future)
        result = pool.cancel(login)
        # Бот ещё не запущен, поэтому False
        assert result is False
    
    def test_pool_cancel_all(self):
        """Тест отмены всех задач."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=2)
        
        pool.submit(account={'login': 'user1', 'password': 'p1'})
        pool.submit(account={'login': 'user2', 'password': 'p2'})
        
        pool.cancel_all()
        
        stats = pool.get_stats()
        assert stats.queued_tasks == 0
    
    def test_pool_status_callback(self):
        """Тест колбэка статуса."""
        from src.bot.parallel import BotWorkerPool
        
        callback = Mock()
        pool = BotWorkerPool(max_workers=2, status_callback=callback)
        
        pool.submit(account={'login': 'callback_test', 'password': 'pass'})
        
        # Проверяем что колбэк был вызван
        callback.assert_called()


class TestBotWorkerPoolIntegration:
    """Интеграционные тесты пула."""
    
    def test_pool_with_settings(self):
        """Тест пула с настройками."""
        from src.bot.parallel import BotWorkerPool
        
        settings = {
            'island_code': 'TEST-1234',
            'headless': True,
            'time_on_island_min': 10,
        }
        
        pool = BotWorkerPool(max_workers=2, settings=settings)
        
        assert pool.settings == settings
    
    def test_multiple_pools(self):
        """Тест нескольких пулов."""
        from src.bot.parallel import BotWorkerPool
        
        pool1 = BotWorkerPool(max_workers=2)
        pool2 = BotWorkerPool(max_workers=3)
        
        assert pool1.max_workers == 2
        assert pool2.max_workers == 3
        
        # Разные пулы - разное состояние
        pool1.submit(account={'login': 'p1_user', 'password': 'p'})
        
        assert pool1.get_stats().total_tasks == 1
        assert pool2.get_stats().total_tasks == 0


class TestAsyncBotPool:
    """Тесты асинхронного пула."""
    
    def test_async_pool_creation(self):
        """Тест создания асинхронного пула."""
        from src.bot.parallel import AsyncBotPool
        
        pool = AsyncBotPool(max_concurrent=5)
        
        assert pool.max_concurrent == 5
        assert pool._shutdown is False
    
    @pytest.mark.asyncio
    async def test_async_pool_submit(self):
        """Тест добавления задачи в асинхронный пул."""
        from src.bot.parallel import AsyncBotPool
        
        pool = AsyncBotPool(max_concurrent=2)
        
        login = await pool.submit(
            account={'login': 'async_user', 'password': 'pass'},
        )
        
        assert login == 'async_user'
        assert 'async_user' in pool._tasks
    
    @pytest.mark.asyncio
    async def test_async_pool_cancel_all(self):
        """Тест отмены всех задач."""
        from src.bot.parallel import AsyncBotPool
        
        pool = AsyncBotPool(max_concurrent=2)
        
        await pool.submit(account={'login': 'u1', 'password': 'p'})
        await pool.submit(account={'login': 'u2', 'password': 'p'})
        
        await pool.cancel_all()
        
        assert pool._shutdown is True
        assert len(pool._tasks) == 0


class TestHelperFunctions:
    """Тесты вспомогательных функций."""
    
    def test_run_bots_parallel_function_exists(self):
        """Тест существования функции."""
        from src.bot.parallel import run_bots_parallel
        
        assert callable(run_bots_parallel)
    
    def test_run_bots_async_function_exists(self):
        """Тест существования асинхронной функции."""
        from src.bot.parallel import run_bots_async
        
        assert asyncio.iscoroutinefunction(run_bots_async)


class TestBotMetricsTracking:
    """Тесты отслеживания метрик."""
    
    def test_metrics_in_result(self):
        """Тест метрик в результате."""
        from src.bot.parallel import BotResult
        
        result = BotResult(
            login='metrics_test',
            success=True,
            started_at=time.time() - 60,
            finished_at=time.time(),
            retry_count=1,
        )
        
        assert result.retry_count == 1
        assert result.duration >= 60
    
    def test_pool_tracks_retries(self):
        """Тест отслеживания ретраев в пуле."""
        from src.bot.parallel import BotWorkerPool, BotTask, BotPriority
        
        pool = BotWorkerPool(max_workers=2)
        
        task = BotTask(
            priority=BotPriority.NORMAL,
            account={'login': 'retry_test', 'password': 'p'},
            retry_count=2,
            max_retries=3,
        )
        
        assert task.retry_count == 2
        assert task.max_retries == 3


class TestEdgeCases:
    """Тесты граничных случаев."""
    
    def test_empty_account_login(self):
        """Тест пустого логина."""
        from src.bot.parallel import BotTask, BotPriority
        
        task = BotTask(
            priority=BotPriority.NORMAL,
            account={},
        )
        
        assert task.login == 'unknown'
    
    def test_pool_double_start(self):
        """Тест двойного запуска пула."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=2)
        
        pool.start()
        executor1 = pool._executor
        
        pool.start()  # Повторный запуск
        executor2 = pool._executor
        
        # Должен остаться тот же executor
        assert executor1 is executor2
        
        pool.stop()
    
    def test_pool_stop_without_start(self):
        """Тест остановки незапущенного пула."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=2)
        
        # Не должно вызвать ошибку
        pool.stop()
    
    def test_get_status_nonexistent(self):
        """Тест получения статуса несуществующего бота."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=2)
        
        status = pool.get_status('nonexistent')
        
        assert status is None
    
    def test_get_result_nonexistent(self):
        """Тест получения результата несуществующего бота."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=2)
        
        result = pool.get_result('nonexistent')
        
        assert result is None


class TestThreadSafety:
    """Тесты потокобезопасности."""
    
    def test_concurrent_submit(self):
        """Тест параллельного добавления задач."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=5)
        results = []
        
        def submit_task(i):
            login = pool.submit(
                account={'login': f'user_{i}', 'password': 'p'},
            )
            results.append(login)
        
        threads = [threading.Thread(target=submit_task, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert pool.get_stats().total_tasks == 10
    
    def test_concurrent_get_stats(self):
        """Тест параллельного получения статистики."""
        from src.bot.parallel import BotWorkerPool
        
        pool = BotWorkerPool(max_workers=2)
        pool.submit(account={'login': 'test', 'password': 'p'})
        
        results = []
        
        def get_stats():
            stats = pool.get_stats()
            results.append(stats.total_tasks)
        
        threads = [threading.Thread(target=get_stats) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Все должны получить одинаковый результат
        assert all(r == 1 for r in results)
