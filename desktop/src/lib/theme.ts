/**
 * Централизованная тема UI — единственный источник правды для всех стилей.
 *
 * Принципы:
 *  - Open/Closed: добавляй новые токены без изменения существующих
 *  - DRY: ни одна Tailwind-строка не дублируется в компонентах
 *  - Single Responsibility: theme.ts = стили, компоненты = логика + разметка
 *
 * Использование:
 *   import { theme } from '../lib/theme';
 *   <div className={theme.layout.page}>...</div>
 */

// ============================================================================
// Цветовые палитры — маппинг semantic-имя → Tailwind-классы
// ============================================================================

/** Цвета статистических карточек (иконка + фон) */
export const statColors = {
  brand:  'text-brand-400 bg-brand-500/10',
  green:  'text-green-400 bg-green-500/10',
  yellow: 'text-yellow-400 bg-yellow-500/10',
  red:    'text-red-400 bg-red-500/10',
} as const;

/** Цвета бейджей */
export const badgeColors = {
  gray:   'bg-surface-700 text-surface-200',
  green:  'bg-green-500/20 text-green-400',
  yellow: 'bg-yellow-500/20 text-yellow-400',
  red:    'bg-red-500/20 text-red-400',
  blue:   'bg-brand-500/20 text-brand-400',
} as const;

/** Варианты кнопок */
export const buttonVariants = {
  primary:   'bg-brand-600 hover:bg-brand-700 text-white',
  secondary: 'bg-surface-700 hover:bg-surface-600 text-white',
  danger:    'bg-red-600 hover:bg-red-700 text-white',
  ghost:     'hover:bg-surface-800 text-surface-200',
} as const;

/** Размеры кнопок */
export const buttonSizes = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
  lg: 'px-5 py-2.5 text-base',
} as const;

/** Цвета точек-индикаторов (инстансы) */
export const dotColors = {
  active:  'bg-green-400 animate-pulse-dot',
  warning: 'bg-yellow-400',
  idle:    'bg-surface-200',
} as const;

/** Цвета лог-строк по уровню */
export const logColors = {
  error:   'text-red-400',
  warning: 'text-yellow-400',
  debug:   'text-surface-200',
  success: 'text-green-400',
  default: 'text-slate-300',
} as const;

/** Цвета статуса подключения */
export const connectionColors = {
  online:  'text-green-400',
  offline: 'text-red-400',
} as const;

// ============================================================================
// Типы (экспортируемые для props)
// ============================================================================

export type StatColor = keyof typeof statColors;
export type BadgeColor = keyof typeof badgeColors;
export type ButtonVariant = keyof typeof buttonVariants;
export type ButtonSize = keyof typeof buttonSizes;

// ============================================================================
// Компонентные стили — структурированные классы для каждого компонента
// ============================================================================

export const theme = {
  // --- Layout ---
  layout: {
    root: 'flex h-screen bg-surface-950',
    page: 'p-6 animate-fade-in',
    pageFullHeight: 'p-6 h-full flex flex-col animate-fade-in',
    center: 'flex items-center justify-center h-full',
    grid2: 'grid grid-cols-1 lg:grid-cols-2 gap-4',
    grid3: 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4',
    grid4: 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6',
    main: 'flex-1 overflow-y-auto',
  },

  // --- Sidebar ---
  sidebar: {
    root: 'w-56 bg-surface-900 border-r border-surface-700 flex flex-col',
    logo: 'p-4 border-b border-surface-700',
    logoTitle: 'text-xl font-bold text-brand-400 tracking-tight',
    logoSubtitle: 'text-xs text-surface-200 mt-0.5',
    nav: 'flex-1 py-2 px-2 space-y-0.5',
    navItem: 'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
    navItemActive: 'bg-brand-600/20 text-brand-400',
    navItemInactive: 'text-surface-200 hover:bg-surface-800 hover:text-white',
    footer: 'p-3 border-t border-surface-700',
    footerContent: 'flex items-center gap-2 text-xs',
  },

  // --- Card ---
  card: {
    base: 'bg-surface-900 border border-surface-700 rounded-xl p-5',
    hover: 'card-hover',
    noPadding: '!p-0',
    title: 'text-lg font-semibold text-white mb-4',
  },

  // --- PageHeader ---
  header: {
    root: 'flex items-center justify-between mb-6',
    title: 'text-2xl font-bold text-white',
    subtitle: 'text-sm text-surface-200 mt-0.5',
    actions: 'flex items-center gap-2',
  },

  // --- Button ---
  button: {
    base: 'inline-flex items-center gap-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed',
  },

  // --- Badge ---
  badge: {
    base: 'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
  },

  // --- StatCard ---
  stat: {
    wrapper: 'flex items-center gap-4',
    iconBox: 'p-3 rounded-lg',
    value: 'text-2xl font-bold text-white',
    label: 'text-sm text-surface-200',
  },

  // --- Input ---
  input: {
    base: 'px-3 py-2 bg-surface-800 border border-surface-700 rounded-lg text-sm text-white placeholder:text-surface-200 focus:outline-none focus:border-brand-500',
    full: 'flex-1 px-3 py-2 bg-surface-800 border border-surface-700 rounded-lg text-sm text-white placeholder:text-surface-200 focus:outline-none focus:border-brand-500',
    textarea: 'w-full px-3 py-2 bg-surface-800 border border-surface-700 rounded-lg text-sm text-white font-mono placeholder:text-surface-200 focus:outline-none focus:border-brand-500 resize-y',
  },

  // --- EmptyState ---
  empty: {
    root: 'flex flex-col items-center justify-center py-16 text-center',
    icon: 'mb-4 text-surface-200',
    title: 'text-lg font-medium text-white mb-1',
    description: 'text-sm text-surface-200 max-w-md',
    action: 'mt-4',
  },

  // --- SettingField ---
  setting: {
    root: 'flex items-start justify-between gap-4',
    info: 'flex-1 min-w-0',
    label: 'text-sm font-medium text-white',
    description: 'text-xs text-surface-200 mt-0.5',
    control: 'shrink-0',
  },

  // --- Instance row ---
  instance: {
    row: 'flex items-center justify-between p-3 bg-surface-800 rounded-lg',
    name: 'text-sm font-medium text-white',
    state: 'text-xs text-surface-200 capitalize',
    dot: 'w-2 h-2 rounded-full',
    meta: 'text-xs text-surface-200 mb-4 space-y-1',
    actions: 'flex items-center gap-2',
    cardHeader: 'flex items-center justify-between mb-3',
    cardHeaderLeft: 'flex items-center gap-2',
  },

  // --- Account row ---
  account: {
    row: 'flex items-center justify-between py-3 first:pt-0 last:pb-0',
    email: 'text-sm font-medium text-white',
    password: 'text-xs text-surface-200 font-mono',
    divider: 'divide-y divide-surface-700',
  },

  // --- Logs ---
  logs: {
    container: 'h-full overflow-y-auto p-4 font-mono text-xs leading-relaxed',
    line: 'py-0.5',
    emptyRoot: 'flex flex-col items-center justify-center h-full text-surface-200',
  },

  // --- Event item (Dashboard) ---
  event: {
    list: 'space-y-1.5 max-h-64 overflow-y-auto',
    item: 'text-xs text-surface-200 font-mono px-2 py-1 bg-surface-800 rounded',
  },

  // --- Spinner ---
  spinner: {
    root: 'animate-spin text-brand-400',
  },

  // --- Misc text ---
  text: {
    muted: 'text-sm text-surface-200',
    brand: 'text-brand-400',
    white: 'text-white',
    red: 'text-red-400',
    mono: 'font-mono',
  },
} as const;

// ============================================================================
// Хелпер-функции — маппинг данных → классы
// ============================================================================

/** Цвет точки-индикатора инстанса */
export function getInstanceDotColor(farmState: string, status: string): string {
  if (farmState === 'farming') return dotColors.active;
  if (status === 'running') return dotColors.warning;
  return dotColors.idle;
}

/** Цвет бейджа по состоянию фарма */
export function getFarmBadgeColor(state: string): BadgeColor {
  if (state === 'farming' || state === 'running_macro') return 'green';
  if (state === 'launching' || state === 'loading') return 'yellow';
  if (state === 'error') return 'red';
  return 'gray';
}

/** Цвет строки лога по содержимому */
export function getLogLineColor(line: string): string {
  if (line.includes('ERROR') || line.includes('CRITICAL')) return logColors.error;
  if (line.includes('WARNING')) return logColors.warning;
  if (line.includes('DEBUG')) return logColors.debug;
  if (line.includes('SUCCESS') || line.includes('✓')) return logColors.success;
  return logColors.default;
}

/** Класс для NavLink (active/inactive) */
export function getNavLinkClass(isActive: boolean): string {
  return `${theme.sidebar.navItem} ${
    isActive ? theme.sidebar.navItemActive : theme.sidebar.navItemInactive
  }`;
}

/** Безопасное приведение значения для рендера (защита от объектов) */
export function safeDisplayValue(value: unknown): string | number {
  if (typeof value === 'string' || typeof value === 'number') return value;
  return '—';
}
