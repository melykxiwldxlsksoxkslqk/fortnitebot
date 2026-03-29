/**
 * Библиотека переиспользуемых UI-компонентов.
 *
 * Все стили читаются из theme.ts — здесь нет ни одной хардкод-строки Tailwind.
 * Компоненты содержат только структуру (JSX) и минимальную логику.
 */

import type { ReactNode } from 'react';
import {
  theme,
  statColors,
  badgeColors,
  buttonVariants,
  buttonSizes,
  safeDisplayValue,
} from '../lib/theme';
import type { StatColor, BadgeColor, ButtonVariant, ButtonSize } from '../lib/theme';

// Реэкспорт типов для обратной совместимости
export type { StatColor, BadgeColor, ButtonVariant, ButtonSize };

// ============================================================================
// PageHeader
// ============================================================================

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className={theme.header.root}>
      <div>
        <h2 className={theme.header.title}>{title}</h2>
        {subtitle && <p className={theme.header.subtitle}>{subtitle}</p>}
      </div>
      {actions && <div className={theme.header.actions}>{actions}</div>}
    </div>
  );
}

// ============================================================================
// Card
// ============================================================================

export function Card({
  children,
  className = '',
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`${theme.card.base} ${className}`}>
      {children}
    </div>
  );
}

// ============================================================================
// StatCard
// ============================================================================

export function StatCard({
  label,
  value,
  icon,
  color = 'brand',
}: {
  label: string;
  value: string | number;
  icon: ReactNode;
  color?: StatColor;
}) {
  return (
    <Card className={theme.card.hover}>
      <div className={theme.stat.wrapper}>
        <div className={`${theme.stat.iconBox} ${statColors[color]}`}>{icon}</div>
        <div>
          <p className={theme.stat.value}>{safeDisplayValue(value)}</p>
          <p className={theme.stat.label}>{label}</p>
        </div>
      </div>
    </Card>
  );
}

// ============================================================================
// Button
// ============================================================================

export function Button({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  disabled = false,
  className = '',
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${theme.button.base} ${buttonVariants[variant]} ${buttonSizes[size]} ${className}`}
    >
      {children}
    </button>
  );
}

// ============================================================================
// Badge
// ============================================================================

export function Badge({
  children,
  color = 'gray',
}: {
  children: ReactNode;
  color?: BadgeColor;
}) {
  return (
    <span className={`${theme.badge.base} ${badgeColors[color]}`}>
      {children}
    </span>
  );
}

// ============================================================================
// Spinner
// ============================================================================

export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <svg
      className={theme.spinner.root}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

// ============================================================================
// EmptyState
// ============================================================================

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className={theme.empty.root}>
      {icon && <div className={theme.empty.icon}>{icon}</div>}
      <h3 className={theme.empty.title}>{title}</h3>
      {description && <p className={theme.empty.description}>{description}</p>}
      {action && <div className={theme.empty.action}>{action}</div>}
    </div>
  );
}

// ============================================================================
// SettingField
// ============================================================================

export function SettingField({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className={theme.setting.root}>
      <div className={theme.setting.info}>
        <label className={theme.setting.label}>{label}</label>
        {description && <p className={theme.setting.description}>{description}</p>}
      </div>
      <div className={theme.setting.control}>{children}</div>
    </div>
  );
}
