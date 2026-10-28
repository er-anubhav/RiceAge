import * as React from 'react';
import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

export type SummaryCardTheme = 'teal' | 'coral' | 'blue' | 'orange' | 'slate' | 'emerald';

export interface SummaryCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  denominator?: string;
  icon?: LucideIcon;
  theme?: SummaryCardTheme;
  trend?: {
    value: string;
    isPositive?: boolean;
  };
  className?: string;
  onClick?: () => void;
}

export const SummaryCard: React.FC<SummaryCardProps> = ({
  title,
  value,
  subtitle,
  denominator,
  icon: Icon,
  theme = 'blue',
  trend,
  className,
  onClick,
}) => {
  const themeStyles: Record<
    SummaryCardTheme,
    { border: string; bg: string; iconBg: string; iconColor: string; accentBar: string }
  > = {
    teal: {
      border: 'border-teal-200',
      bg: 'bg-white',
      iconBg: 'bg-teal-50',
      iconColor: 'text-teal-700',
      accentBar: 'bg-teal-600',
    },
    coral: {
      border: 'border-rose-200',
      bg: 'bg-white',
      iconBg: 'bg-rose-50',
      iconColor: 'text-rose-700',
      accentBar: 'bg-rose-600',
    },
    blue: {
      border: 'border-blue-200',
      bg: 'bg-white',
      iconBg: 'bg-blue-50',
      iconColor: 'text-blue-700',
      accentBar: 'bg-blue-600',
    },
    orange: {
      border: 'border-amber-200',
      bg: 'bg-white',
      iconBg: 'bg-amber-50',
      iconColor: 'text-amber-700',
      accentBar: 'bg-amber-600',
    },
    slate: {
      border: 'border-slate-200',
      bg: 'bg-white',
      iconBg: 'bg-slate-100',
      iconColor: 'text-slate-700',
      accentBar: 'bg-slate-600',
    },
    emerald: {
      border: 'border-emerald-200',
      bg: 'bg-white',
      iconBg: 'bg-emerald-50',
      iconColor: 'text-emerald-700',
      accentBar: 'bg-emerald-600',
    },
  };

  const currentTheme = themeStyles[theme];

  return (
    <div
      onClick={onClick}
      className={cn(
        'relative overflow-hidden rounded border bg-white p-3 shadow-none transition-shadow',
        currentTheme.border,
        onClick && 'cursor-pointer hover:shadow-sm',
        className
      )}
    >
      {/* Top accent line */}
      <div className={cn('absolute left-0 top-0 h-1 w-full', currentTheme.accentBar)} />

      <div className="flex items-start justify-between">
        <div className="space-y-0.5">
          <p className="text-[11px] font-medium tracking-wide text-slate-500 uppercase">
            {title}
          </p>
          <div className="flex items-baseline space-x-2">
            <span className="font-mono text-2xl font-bold tracking-tight text-slate-900">
              {value}
            </span>
            {subtitle && (
              <span className="text-xs font-semibold text-slate-600">{subtitle}</span>
            )}
          </div>
        </div>

        {Icon && (
          <div
            className={cn(
              'flex h-8 w-8 items-center justify-center rounded border border-slate-100',
              currentTheme.iconBg,
              currentTheme.iconColor
            )}
          >
            <Icon className="h-4 w-4" />
          </div>
        )}
      </div>

      {(denominator || trend) && (
        <div className="mt-2 flex items-center justify-between border-t border-slate-100 pt-1.5 text-[11px] text-slate-500">
          {denominator && <span className="truncate">{denominator}</span>}
          {trend && (
            <span
              className={cn(
                'font-medium font-mono',
                trend.isPositive ? 'text-emerald-700' : 'text-rose-700'
              )}
            >
              {trend.value}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
