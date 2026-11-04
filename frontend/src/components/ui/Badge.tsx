import * as React from 'react';
import { cn } from '@/lib/utils';

export type BadgeVariant =
  | 'default'
  | 'pass'
  | 'warning'
  | 'danger'
  | 'info'
  | 'neutral'
  | 'indeterminate';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: 'sm' | 'md';
}

export const Badge: React.FC<BadgeProps> = ({
  className,
  variant = 'default',
  size = 'md',
  children,
  ...props
}) => {
  const variantStyles: Record<BadgeVariant, string> = {
    default: 'bg-slate-100 text-slate-700 border-slate-200',
    pass: 'bg-emerald-50 text-emerald-800 border-emerald-300 font-medium',
    warning: 'bg-amber-50 text-amber-800 border-amber-300 font-medium',
    danger: 'bg-rose-50 text-rose-800 border-rose-300 font-medium',
    info: 'bg-sky-50 text-sky-800 border-sky-300 font-medium',
    neutral: 'bg-slate-100 text-slate-600 border-slate-200',
    indeterminate: 'bg-purple-50 text-purple-800 border-purple-300 font-medium',
  };

  const sizeStyles = {
    sm: 'px-1.5 py-0.2 text-[10px] leading-tight',
    md: 'px-2 py-0.5 text-xs leading-none',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center rounded-sm border font-mono tracking-tight transition-colors',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};
