import * as React from 'react';
import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'success';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  icon?: LucideIcon;
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      icon: Icon,
      loading = false,
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    const variantStyles = {
      primary:
        'bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 border-blue-600 shadow-none',
      secondary:
        'bg-slate-100 text-slate-800 hover:bg-slate-200 active:bg-slate-300 border-slate-300',
      outline:
        'bg-white text-slate-700 hover:bg-slate-50 active:bg-slate-100 border-slate-300',
      ghost:
        'bg-transparent text-slate-600 hover:bg-slate-100 active:bg-slate-200 border-transparent',
      danger:
        'bg-rose-600 text-white hover:bg-rose-700 active:bg-rose-800 border-rose-600',
      success:
        'bg-emerald-600 text-white hover:bg-emerald-700 active:bg-emerald-800 border-emerald-600',
    };

    const sizeStyles = {
      sm: 'h-7 px-2.5 text-xs',
      md: 'h-8 px-3 text-xs',
      lg: 'h-9 px-4 text-sm',
      icon: 'h-8 w-8 p-0 text-xs justify-center',
    };

    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={cn(
          'inline-flex items-center justify-center font-medium rounded border transition-colors',
          'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500',
          'disabled:pointer-events-none disabled:opacity-50 select-none',
          variantStyles[variant],
          sizeStyles[size],
          className
        )}
        {...props}
      >
        {loading ? (
          <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent mr-1.5" />
        ) : (
          Icon && <Icon className={cn('h-3.5 w-3.5', children ? 'mr-1.5' : '')} />
        )}
        {children}
      </button>
    );
  }
);
Button.displayName = 'Button';
