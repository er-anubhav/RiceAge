import * as React from 'react';
import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

export interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  icon?: LucideIcon;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  subtitle,
  icon: Icon,
  badge,
  actions,
  className,
}) => {
  return (
    <div className={cn('mb-3 flex items-center justify-between border-b border-slate-200 pb-2', className)}>
      <div className="flex items-center space-x-2">
        {Icon && (
          <div className="flex h-6 w-6 items-center justify-center rounded bg-blue-50 text-blue-700">
            <Icon className="h-3.5 w-3.5" />
          </div>
        )}
        <div className="flex items-center space-x-2">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
            {title}
          </h2>
          {badge}
        </div>
        {subtitle && (
          <>
            <span className="text-slate-300">|</span>
            <span className="text-xs text-slate-500">{subtitle}</span>
          </>
        )}
      </div>

      {actions && <div className="flex items-center space-x-2">{actions}</div>}
    </div>
  );
};
