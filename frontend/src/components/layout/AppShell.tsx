import * as React from 'react';
import { Outlet, useSearchParams } from 'react-router-dom';
import { Header } from './Header';

export const AppShell: React.FC = () => {
  const [searchParams] = useSearchParams();
  const sampleParam = searchParams.get('sample');
  const [liveTimestamp, setLiveTimestamp] = React.useState<string | null>(null);

  React.useEffect(() => {
    let active = true;
    const fetchLive = async () => {
      try {
        const res = await fetch('/api/results');
        if (res.ok) {
          const data = await res.json();
          if (data && active) {
            const ts =
              data.timestamp_str ||
              (data.timestamp ? new Date(data.timestamp * 1000).toLocaleString() : null);
            if (ts) setLiveTimestamp(ts);
          }
        }
      } catch {
        // Ignore network errors in standalone mode
      }
    };
    fetchLive();
    const interval = setInterval(fetchLive, 3000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const defaultTimestamp =
    sampleParam === 'SAMPLE_MIXED_01' ? '12/10/2026 12:55:48' : '12/10/2026 14:30:22';
  const recordedTimestamp = liveTimestamp || defaultTimestamp;

  return (
    <div className="flex min-h-screen w-screen flex-col overflow-x-hidden bg-[#f4f6f8] text-slate-800 antialiased font-sans">
      {/* Sticky Top Header */}
      <Header recordedTimestamp={recordedTimestamp} />

      {/* Scrollable Content Body */}
      <main className="flex-1 p-4 md:p-6 bg-[#f4f6f8]">
        <div className="mx-auto max-w-7xl">
          <Outlet />
        </div>
      </main>
    </div>
  );
};
