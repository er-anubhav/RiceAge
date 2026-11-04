import * as React from 'react';
import { Link } from 'react-router-dom';
import {
  History,
  Printer,
  Camera,
  CheckSquare,
  Square,
  X,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import { api } from '@/services/api';
import type { ScanHistoryRecord } from '@/types/analysis';

export interface HeaderProps {
  recordedTimestamp?: string | null;
}

export const Header: React.FC<HeaderProps> = ({ recordedTimestamp }) => {
  const [backendOnline, setBackendOnline] = React.useState<boolean | null>(null);
  const [accepted, setAccepted] = React.useState<boolean>(true);

  // Modals for History and Scan
  const [showHistoryModal, setShowHistoryModal] = React.useState<boolean>(false);
  const [showScanModal, setShowScanModal] = React.useState<boolean>(false);
  const [historyList, setHistoryList] = React.useState<ScanHistoryRecord[]>([]);
  const [selectedScanFile, setSelectedScanFile] = React.useState<File | null>(null);
  const [scanRunning, setScanRunning] = React.useState<boolean>(false);
  const [scanError, setScanError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      try {
        const isUp = await api.checkBackendHealth();
        if (isMounted) setBackendOnline(isUp);
      } catch {
        if (isMounted) setBackendOnline(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Fetch scan history on mount
  React.useEffect(() => {
    api
      .getScanHistory()
      .then((records) => {
        setHistoryList(records);
      })
      .catch(() => {
        // Standby fallback
      });
  }, []);

  // Trigger optical scan via API or hardware simulation
  const handleTriggerScan = async () => {
    setScanRunning(true);
    setScanError(null);

    if (selectedScanFile) {
      try {
        const sampleId = `SMP-${Date.now().toString().slice(-6)}`;
        const res = await api.analyzeScan(selectedScanFile, sampleId);
        setScanRunning(false);
        setShowScanModal(false);
        const resolvedId = res.sample_id || res.sampleId || 'SAMPLE_M4_01';
        window.location.href = `/?sample=${resolvedId}`;
      } catch (err: unknown) {
        setScanRunning(false);
        const msg = err instanceof Error ? err.message : 'Hardware scan acquisition failed';
        setScanError(msg);
      }
    } else {
      try {
        const res = await fetch('/api/scan', { method: 'POST' });
        if (!res.ok) {
          throw new Error(`Scan failed (${res.status})`);
        }
        const data = await res.json();
        setScanRunning(false);
        setShowScanModal(false);
        const resolvedId = data.serial || data.sample_id || 'Sample-0001';
        window.location.href = `/?sample=${resolvedId}`;
      } catch (err: unknown) {
        setScanRunning(false);
        const msg = err instanceof Error ? err.message : 'Hardware scan acquisition failed';
        setScanError(msg);
      }
    }
  };


  // Determine current active recorded timestamp label
  const timestampLabel = recordedTimestamp || '12/10/2026 14:30:22';

  return (
    <>
      <header className="sticky top-0 z-30 flex h-14 w-full items-center justify-between border-b border-slate-300 bg-white px-4 shadow-2xs select-none">
        {/* Left: Branding & Active Recorded Timestamp */}
        <div className="flex items-center space-x-4">
          {/* Quality Analyzer Title (Big, No Bold) */}
          <Link to="/" className="flex items-center hover:opacity-85 transition-opacity">
            <span className="text-xl md:text-2xl font-normal text-slate-800 tracking-tight">
              Quality Analyzer
            </span>
          </Link>

          {/* Divider */}
          <div className="h-5 w-px bg-slate-300" />

          {/* Active Recorded Timestamp Context Badge */}
          <div className="flex items-center space-x-1.5 text-xs">
            <span className="text-slate-500 font-medium">Recorded:</span>
            <span className="font-bold text-slate-800 font-mono tracking-tight bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              {timestampLabel}
            </span>
          </div>
        </div>

        {/* Right: Accepted Checkbox + 3 Action Buttons (History, Print, Scan) + Backend Status */}
        <div className="flex items-center space-x-3">
          {/* Accepted checkbox */}
          <div
            onClick={() => setAccepted(!accepted)}
            className="flex items-center space-x-1.5 cursor-pointer text-xs font-semibold text-slate-700 select-none hover:text-slate-900"
          >
            <span>Accepted</span>
            {accepted ? (
              <CheckSquare className="h-4 w-4 text-emerald-600" />
            ) : (
              <Square className="h-4 w-4 text-slate-400" />
            )}
          </div>

          {/* Exactly 3 Action Buttons: History, Print, and Scan */}
          <div className="flex items-center space-x-1.5">
            {/* Button 1: History */}
            <button
              onClick={() => setShowHistoryModal(true)}
              className="flex items-center space-x-1.5 h-7 px-3 rounded-full border border-rose-300 text-rose-600 hover:bg-rose-50 text-xs font-semibold transition-colors shadow-2xs"
              title="View Scan History"
            >
              <History className="h-3.5 w-3.5" />
              <span>History</span>
            </button>

            {/* Button 2: Print */}
            <button
              onClick={() => window.print()}
              className="flex items-center space-x-1.5 h-7 px-3 rounded-full border border-rose-300 text-rose-600 hover:bg-rose-50 text-xs font-semibold transition-colors shadow-2xs"
              title="Print Certificate & Report"
            >
              <Printer className="h-3.5 w-3.5" />
              <span>Print</span>
            </button>

            {/* Button 3: Scan */}
            <button
              onClick={() => setShowScanModal(true)}
              className="flex items-center space-x-1.5 h-7 px-3.5 rounded-full border border-rose-600 bg-rose-600 text-white hover:bg-rose-700 text-xs font-bold transition-colors shadow-2xs"
              title="Initiate Optical Scan"
            >
              <Camera className="h-3.5 w-3.5" />
              <span>Scan</span>
            </button>
          </div>

          {/* Backend connectivity indicator */}
          <div
            className="hidden sm:flex items-center text-[10px] font-mono px-1.5 py-0.5 rounded border border-slate-200 bg-slate-50"
            title={backendOnline ? 'Backend Online' : 'Backend Standby'}
          >
            <span
              className={`h-2 w-2 rounded-full mr-1 ${
                backendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span className="text-slate-600">{backendOnline ? 'ONLINE' : 'STANDBY'}</span>
          </div>
        </div>
      </header>

      {/* History Modal Dialog */}
      {showHistoryModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4 animate-in fade-in duration-150">
          <div className="w-full max-w-lg rounded-lg border border-slate-300 bg-white p-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="flex items-center space-x-2">
                <History className="h-5 w-5 text-rose-600" />
                <h2 className="text-base font-bold text-slate-800">Inspection Scan History</h2>
              </div>
              <button
                onClick={() => setShowHistoryModal(false)}
                className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 transition-colors"
                aria-label="Close History Modal"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-4 space-y-2 max-h-80 overflow-y-auto pr-1">
              {/* Benchmark sample 1 */}
              <div
                onClick={() => {
                  window.location.href = '/?sample=SAMPLE_M4_01';
                  setShowHistoryModal(false);
                }}
                className="flex cursor-pointer items-center justify-between rounded border border-slate-200 p-3 hover:border-sky-500 hover:bg-sky-50/50 transition-colors"
              >
                <div>
                  <div className="font-mono text-xs font-bold text-slate-800">SAMPLE_M4_01</div>
                  <div className="text-[11px] text-slate-500">
                    Recorded: 12/10/2026 14:30:22 · 986 Grains
                  </div>
                </div>
                <span className="rounded bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-800">
                  GRADE 1
                </span>
              </div>

              {/* Benchmark sample 2 */}
              <div
                onClick={() => {
                  window.location.href = '/?sample=SAMPLE_MIXED_01';
                  setShowHistoryModal(false);
                }}
                className="flex cursor-pointer items-center justify-between rounded border border-slate-200 p-3 hover:border-sky-500 hover:bg-sky-50/50 transition-colors"
              >
                <div>
                  <div className="font-mono text-xs font-bold text-slate-800">SAMPLE_MIXED_01</div>
                  <div className="text-[11px] text-slate-500">
                    Recorded: 12/10/2026 12:55:48 · 24 Grains
                  </div>
                </div>
                <span className="rounded bg-rose-100 px-2 py-0.5 text-[10px] font-bold text-rose-800">
                  SUBSTANDARD
                </span>
              </div>

              {/* Server scans */}
              {historyList.map((item) => {
                const id = item.sample_id || item.sampleId || item.id;
                const date = item.analysis_date || item.analysisDate || 'Recent';
                const grains = item.total_rice_grains ?? item.grainCount ?? 950;

                return (
                  <div
                    key={id}
                    onClick={() => {
                      window.location.href = `/?sample=${id}`;
                      setShowHistoryModal(false);
                    }}
                    className="flex cursor-pointer items-center justify-between rounded border border-slate-200 p-3 hover:border-sky-500 hover:bg-sky-50/50 transition-colors"
                  >
                    <div>
                      <div className="font-mono text-xs font-bold text-slate-800">{id}</div>
                      <div className="text-[11px] text-slate-500">
                        Recorded: {date} · {grains} Grains
                      </div>
                    </div>
                    <span className="rounded bg-slate-100 px-2 py-0.5 text-[10px] font-bold text-slate-700">
                      INSPECTED
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="mt-4 flex justify-end border-t border-slate-200 pt-3">
              <button
                onClick={() => setShowHistoryModal(false)}
                className="rounded border border-slate-300 bg-white px-3.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Optical Scan Modal Dialog */}
      {showScanModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4 animate-in fade-in duration-150">
          <div className="w-full max-w-md rounded-lg border border-slate-300 bg-white p-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="flex items-center space-x-2">
                <Camera className="h-5 w-5 text-rose-600" />
                <h2 className="text-base font-bold text-slate-800">Optical Grain Scan</h2>
              </div>
              <button
                onClick={() => setShowScanModal(false)}
                className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 transition-colors"
                aria-label="Close Scan Modal"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-4 space-y-4">
              <div className="rounded border border-blue-200 bg-blue-50/60 p-3 text-xs text-blue-900">
                <div className="font-bold flex items-center space-x-1.5">
                  <CheckCircle2 className="h-4 w-4 text-blue-600" />
                  <span>Optical Flatbed Ready (600 DPI)</span>
                </div>
                <p className="mt-1 text-[11px] text-blue-800">
                  Ensure 20g of rice kernels are evenly distributed across the scan tray without grain overlapping.
                </p>
              </div>

              {/* Upload image or hardware trigger */}
              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-slate-700">
                  Scan File / Image (Optional)
                </label>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.tiff,.tif,.bmp"
                  className="w-full text-xs text-slate-600 file:mr-2 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-rose-50 file:text-rose-700 hover:file:bg-rose-100 transition-colors cursor-pointer"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setSelectedScanFile(e.target.files[0]);
                    }
                  }}
                />
                <p className="text-[10px] text-slate-400">
                  Leave blank to trigger standard hardware scan capture directly.
                </p>
              </div>

              {scanError && (
                <div className="rounded border border-rose-200 bg-rose-50 p-2 text-xs text-rose-800 flex items-start space-x-1.5">
                  <AlertCircle className="h-4 w-4 text-rose-600 mt-0.5 shrink-0" />
                  <span>{scanError}</span>
                </div>
              )}

              {scanRunning ? (
                <div className="flex items-center justify-center space-x-2 py-3 text-xs font-semibold text-rose-600">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-rose-600 border-t-transparent" />
                  <span>Acquiring optical capture & inferencing M4...</span>
                </div>
              ) : (
                <div className="flex space-x-2 pt-2">
                  <button
                    onClick={handleTriggerScan}
                    className="flex-1 rounded bg-rose-600 py-2 text-xs font-bold text-white hover:bg-rose-700 transition-colors shadow-2xs"
                  >
                    Start Scan
                  </button>
                  <button
                    onClick={() => setShowScanModal(false)}
                    className="rounded border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};
