import * as React from 'react';
import { useSearchParams } from 'react-router-dom';

interface ParameterRow {
  name: string;
  denominator: string;
  limit: string;
  resultPct: string;
  resultCountAndGms: string;
  status: 'acceptable' | 'warning' | 'substandard';
}

interface SampleDetail {
  sampleId: string;
  sampleName: string;
  station: string;
  timestamp: string;
  scanMode: string;
  totalObjects: number;
  totalGrains: number;
  totalForeignMatter: number;
  avgLengthMm: number;
  avgBreadthMm: number;
  disposition: 'GRADE_1' | 'SUBSTANDARD' | 'INDETERMINATE';
  rows: ParameterRow[];
}

// Default realistic sample representing a standard compliant laboratory inspection in RiceAge M4
const SAMPLE_M4_01: SampleDetail = {
  sampleId: 'SAMPLE_M4_01',
  sampleName: 'Milled White Rice (Standard Inspection)',
  station: 'RiceAge Optical Station 1',
  timestamp: '12/10/2026 14:30:22',
  scanMode: '600 DPI Calibrated Flatbed',
  totalObjects: 1024,
  totalGrains: 986,
  totalForeignMatter: 38,
  avgLengthMm: 6.452,
  avgBreadthMm: 2.115,
  disposition: 'GRADE_1',
  rows: [
    {
      name: 'Whole Grains (Head Rice)',
      denominator: 'count(Whole) / total_rice_grains',
      limit: 'Min 80.0%',
      resultPct: '82.25%',
      resultCountAndGms: '811 grains (16.45 gms)',
      status: 'acceptable',
    },
    {
      name: 'Broken Grains',
      denominator: 'count(Broken) / total_rice_grains',
      limit: 'Max 15.0%',
      resultPct: '14.75%',
      resultCountAndGms: '145 grains (2.95 gms)',
      status: 'acceptable',
    },
    {
      name: 'Chalky Grains',
      denominator: 'count(Chalky >= 50%) / total_rice_grains',
      limit: 'Max 3.0%',
      resultPct: '2.84%',
      resultCountAndGms: '28 grains (0.57 gms)',
      status: 'acceptable',
    },
    {
      name: 'Damaged Grains',
      denominator: 'count(Insect/Cavitation) / total_rice_grains',
      limit: 'Max 2.0%',
      resultPct: '1.22%',
      resultCountAndGms: '12 grains (0.24 gms)',
      status: 'acceptable',
    },
    {
      name: 'Discoloured Grains',
      denominator: 'count(Peck/Heat) / total_rice_grains',
      limit: 'Max 1.0%',
      resultPct: '0.91%',
      resultCountAndGms: '9 grains (0.18 gms)',
      status: 'acceptable',
    },
    {
      name: 'Foreign Matter',
      denominator: 'total_foreign_matter / total_detected_objects',
      limit: 'Max 0.5%',
      resultPct: '0.39%',
      resultCountAndGms: '4 particles (0.08 gms)',
      status: 'acceptable',
    },
  ],
};

// Real sample from outputs/reports/SAMPLE_MIXED_01_report.json in RiceAge repo
const SAMPLE_MIXED_01: SampleDetail = {
  sampleId: 'SAMPLE_MIXED_01',
  sampleName: 'Mixed Defect Test Tray',
  station: 'RiceAge M4 Benchmark Runner',
  timestamp: '12/10/2026 12:55:48',
  scanMode: 'High-Res Optical Flatbed',
  totalObjects: 32,
  totalGrains: 24,
  totalForeignMatter: 8,
  avgLengthMm: 5.120,
  avgBreadthMm: 2.340,
  disposition: 'SUBSTANDARD',
  rows: [
    {
      name: 'Whole Grains (Head Rice)',
      denominator: 'count(Whole) / total_rice_grains',
      limit: 'Min 80.0%',
      resultPct: '16.67%',
      resultCountAndGms: '4 grains (3.33 gms)',
      status: 'substandard',
    },
    {
      name: 'Broken Grains',
      denominator: 'count(Broken) / total_rice_grains',
      limit: 'Max 15.0%',
      resultPct: '83.33%',
      resultCountAndGms: '20 grains (16.67 gms)',
      status: 'substandard',
    },
    {
      name: 'Chalky Grains',
      denominator: 'count(Chalky >= 50%) / total_rice_grains',
      limit: 'Max 3.0%',
      resultPct: '25.00%',
      resultCountAndGms: '6 grains (5.00 gms)',
      status: 'substandard',
    },
    {
      name: 'Damaged Grains',
      denominator: 'count(Insect/Cavitation) / total_rice_grains',
      limit: 'Max 2.0%',
      resultPct: '4.17%',
      resultCountAndGms: '1 grain (0.83 gms)',
      status: 'substandard',
    },
    {
      name: 'Discoloured Grains',
      denominator: 'count(Peck/Heat) / total_rice_grains',
      limit: 'Max 1.0%',
      resultPct: '8.33%',
      resultCountAndGms: '2 grains (1.67 gms)',
      status: 'substandard',
    },
    {
      name: 'Foreign Matter',
      denominator: 'total_foreign_matter / total_detected_objects',
      limit: 'Max 0.5%',
      resultPct: '25.00%',
      resultCountAndGms: '8 particles (5.00 gms)',
      status: 'substandard',
    },
  ],
};

export const DashboardPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const sampleParam = searchParams.get('sample');
  const defaultSample = sampleParam === 'SAMPLE_MIXED_01' ? SAMPLE_MIXED_01 : SAMPLE_M4_01;
  const [currentSample, setCurrentSample] = React.useState<SampleDetail>(defaultSample);

  React.useEffect(() => {
    if (sampleParam === 'SAMPLE_MIXED_01') {
      setCurrentSample(SAMPLE_MIXED_01);
      return;
    }
    if (sampleParam === 'SAMPLE_M4_01') {
      setCurrentSample(SAMPLE_M4_01);
      return;
    }

    let active = true;
    const fetchLive = async () => {
      try {
        const targetUrl = sampleParam ? `/api/results/${encodeURIComponent(sampleParam)}` : '/api/results';
        const res = await fetch(targetUrl);
        if (res.ok) {
          const data = await res.json();

          if (data && data.rows && data.rows.length > 0 && active) {
            setCurrentSample({
              sampleId: data.serial || 'Sample-0001',
              sampleName: data.commodity || 'Raw Rice (Optical Scan)',
              station: 'RiceAge Optical Station 1',
              timestamp:
                data.timestamp_str ||
                (data.timestamp
                  ? new Date(data.timestamp * 1000).toLocaleString()
                  : '12/10/2026 14:30:22'),
              scanMode: '300 DPI Optical Capture',
              totalObjects: data.total_objects || data.total_grains || 0,
              totalGrains: data.total_grains || 0,
              totalForeignMatter: data.total_foreign_matter || 0,
              avgLengthMm: data.avg_length_mm || 6.452,
              avgBreadthMm: data.avg_breadth_mm || 2.115,
              disposition: data.disposition || 'GRADE_1',
              rows: data.rows,
            });
          }
        }
      } catch {
        // Fallback silently
      }
    };

    fetchLive();
    const interval = setInterval(fetchLive, 3000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [sampleParam]);

  const selectedSample = currentSample;

  return (
    <div className="space-y-3 font-sans">
      {/* 4 Colored Summary Cards in a row across the top */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
        {/* Card 1: Green Card (Commodity & Scan Specs) */}
        <div className="rounded bg-[#16a34a] p-3 text-white shadow-xs flex flex-col justify-between min-h-[96px]">
          <div className="font-mono text-xs font-semibold tracking-tight text-emerald-100">
            RAW Rice · 20 gms Tray
          </div>
          <div className="text-sm font-bold tracking-tight">
            {selectedSample.scanMode}
          </div>
          <div className="text-xs font-semibold uppercase tracking-wider text-emerald-100">
            <span>Calibrated Optical Tray</span>
          </div>
        </div>

        {/* Card 2: Red Card (Grain Count / Objects) */}
        <div className="rounded bg-[#dc2626] p-3 text-white shadow-xs flex flex-col justify-between min-h-[96px]">
          <div className="text-xs font-medium tracking-tight text-rose-100 flex items-center justify-between">
            <span>Grain Count</span>
            <span className="text-[10px] opacity-80">{selectedSample.totalObjects} Objects</span>
          </div>
          <div className="border-t border-rose-400/40 my-1" />
          <div className="text-2xl font-bold tracking-tight">
            {selectedSample.totalGrains.toLocaleString()} Grains
          </div>
        </div>

        {/* Card 3: Blue Card (Recorded Timestamp) */}
        <div className="rounded bg-[#0284c7] p-3 text-white shadow-xs flex flex-col justify-between min-h-[96px]">
          <div className="text-xs font-medium tracking-tight text-sky-100 underline decoration-sky-300 underline-offset-2 flex items-center justify-between">
            <span>Recorded Timestamp</span>
            <span className="text-[10px] opacity-80">{selectedSample.disposition}</span>
          </div>
          <div className="text-lg sm:text-xl font-bold tracking-tight mt-auto font-mono">
            {selectedSample.timestamp}
          </div>
        </div>

        {/* Card 4: Orange Card (Avg Length, Avg Breadth) */}
        <div className="rounded bg-[#ea580c] p-3 text-white shadow-xs flex flex-col justify-center space-y-1.5 min-h-[96px]">
          <div className="flex items-center justify-between text-xs font-semibold">
            <span className="text-amber-100">Avg Length</span>
            <span className="font-mono text-sm font-bold">
              {selectedSample.avgLengthMm.toFixed(3)} mm
            </span>
          </div>
          <div className="flex items-center justify-between text-xs font-semibold">
            <span className="text-amber-100">Avg Breadth</span>
            <span className="font-mono text-sm font-bold">
              {selectedSample.avgBreadthMm.toFixed(3)} mm
            </span>
          </div>
        </div>
      </div>

      {/* Main "Analysis Results" Table Card */}
      <div className="rounded border border-slate-300 bg-white shadow-2xs overflow-hidden">
        {/* Full-width Blue Banner Header */}
        <div className="bg-[#0284c7] text-white text-center font-bold text-sm py-2 uppercase tracking-wide select-none">
          <span>Analysis Results</span>
        </div>

        {/* Table Content */}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="bg-[#e9f1f7] text-slate-700 font-semibold border-b border-slate-300">
                <th className="py-2.5 px-4 text-left font-bold w-[42%]">
                  Parameter
                </th>
                <th className="py-2.5 px-4 text-center font-bold w-[18%] border-l border-slate-300">
                  Limits in %
                </th>
                <th className="py-2.5 px-4 text-center font-bold w-[18%] border-l border-slate-300">
                  Result(%)
                </th>
                <th className="py-2.5 px-4 text-center font-bold w-[22%] border-l border-slate-300">
                  Result(Count / gms)
                </th>
              </tr>
            </thead>
            <tbody>
              {selectedSample.rows.map((row, idx) => {
                const isAcceptable = row.status === 'acceptable';

                return (
                  <tr
                    key={idx}
                    className="border-b border-slate-200 hover:bg-slate-50/80 transition-colors"
                  >
                    {/* Parameter name */}
                    <td className="py-2 px-4 font-semibold text-sky-700">
                      <span className="hover:underline cursor-pointer">
                        {row.name}
                      </span>
                    </td>

                    {/* Limits in % */}
                    <td className="py-2 px-4 text-center font-mono text-slate-700 border-l border-slate-200 font-medium">
                      {row.limit}
                    </td>

                    {/* Result(%) - Pale Green background for acceptable values */}
                    <td
                      className={`py-2 px-4 text-center font-mono font-bold border-l border-slate-200 ${
                        isAcceptable
                          ? 'bg-[#dcfce7] text-emerald-900'
                          : 'bg-rose-100 text-rose-900'
                      }`}
                    >
                      {row.resultPct}
                    </td>

                    {/* Result(Count / gms) */}
                    <td
                      className={`py-2 px-4 text-center font-mono font-semibold border-l border-slate-200 ${
                        isAcceptable
                          ? 'bg-[#dcfce7] text-emerald-900'
                          : 'bg-rose-100 text-rose-900'
                      }`}
                    >
                      {row.resultCountAndGms}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
