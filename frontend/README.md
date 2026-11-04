# RiceAge M5.0 — Laboratory Dashboard UI Foundation

The web-based frontend application for **RiceAge**, an optical computer-vision rice grain quality analysis and grading system.

The user interface adheres to a high-density **laboratory instrument aesthetic** inspired by commercial grain inspection workstations:
- Light grey workspace canvas (`#f4f6f8`)
- White content cards and data panels
- Dark navy-blue persistent left navigation sidebar (`#0f172a`)
- Blue section headers, indicators, and dividers
- Multi-tier laboratory summary cards (Teal, Coral-Red, Blue, Orange, Slate)
- Pale green result cells (`#ecfdf5` / `#10b981`) for normal/acceptable tolerance parameters
- Desktop-first layout optimized for 1366×768 and higher-resolution laboratory monitors with full responsive adaptation

---

## 1. Technology Stack

- **Core**: React 19, TypeScript
- **Bundler & Tooling**: Vite
- **Styling**: Tailwind CSS (with customized laboratory color tokens & compact table utility classes)
- **Icons**: Lucide React
- **Routing**: React Router (v7)
- **Data Visualization**: Recharts (Pie & Bar charts)
- **Architecture**: Decoupled typed API client layer (`src/services/api.ts`) connecting to the Python CV pipeline

---

## 2. Directory Structure

```
frontend/
├── dist/                      # Production build output
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── AppShell.tsx   # Persistent layout container
│   │   │   ├── Header.tsx     # Top telemetry header with breadcrumbs & API status
│   │   │   └── Sidebar.tsx    # Navy navigation sidebar with route links & badges
│   │   └── ui/
│   │       ├── Badge.tsx      # Laboratory status indicator badges
│   │       ├── Button.tsx     # Compact instrument buttons
│   │       ├── Card.tsx       # White laboratory panel primitives
│   │       ├── EmptyState.tsx # Honest empty-state component
│   │       ├── Input.tsx      # Compact text inputs and selects
│   │       ├── SectionHeader.tsx # Blue section title with divider
│   │       └── SummaryCard.tsx   # Color-accented metric indicator cards
│   ├── lib/
│   │   └── utils.ts           # Number, percentage, date and class utilities
│   ├── pages/
│   │   ├── AnalysisResultPage.tsx # Single scan certificate & image inspector
│   │   ├── AnalyzeScanPage.tsx    # Drag-and-drop scan ingestion & execution
│   │   ├── DashboardPage.tsx      # Overview summary, charts, recent scans
│   │   ├── DatasetExplorerPage.tsx # Ground-truth 119 image catalog
│   │   ├── ModelInformationPage.tsx # M4 model architecture, CV metrics, limitations
│   │   ├── QualityReportsPage.tsx   # Official inspection report registry
│   │   ├── ScanHistoryPage.tsx    # Chronological scan archive & filter
│   │   └── SettingsPage.tsx       # Sensor calibration & tolerance limits
│   ├── services/
│   │   └── api.ts             # Typed REST API service layer with fallbacks
│   ├── types/
│   │   └── analysis.ts        # TypeScript schemas mapped to M4 Python dataclasses
│   ├── App.tsx                # Client-side router configuration
│   ├── index.css              # Tailwind imports & lab table styles
│   └── main.tsx               # Application entry point
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

---

## 3. Installation & Development

### Prerequisites
- Node.js 18+ (tested on Node v22)
- npm 9+

### Install Dependencies
```bash
cd frontend
npm install
```

### Start Development Server
```bash
npm run dev
```
The application will launch at `http://localhost:5173/`.

### Run Production Build
```bash
npm run build
```
This executes `tsc -b` for strict type-checking followed by Vite's production bundling into `frontend/dist/`.

### Preview Production Build
```bash
npm run preview
```

---

## 4. Environment Variables

Create a `.env` or `.env.local` file in `frontend/`:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | Base URL of the RiceAge Python FastAPI inference backend | `http://localhost:8000` |

If the backend server is offline or unreachable, the frontend transparently presents a **DEV STANDBY** state without fabricating false predictions or mock scans.

---

## 5. Application Route Map

| Path | Screen Name | Description |
| :--- | :--- | :--- |
| `/` | **Dashboard** | Aggregated summary metrics, Whole vs Broken pie distribution, Defect profile bar chart, and recent analyses table with honest empty state. |
| `/analyze` | **Analyze Scan** | Drag-and-drop flatbed scan ingestion, file validation, natural image resolution inspection, parameter selection, and execution trigger. |
| `/analysis/:id` | **Analysis Result** | Detailed laboratory certificate with 5 summary cards, Quality Parameters table (with pale green compliance cells), and optical image inspection viewer. |
| `/history` | **Scan History** | Chronological analysis archive with search, status filters (Decided, Indeterminate, Incomplete), and quick actions. |
| `/reports` | **Quality Reports** | Registry of official certificates with standards compliance filters and JSON/CSV export actions. |
| `/dataset` | **Dataset Explorer** | Safe browser for the 119 ground-truth reference scans grouped by category without exposing arbitrary server filesystem paths. |
| `/models` | **Model Information** | Documented M4 classifier architectures, 5-fold cross-validation metrics, feature sets, and documented operational limitations. |
| `/settings` | **Settings** | Backend connection test ping, 600 DPI scanner resolution calibration, and quality standards tolerance matrix configuration. |

---

## 6. Planned Backend Integration (FastAPI M4)

The frontend's `src/services/api.ts` connects directly to the following endpoints:

- `GET /api/health` — Checks service status and pipeline readiness.
- `POST /api/analyze` — Multipart form upload receiving raw image file, `sample_id`, and `grading_profile`. Returns `AnalysisResult`.
- `GET /api/history` — Returns paginated `ScanHistoryRecord[]`.
- `GET /api/results/{sample_id}` — Retrieves comprehensive analysis report for a specific sample ID.
- `GET /api/dataset` — Returns ground-truth reference dataset entries with category groupings.
- `GET /api/models/info` — Returns model versions, cross-validation metrics, and documented limitations.

---

## 7. Data Integrity & Laboratory Standards Policy

- **Zero Mock Data Fabrication**: When no scans have been analyzed, the dashboard and history archive present clean, honest empty states.
- **Count-Based Percentages**: Denominators are explicitly marked (`Total rice grains` or `Total detected objects`). Gram values display `—` because optical flatbed scans do not perform gravimetric weighing.
- **Regulatory Disclaimers**: Displays clear caveats that internal 5-fold cross-validation metrics do not constitute independent external factory certification.
