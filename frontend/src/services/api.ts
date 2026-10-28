import type {
  AnalysisResult,
  ScanHistoryRecord,
  DatasetItem,
  ModelInfo,
} from '@/types/analysis'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  status?: number
  details?: unknown

  constructor(message: string, status?: number, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.details = details
  }
}

/**
 * Checks backend API health and connectivity.
 */
export async function checkBackendHealth(): Promise<{ online: boolean; message: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    })
    if (!res.ok) {
      return { online: false, message: `Server responded with status ${res.status}` }
    }
    const data = await res.json()
    return { online: true, message: data.message || 'FastAPI Backend Online' }
  } catch (err: unknown) {
    return {
      online: false,
      message: err instanceof Error ? err.message : 'Cannot connect to backend server',
    }
  }
}

/**
 * Submits a raw scan image to the pipeline analysis endpoint.
 * Note: Never simulates fake predictions. Will throw or fail cleanly if backend is unavailable.
 */
export async function analyzeScan(
  file: File,
  sampleId?: string,
  profileId = 'CODEX_CXS_198_1995',
  requireFullCriteria = false
): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('image', file)
  if (sampleId) formData.append('sample_id', sampleId)
  formData.append('grading_profile', profileId)
  formData.append('require_full_criteria', String(requireFullCriteria))

  try {
    const res = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
    })

    if (!res.ok) {
      let errMsg = `Analysis failed (${res.status} ${res.statusText})`
      try {
        const errJson = await res.json()
        if (errJson.detail) errMsg = errJson.detail
      } catch {
        // Fallback to HTTP status text
      }
      throw new ApiError(errMsg, res.status)
    }

    return await res.json()
  } catch (err) {
    if (err instanceof ApiError) throw err
    throw new ApiError(
      `Cannot connect to RiceAge analysis backend at ${API_BASE_URL}. Ensure the FastAPI server is running or configure VITE_API_BASE_URL in your environment.`,
      0,
      err
    )
  }
}

/**
 * Retrieves scan analysis history.
 */
export async function getScanHistory(params?: {
  search?: string
  status?: string
  limit?: number
  offset?: number
}): Promise<{ records: ScanHistoryRecord[]; total: number }> {
  const query = new URLSearchParams()
  if (params?.search) query.append('search', params.search)
  if (params?.status) query.append('status', params.status)
  if (params?.limit) query.append('limit', String(params.limit))
  if (params?.offset) query.append('offset', String(params.offset))

  try {
    const res = await fetch(`${API_BASE_URL}/api/history?${query.toString()}`)
    if (!res.ok) {
      throw new ApiError(`Failed to fetch history (${res.status})`, res.status)
    }
    const data = await res.json()
    if (Array.isArray(data)) {
      return { records: data, total: data.length }
    }
    return data
  } catch (err) {
    if (err instanceof ApiError) throw err
    // When backend is not yet started, return empty list cleanly without inventing data
    return { records: [], total: 0 }
  }
}

/**
 * Retrieves a single analysis result by ID.
 */
export async function getScanResult(id: string): Promise<AnalysisResult> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/results/${encodeURIComponent(id)}`)
    if (!res.ok) {
      throw new ApiError(`Analysis result not found (${res.status})`, res.status)
    }
    return await res.json()
  } catch (err) {
    if (err instanceof ApiError) throw err
    throw new ApiError(`Failed to fetch analysis result for ID "${id}".`, 0, err)
  }
}

/**
 * Retrieves information about the dataset items.
 */
export async function getDatasetItems(category?: string): Promise<{ items: DatasetItem[]; total: number }> {
  const query = category ? `?category=${encodeURIComponent(category)}` : ''
  try {
    const res = await fetch(`${API_BASE_URL}/api/dataset${query}`)
    if (!res.ok) {
      throw new ApiError(`Failed to fetch dataset items (${res.status})`, res.status)
    }
    return await res.json()
  } catch (err) {
    if (err instanceof ApiError) throw err
    // Return empty list if server is offline
    return { items: [], total: 0 }
  }
}

/**
 * Retrieves documented model configuration and validation metrics.
 */
export async function getModelInfo(): Promise<ModelInfo> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/models/info`)
    if (res.ok) {
      return await res.json()
    }
  } catch {
    // Graceful fallback to verified M4 static metadata below
  }

  // Authoritative static metadata sourced directly from Milestone M4 weights and reports
  return {
    version: 'M4.0 Integrated Quality',
    architecture: 'Classical Morphological + CIELAB Colorimetry + Regularized Logistic Regression Heads',
    last_updated: '2026-10-12',
    heads: [
      {
        head_name: 'foreign_matter',
        display_name: 'Foreign Matter Detector',
        features: ['solidity', 'circularity', 'roughness', 'edge_density', 'aspect_ratio', 'length_px', 'area_px', 'yellow_ratio', 'mean_sat', 'mean_b', 'mean_L'],
        samples_trained: 2529,
        positives_trained: 461,
        threshold: 0.50,
        accuracy: 0.9992,
        precision: 0.9957,
        recall: 1.0000,
        f1_score: 0.9978,
        description: 'Separates non-rice contaminants (stones, seeds, chaff, debris) from cereal grains.'
      },
      {
        head_name: 'whole_broken',
        display_name: 'Whole vs. Broken Morphology',
        features: ['length_px', 'width_px', 'aspect_ratio', 'area_px', 'perimeter_px', 'circularity', 'solidity', 'extent', 'eq_diameter', 'max_bluntness', 'min_bluntness', 'end_asymmetry'],
        samples_trained: 1017,
        positives_trained: 511,
        threshold: 0.50,
        accuracy: 0.9961,
        precision: 0.9942,
        recall: 0.9980,
        f1_score: 0.9961,
        description: 'Classifies rice kernels as Whole (Head Rice >= 75% length) or Broken fragments.'
      },
      {
        head_name: 'chalky',
        display_name: 'Chalky Endosperm Detector',
        features: ['mean_L', 'std_L', 'chalky_ratio', 'high_whiteness', 'mean_gray', 'std_gray', 'mean_sat', 'std_sat'],
        samples_trained: 934,
        positives_trained: 423,
        threshold: 0.50,
        accuracy: 0.9882,
        precision: 0.9882,
        recall: 0.9858,
        f1_score: 0.9870,
        description: 'Identifies opaque, milky-white chalkiness occupying >= 50% of the kernel area.'
      },
      {
        head_name: 'discoloured',
        display_name: 'Discoloured & Peck Detector',
        features: ['mean_b', 'std_b', 'mean_sat', 'std_sat', 'mean_hue', 'yellow_ratio', 'dark_spot_ratio', 'mean_L', 'chalky_ratio'],
        samples_trained: 1111,
        positives_trained: 177,
        threshold: 0.60,
        accuracy: 0.8632,
        precision: 0.5532,
        recall: 0.7345,
        f1_score: 0.6311,
        description: 'Detects fungal peck, enzymatic browning, heat damage, and yellowed grains.'
      },
      {
        head_name: 'damaged',
        display_name: 'Insect Damage & Cavitation Detector',
        features: ['edge_density', 'roughness', 'solidity', 'circularity', 'max_defect_depth', 'num_defects', 'dark_spot_ratio', 'std_gray'],
        samples_trained: 1468,
        positives_trained: 451,
        threshold: 0.60,
        accuracy: 0.7881,
        precision: 0.6750,
        recall: 0.5987,
        f1_score: 0.6345,
        description: 'Identifies insect-chewed edges, perimeter cavitation, stress cracks, and dark blemishes.'
      }
    ],
    cross_trigger_reduction: {
      broken_to_damaged_reduction_pct: 85.94,
      chalky_to_discoloured_reduction_pct: 77.02,
      broken_to_damaged_rate_m3: 0.9980,
      broken_to_damaged_rate_m4: 0.1403,
      chalky_to_discoloured_rate_m3: 0.9362,
      chalky_to_discoloured_rate_m4: 0.2151,
    },
    disclaimers: [
      'Metrics reflect 5-fold StratifiedGroupKFold internal cross-validation on 119 flatbed scans and do not constitute independent external factory certification.',
      'Results represent optical count-based screening estimates and do not substitute for official certified mass-fraction weighing or moisture tests.'
    ],
    metrics: {
      accuracy: 0.9961,
      precision: 0.9942,
      recall: 0.9980,
      f1Score: 0.9961
    },
    features: [
      'length_px',
      'width_px',
      'aspect_ratio',
      'area_px',
      'perimeter_px',
      'circularity',
      'solidity',
      'mean_L',
      'mean_b',
      'mean_sat',
      'edge_density',
      'defect_spot_ratio'
    ],
    limitations: [
      'Internal 119-image dataset collected exclusively on a single 600 DPI flatbed optical scanner model.',
      'Cross-validation performance does not guarantee equivalent generalization to mobile cameras or varied lighting.',
      'Mass balances are non-gravimetric count-based proportions, not certified weight fractions (w/w).',
      'Touching or clumped grains require physical tray spreading to prevent false defect cavitation.'
    ]
  }
}

export const api = {
  checkBackendHealth: async (): Promise<boolean> => {
    const res = await checkBackendHealth()
    return res.online
  },
  checkBackendHealthDetail: checkBackendHealth,
  analyzeScan,
  getScanHistory: async (params?: { search?: string; status?: string; limit?: number; offset?: number }): Promise<ScanHistoryRecord[]> => {
    const res = await getScanHistory(params)
    return res.records
  },
  getScanHistoryFull: getScanHistory,
  getScanResult,
  getDatasetItems: async (category?: string): Promise<DatasetItem[]> => {
    const res = await getDatasetItems(category)
    return res.items
  },
  getDatasetItemsFull: getDatasetItems,
  getModelInfo,
}
