/**
 * TypeScript Type Definitions for RiceAge Quality Analysis & Grading System
 * Directly mapped to Milestone M4 Python Dataclasses and Artifact Schemas.
 */

export type GrainMorphology = 'Whole' | 'Broken' | 'N/A'

export type GradingStatus = 'DECIDED' | 'INDETERMINATE' | 'INCOMPLETE'

export type StandardProfileId = 'CODEX_CXS_198_1995' | 'USDA_MILLED_RICE' | 'IS_4333_AGMARK'

export interface GrainConfidences {
  foreign_matter?: number
  whole?: number
  chalky?: number
  discoloured?: number
  damaged?: number
}

export interface GrainClassification {
  object_id: string
  is_rice: boolean
  is_foreign_matter: boolean
  morphology: GrainMorphology
  is_chalky: boolean
  is_damaged: boolean
  is_discoloured: boolean
  confidences?: GrainConfidences
}

export interface SampleQualityMetrics {
  sample_id: string
  total_detected_objects: number
  total_rice_grains: number
  total_foreign_matter: number

  whole_count: number
  broken_count: number
  chalky_count: number
  damaged_count: number
  discoloured_count: number
  multi_defect_rice_count: number

  whole_percentage: number
  broken_percentage: number
  chalky_percentage: number
  damaged_percentage: number
  discoloured_percentage: number
  foreign_matter_percentage: number

  denominator_definitions?: Record<string, string>
}

export interface GradingDecision {
  profile_id: StandardProfileId | string
  standard_name: string
  assigned_grade: string | null
  status: GradingStatus
  is_compliant: boolean
  passed_criteria: Record<string, boolean>
  failed_criteria: Record<string, string>
  unsupported_criteria: string[]
  audit_notes: string[]
}

export interface ReportFiles {
  html_path?: string
  json_path?: string
  csv_path?: string
  html_url?: string
  json_url?: string
  csv_url?: string
}

export interface AnalysisSummary {
  timestamp: string
  image_path: string
  image_dimensions: { width: number; height: number }
  total_grains_detected: number
  calibration: {
    is_verified: boolean
    effective_dpi: number | null
    pixel_scale_mm: number | null
    pixels_per_mm: number | null
    source_info: string
  }
  separation: {
    clusters_detected: number
    clusters_successfully_split: number
    clusters_unresolved: number
  }
  quality_classification: {
    total_detected_objects: number
    total_rice_grains: number
    whole_count: number
    broken_count: number
    chalky_count: number
    damaged_count: number
    discoloured_count: number
    foreign_matter_count: number
    multi_defect_rice_count: number
    whole_percentage: number
    broken_percentage: number
    chalky_percentage: number
    damaged_percentage: number
    discoloured_percentage: number
    foreign_matter_percentage: number
    screening_type: string
    fallback_warning?: string | null
  }
  quality_grading: GradingDecision
  quality_reports?: ReportFiles
}

export interface QualityParameterResult {
  parameter: string
  applicableLimit?: string | null
  resultPercentage: number
  resultGrams?: number | null
  denominator: string
  status: 'acceptable' | 'warning' | 'substandard'
}

export interface AnalysisResult {
  id: string
  sample_id: string
  sampleId?: string
  image_name: string
  imageName?: string
  analysis_date?: string
  analysisDate?: string
  pipeline_version?: string
  pipelineVersion?: string
  original_image_url?: string
  originalImageUrl?: string
  overlay_image_url?: string
  overlayImageUrl?: string
  image_url?: string
  overlay_url?: string
  summary: AnalysisSummary & {
    totalObjects?: number
    totalGrains?: number
    wholeGrains?: number
    brokenGrains?: number
    wholePercentage?: number
    brokenPercentage?: number
    foreignMatterCount?: number
    foreignMatterPercentage?: number
  }
  quality_metrics: SampleQualityMetrics
  grading_decision: GradingDecision
  grading?: {
    decision: GradingStatus
    gradeClassification: string
    standardName: string
    reasons: string[]
  }
  parameters?: QualityParameterResult[]
  grain_classifications: GrainClassification[]
  report_files?: ReportFiles
}

export interface ScanHistoryRecord {
  id: string
  sample_id: string
  sampleId?: string
  image_name: string
  imageName?: string
  analysis_date: string
  analysisDate?: string
  total_objects: number
  totalObjects?: number
  grainCount?: number
  total_rice_grains: number
  broken_percentage: number
  brokenPercentage?: number
  chalky_percentage: number
  chalkyPercentage?: number
  damaged_percentage: number
  damagedPercentage?: number
  discoloured_percentage: number
  discolouredPercentage?: number
  foreign_matter_percentage: number
  foreignMatterCount?: number
  defectCount?: number
  status: GradingStatus
  screeningStatus?: GradingStatus
  assigned_grade: string | null
  standard_name: string
  has_report: boolean
}

export interface DatasetItem {
  id: string
  filename: string
  fileName?: string
  category: 'Whole' | 'Broken' | 'Chalky' | 'Damaged' | 'Discoloured' | 'Foreign Matter' | 'Mixed'
  resolution: string
  grain_count: number
  annotatedCount?: number
  ground_truth_condition: string
  thumbnail_url?: string
}

export interface ModelHeadMetadata {
  head_name: string
  display_name: string
  features: string[]
  samples_trained: number
  positives_trained: number
  threshold: number
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  description: string
}

export interface ModelInfo {
  version: string
  architecture: string
  last_updated: string
  heads: ModelHeadMetadata[]
  cross_trigger_reduction: {
    broken_to_damaged_reduction_pct: number
    chalky_to_discoloured_reduction_pct: number
    broken_to_damaged_rate_m3: number
    broken_to_damaged_rate_m4: number
    chalky_to_discoloured_rate_m3: number
    chalky_to_discoloured_rate_m4: number
  }
  disclaimers: string[]
  metrics?: {
    accuracy: number
    precision: number
    recall: number
    f1Score: number
  }
  features?: string[]
  limitations?: string[]
}
