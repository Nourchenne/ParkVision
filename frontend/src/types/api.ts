export type SessionStatus = 'queued' | 'acquiring' | 'processing' | 'completed' | 'failed' | 'cancelled';
export type SourceType = 'upload' | 'youtube';

export interface ModelInfo {
  name: string;
  alias: string;
  version: string | null;
  architecture: string | null;
  source_run_id: string | null;
  model_uri: string | null;
  status: 'loaded' | 'unavailable';
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: number[];
}

export interface Counts { empty: number; occupied: number; total: number }

export interface FrameResult {
  timestamp_seconds: number;
  source_width: number;
  source_height: number;
  model: Pick<ModelInfo, 'name' | 'version' | 'architecture'>;
  counts: Counts;
  detections: Detection[];
  decode_time_ms: number;
  inference_time_ms: number;
  analysis_time_ms: number;
  session_elapsed_ms: number;
}

export interface VideoSession {
  id: string;
  source_type: SourceType;
  status: SessionStatus;
  revision: number;
  created_at: string;
  updated_at: string;
  duration_seconds: number | null;
  source_width: number | null;
  source_height: number | null;
  current_timestamp_seconds: number | null;
  analyzed_frontier_seconds: number | null;
  frames_processed: number;
  sample_fps: number;
  latest: FrameResult | null;
  error: { code: string; message: string } | null;
  youtube_video_id: string | null;
}

export interface FramePage {
  items: FrameResult[];
  offset: number;
  limit: number;
  total: number;
}

export interface Prediction {
  model: Pick<ModelInfo, 'name' | 'version' | 'architecture'>;
  counts: Counts;
  detections: Detection[];
  inference_time_ms: number;
}
