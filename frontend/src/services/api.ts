import type { FramePage, ModelInfo, Prediction, VideoSession } from '../types/api';

export class ApiError extends Error {
  constructor(message: string, public status: number, public code?: string) {
    super(message);
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { ...options, cache: 'no-store' });
  } catch {
    throw new ApiError('Cannot reach the FastAPI server.', 0, 'connection_failed');
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    throw new ApiError(detail?.message ?? `Request failed (${response.status}).`,
      response.status, detail?.code);
  }
  return body as T;
}

export const api = {
  health: () => request<{ status: string }>('/health'),
  ready: () => request<{ status: string }>('/ready'),
  modelInfo: () => request<ModelInfo>('/model-info'),
  session: (id: string) => request<VideoSession>(`/video/sessions/${encodeURIComponent(id)}`),
  frames: (id: string, offset: number) => request<FramePage>(
    `/video/sessions/${encodeURIComponent(id)}/frames?offset=${offset}&limit=100`),
  youtube: (url: string) => request<VideoSession>('/video/youtube', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url }),
  }),
  upload: (file: File) => {
    const body = new FormData();
    body.append('file', file);
    return request<VideoSession>('/video/upload', { method: 'POST', body });
  },
  predict: (file: File) => {
    const body = new FormData();
    body.append('file', file);
    return request<Prediction>('/predict', { method: 'POST', body });
  },
  predictVisualize: async (file: File): Promise<string> => {
    const body = new FormData();
    body.append('file', file);
    let response: Response;
    try {
      response = await fetch('/predict/visualize', { method: 'POST', body, cache: 'no-store' });
    } catch {
      throw new ApiError('Cannot reach the FastAPI server.', 0, 'connection_failed');
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new ApiError(detail?.detail?.message ?? `Request failed (${response.status}).`,
        response.status, detail?.detail?.code);
    }
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  },
  cancel: (id: string) => request<VideoSession>(`/video/sessions/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  }),
  playback: (id: string, currentTime: number, playing: boolean, ended = false) =>
    request<VideoSession>(`/video/sessions/${encodeURIComponent(id)}/playback`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_time_seconds: currentTime, playing, ended }),
    }),
  source: (id: string) => `/video/sessions/${encodeURIComponent(id)}/source`,
  preview: (id: string, revision: number) =>
    `/video/sessions/${encodeURIComponent(id)}/preview?revision=${revision}`,
  events: (id: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/video/sessions/${encodeURIComponent(id)}/events`;
  },
};

export async function allFrames(id: string): Promise<FramePage['items']> {
  const frames: FramePage['items'] = [];
  let total: number;
  do {
    const page = await api.frames(id, frames.length);
    frames.push(...page.items);
    total = page.total;
    if (page.items.length === 0) break;
  } while (frames.length < total);
  return frames;
}
