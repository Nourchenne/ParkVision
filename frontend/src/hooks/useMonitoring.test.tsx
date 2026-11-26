// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useMonitoring } from './useMonitoring';

class FakeSocket {
  static OPEN = 1;
  static instances: FakeSocket[] = [];
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  constructor(public url: string) { FakeSocket.instances.push(this); }
  close() { this.closed = true; this.readyState = 3; }
}

const snapshot = {
  id: 'saved-session', source_type: 'upload', status: 'processing', revision: 2,
  created_at: '', updated_at: '', duration_seconds: 10, source_width: 32,
  source_height: 24, current_timestamp_seconds: null, analyzed_frontier_seconds: null,
  frames_processed: 0, sample_fps: 1, latest: null, error: null, youtube_video_id: null,
};

afterEach(() => { vi.unstubAllGlobals(); FakeSocket.instances = []; });

describe('session restoration', () => {
  it('keeps a saved ID while the API is offline without opening a WebSocket', async () => {
    const storage = new Map([['parkvision.video.session', 'saved-session']]);
    vi.stubGlobal('sessionStorage', {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
    });
    vi.stubGlobal('WebSocket', FakeSocket);
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('connection refused'));
    vi.stubGlobal('fetch', fetchMock);
    const view = renderHook(() => useMonitoring());
    await waitFor(() => expect(view.result.current.health).toBe(false));
    expect(view.result.current.sessionId).toBe('saved-session');
    expect(FakeSocket.instances).toHaveLength(0);
    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual(['/health']);
    view.unmount();
  });

  it('reads a saved session without creating a new analysis and closes its socket on unmount', async () => {
    const storage = new Map([['parkvision.video.session', 'saved-session']]);
    vi.stubGlobal('sessionStorage', {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
    });
    vi.stubGlobal('WebSocket', FakeSocket);
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(new Response(
      JSON.stringify(path === '/video/sessions/saved-session' ? snapshot :
        path === '/model-info' ? { name: 'ParkVision-Detector', version: '2', status: 'loaded' } :
          { status: 'ok' }), { status: 200, headers: { 'content-type': 'application/json' } },
    )));
    vi.stubGlobal('fetch', fetchMock);
    const view = renderHook(() => useMonitoring());
    await waitFor(() => expect(view.result.current.session?.id).toBe('saved-session'));
    expect(FakeSocket.instances).toHaveLength(1);
    expect(fetchMock.mock.calls.every(([path]) => !String(path).includes('/video/upload') &&
      !String(path).includes('/video/youtube'))).toBe(true);
    await act(async () => { view.rerender(); });
    expect(FakeSocket.instances).toHaveLength(1);
    view.unmount();
    expect(FakeSocket.instances[0].closed).toBe(true);
  });

  it('starts only once and cancels a late response after reset', async () => {
    const storage = new Map<string, string>();
    vi.stubGlobal('sessionStorage', {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
    });
    vi.stubGlobal('WebSocket', FakeSocket);
    let resolveStart!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => { resolveStart = resolve; });
    const fetchMock = vi.fn().mockImplementation((path: string, options?: RequestInit) => {
      if (path === '/video/youtube') return pending;
      return Promise.resolve(new Response(JSON.stringify(options?.method === 'DELETE'
        ? { ...snapshot, status: 'cancelled' } : { status: 'ok' }),
      { status: 200, headers: { 'content-type': 'application/json' } }));
    });
    vi.stubGlobal('fetch', fetchMock);
    const view = renderHook(() => useMonitoring());
    act(() => {
      void view.result.current.start({ url: 'https://www.youtube.com/watch?v=abcdefghijk' });
      void view.result.current.start({ url: 'https://www.youtube.com/watch?v=abcdefghijk' });
    });
    expect(fetchMock.mock.calls.filter(([path]) => path === '/video/youtube')).toHaveLength(1);
    await act(async () => { await view.result.current.reset(); });
    resolveStart(new Response(JSON.stringify(snapshot),
      { status: 202, headers: { 'content-type': 'application/json' } }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([path, options]) =>
      path === '/video/sessions/saved-session' && options?.method === 'DELETE')).toBe(true));
    expect(view.result.current.session).toBeNull();
    view.unmount();
  });
});
