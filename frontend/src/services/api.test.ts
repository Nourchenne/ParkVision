import { afterEach, describe, expect, it, vi } from 'vitest';
import { allFrames, api, ApiError } from './api';

afterEach(() => vi.unstubAllGlobals());

describe('typed API client', () => {
  it('passes the backend capacity code through for the session UI', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: { code: 'session_capacity', message: 'capacity is full' },
    }), { status: 409, headers: { 'content-type': 'application/json' } })));
    await expect(api.youtube('https://www.youtube.com/watch?v=abcdefghijk'))
      .rejects.toMatchObject<Partial<ApiError>>({ status: 409, code: 'session_capacity' });
  });

  it('retrieves all paginated frames for a restored session', async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(new Response(
      JSON.stringify(path.includes('offset=0')
        ? { items: Array.from({ length: 100 }, (_, timestamp_seconds) => ({ timestamp_seconds })), total: 101 }
        : { items: [{ timestamp_seconds: 100 }], total: 101 }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    )));
    vi.stubGlobal('fetch', fetchMock);
    const frames = await allFrames('existing-id');
    expect(frames).toHaveLength(101);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('sends the playback clock and pause state to the video session', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    await api.playback('session-1', 12.5, false);
    expect(fetchMock).toHaveBeenCalledWith('/video/sessions/session-1/playback', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ current_time_seconds: 12.5, playing: false, ended: false }),
    }));
  });
});
