// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../services/api';
import type { FrameResult, VideoSession } from '../types/api';
import { frameAt, VideoPlayer } from './VideoPlayer';

afterEach(() => vi.restoreAllMocks());

describe('MP4 overlay sample selection', () => {
  const frames = [0, 1, 2].map((timestamp_seconds) => ({ timestamp_seconds })) as FrameResult[];
  it('uses the newest analyzed sample at or before playback time', () => {
    expect(frameAt(frames, 1.2, 1)?.timestamp_seconds).toBe(1);
    expect(frameAt(frames, 0.2, 1)?.timestamp_seconds).toBe(0);
  });
  it('does not draw a stale or future sample', () => {
    expect(frameAt(frames, 4, 1)).toBeNull();
    expect(frameAt(frames, -1, 1)).toBeNull();
  });
});

it('reports MP4 play, pause, and end to the analysis session', async () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  const session = {
    id: 'session-1', source_type: 'upload', status: 'processing', sample_fps: 1,
  } as VideoSession;
  const report = vi.spyOn(api, 'playback').mockResolvedValue(session);
  render(<VideoPlayer session={session} frames={[]} />);
  const player = screen.getByLabelText('Uploaded source video');
  Object.defineProperties(player, {
    currentTime: { configurable: true, value: 0 },
    paused: { configurable: true, value: false },
    ended: { configurable: true, value: false },
  });
  fireEvent.play(player);
  await waitFor(() => expect(report).toHaveBeenCalledWith('session-1', 0, true, false));
  Object.defineProperty(player, 'paused', { configurable: true, value: true });
  fireEvent.pause(player);
  await waitFor(() => expect(report).toHaveBeenCalledWith('session-1', 0, false, false));
  Object.defineProperty(player, 'ended', { configurable: true, value: true });
  fireEvent.ended(player);
  await waitFor(() => expect(report).toHaveBeenCalledWith('session-1', 0, false, true));
});
