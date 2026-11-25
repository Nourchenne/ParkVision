import { useCallback, useEffect, useRef, useState } from 'react';
import { Film, ScanLine } from 'lucide-react';
import { api } from '../services/api';
import type { FrameResult, VideoSession } from '../types/api';

type YouTubePlayer = { getCurrentTime: () => number; getPlayerState: () => number; destroy: () => void };
type YouTubeApi = { Player: new (element: HTMLElement, options: {
  events: { onReady: () => void; onStateChange: (event: { data: number }) => void };
}) => YouTubePlayer };
declare global {
  interface Window {
    YT?: YouTubeApi;
    onYouTubeIframeAPIReady?: () => void;
  }
}

let youtubeApiPromise: Promise<YouTubeApi> | null = null;
function loadYouTubeApi(): Promise<YouTubeApi> {
  if (window.YT?.Player) return Promise.resolve(window.YT);
  if (!youtubeApiPromise) youtubeApiPromise = new Promise((resolve, reject) => {
    const previous = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      previous?.();
      if (window.YT?.Player) resolve(window.YT);
      else reject(new Error('YouTube player API is unavailable'));
    };
    const script = document.createElement('script');
    script.src = 'https://www.youtube.com/iframe_api';
    script.onerror = () => { youtubeApiPromise = null; reject(new Error('YouTube player API failed to load')); };
    document.head.append(script);
  });
  return youtubeApiPromise;
}

export function frameAt(frames: FrameResult[], time: number, interval: number): FrameResult | null {
  if (!Number.isFinite(time) || interval <= 0) return null;
  for (let index = frames.length - 1; index >= 0; index--) {
    const frame = frames[index];
    if (frame.timestamp_seconds <= time) {
      return time - frame.timestamp_seconds <= interval * 1.5 ? frame : null;
    }
  }
  return null;
}

export function VideoPlayer({ session, frames }: { session: VideoSession | null; frames: FrameResult[] }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const youtubeContainer = useRef<HTMLDivElement>(null);
  const pendingPlayback = useRef<{ id: string; time: number; playing: boolean; ended: boolean } | null>(null);
  const sendingPlayback = useRef(false);
  const [overlayLabel, setOverlayLabel] = useState('Awaiting playback');
  const sessionId = session?.id;
  const analysisActive = Boolean(session && !['completed', 'failed', 'cancelled'].includes(session.status));
  const analysisActiveRef = useRef(analysisActive);
  analysisActiveRef.current = analysisActive;
  const sourceAvailable = session?.source_type === 'upload' && !['failed', 'cancelled'].includes(session.status);
  const youtubeAvailable = session?.source_type === 'youtube' && Boolean(session.youtube_video_id)
    && !['failed', 'cancelled'].includes(session.status);

  const reportPlayback = useCallback((time: number, playing: boolean, ended = false) => {
    if (!sessionId || !analysisActiveRef.current || !Number.isFinite(time)) return;
    pendingPlayback.current = { id: sessionId, time, playing, ended };
    if (sendingPlayback.current) return;
    sendingPlayback.current = true;
    void (async () => {
      while (pendingPlayback.current) {
        const update = pendingPlayback.current;
        pendingPlayback.current = null;
        try { await api.playback(update.id, update.time, update.playing, update.ended); }
        catch { setOverlayLabel('Playback synchronization unavailable · retrying'); }
      }
      sendingPlayback.current = false;
    })();
  }, [sessionId]);

  useEffect(() => {
    const player = video.current;
    if (!sourceAvailable || !analysisActive || !player) return;
    const sync = () => reportPlayback(player.currentTime, !player.paused && !player.ended, player.ended);
    for (const event of ['play', 'pause', 'ended', 'seeked', 'timeupdate']) {
      player.addEventListener(event, sync);
    }
    const timer = window.setInterval(() => { if (!player.paused) sync(); }, 500);
    sync();
    return () => {
      clearInterval(timer);
      for (const event of ['play', 'pause', 'ended', 'seeked', 'timeupdate']) {
        player.removeEventListener(event, sync);
      }
      reportPlayback(player.currentTime, false, player.ended);
    };
  }, [sessionId, sourceAvailable, analysisActive, reportPlayback]);

  useEffect(() => {
    const container = youtubeContainer.current;
    if (!youtubeAvailable || !session?.youtube_video_id || !container) return;
    const iframe = document.createElement('iframe');
    iframe.title = 'YouTube source playback';
    iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
    iframe.allowFullscreen = true;
    iframe.referrerPolicy = 'strict-origin-when-cross-origin';
    iframe.src = `https://www.youtube-nocookie.com/embed/${session.youtube_video_id}?rel=0&enablejsapi=1&origin=${encodeURIComponent(window.location.origin)}`;
    container.appendChild(iframe);
    let disposed = false;
    let player: YouTubePlayer | null = null;
    let timer: number | undefined;
    const clearTimer = () => { clearInterval(timer); timer = undefined; };
    const onStateChange = (event: { data: number }) => {
      if (!player || disposed) return;
      clearTimer();
      reportPlayback(player.getCurrentTime(), event.data === 1, event.data === 0);
      if (event.data === 1) timer = window.setInterval(() => {
        if (player) reportPlayback(player.getCurrentTime(), player.getPlayerState() === 1);
      }, 500);
    };
    void loadYouTubeApi().then((yt) => {
      if (disposed) return;
      player = new yt.Player(iframe, {
        events: {
          onReady: () => { if (player) onStateChange({ data: player.getPlayerState() }); },
          onStateChange,
        },
      });
    }).catch(() => setOverlayLabel('YouTube playback controls unavailable'));
    return () => {
      disposed = true;
      clearTimer();
      if (player) {
        reportPlayback(player.getCurrentTime(), false);
        player.destroy();
      }
      container.replaceChildren();
    };
  }, [sessionId, session?.youtube_video_id, youtubeAvailable, reportPlayback]);

  useEffect(() => {
    if (!sourceAvailable || !session) return;
    const player = video.current;
    const layer = canvas.current;
    if (!player || !layer) return;
    const paint = () => {
      const context = layer.getContext('2d');
      if (!context) return;
      const rect = player.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      layer.width = Math.round(rect.width * ratio);
      layer.height = Math.round(rect.height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, rect.width, rect.height);
      if (!player.videoWidth || !player.videoHeight || player.seeking) {
        setOverlayLabel('Preview only · playback not aligned');
        return;
      }
      const frame = frameAt(frames, player.currentTime, 1 / session.sample_fps);
      if (!frame || frame.source_width !== player.videoWidth || frame.source_height !== player.videoHeight) {
        setOverlayLabel('Preview only · no matching sample');
        return;
      }
      const scale = Math.min(rect.width / player.videoWidth, rect.height / player.videoHeight);
      const left = (rect.width - player.videoWidth * scale) / 2;
      const top = (rect.height - player.videoHeight * scale) / 2;
      for (const detection of frame.detections) {
        const [x1, y1, x2, y2] = detection.bbox;
        const color = detection.class_id === 0 ? '#50e5a4' : '#ff6b77';
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.strokeRect(left + x1 * scale, top + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
        const label = `${detection.class_name} ${Math.round(detection.confidence * 100)}%`;
        context.font = '600 12px system-ui';
        const width = context.measureText(label).width + 10;
        const labelY = Math.max(top, top + y1 * scale - 20);
        context.fillStyle = color;
        context.fillRect(left + x1 * scale, labelY, width, 18);
        context.fillStyle = '#061018';
        context.fillText(label, left + x1 * scale + 5, labelY + 13);
      }
      setOverlayLabel(`Sample at ${frame.timestamp_seconds.toFixed(2)} s`);
    };
    const observer = new ResizeObserver(paint);
    observer.observe(player);
    player.addEventListener('timeupdate', paint);
    player.addEventListener('seeked', paint);
    player.addEventListener('loadedmetadata', paint);
    paint();
    return () => {
      observer.disconnect();
      player.removeEventListener('timeupdate', paint);
      player.removeEventListener('seeked', paint);
      player.removeEventListener('loadedmetadata', paint);
    };
  }, [frames, session, sourceAvailable]);

  return <div className="video-stage">
    {sourceAvailable && session && <>
      <video key={session.id} ref={video} controls preload="metadata" src={api.source(session.id)}
        aria-label="Uploaded source video" />
      <canvas ref={canvas} className="detection-canvas" aria-hidden="true" />
      <span className="video-label"><ScanLine size={14} /> {overlayLabel}</span>
    </>}
    {youtubeAvailable && session && <>
      <div ref={youtubeContainer} className="youtube-player" />
      <span className="video-label"><ScanLine size={14} /> {overlayLabel === 'Awaiting playback'
        ? 'Annotated preview shown separately' : overlayLabel}</span>
    </>}
    {!sourceAvailable && !youtubeAvailable && <div className="video-empty"><Film size={42} strokeWidth={1.2} />
      <strong>Source playback will appear here</strong><span>Start an analysis to view your video.</span></div>}
  </div>;
}
