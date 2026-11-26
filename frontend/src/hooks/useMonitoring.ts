import { useCallback, useEffect, useRef, useState } from 'react';
import { allFrames, api, ApiError } from '../services/api';
import type { FrameResult, ModelInfo, VideoSession } from '../types/api';

const STORAGE_KEY = 'parkvision.video.session';
const terminal = (status?: string) => ['completed', 'failed', 'cancelled'].includes(status ?? '');
export type Connection = 'idle' | 'connecting' | 'live' | 'polling';

function savedSession(): string | null {
  try { return sessionStorage.getItem(STORAGE_KEY); } catch { return null; }
}
function saveSession(id: string | null) {
  try {
    if (id) sessionStorage.setItem(STORAGE_KEY, id);
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch { /* Private browsing can disable storage; the current tab still works. */ }
}

export function useMonitoring() {
  const [sessionId, setSessionId] = useState<string | null>(savedSession);
  const [connectEpoch, setConnectEpoch] = useState(0);
  const [session, setSession] = useState<VideoSession | null>(null);
  const [frames, setFrames] = useState<FrameResult[]>([]);
  const [connection, setConnection] = useState<Connection>('idle');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<boolean | null>(null);
  const [ready, setReady] = useState<boolean | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const requestToken = useRef(0);
  const busyRef = useRef(false);
  const sessionRef = useRef<VideoSession | null>(null);

  const apply = useCallback((value: VideoSession) => {
    if (sessionRef.current?.id === value.id && sessionRef.current.revision > value.revision) return;
    sessionRef.current = value;
    setSession(value);
    if (value.error) setError(value.error.message);
  }, []);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        await api.health();
      } catch {
        if (!active) return;
        setHealth(false);
        setReady(false);
        setModel(null);
        return;
      }
      if (!active) return;
      setHealth(true);
      const [readyResult, modelResult] = await Promise.allSettled([
        api.ready(), api.modelInfo(),
      ]);
      if (!active) return;
      setReady(readyResult.status === 'fulfilled');
      if (modelResult.status === 'fulfilled') setModel(modelResult.value);
      else setModel(null);
    };
    void refresh();
    const timer = window.setInterval(refresh, 15000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (!sessionId || health !== true) {
      setConnection('idle');
      return;
    }
    let active = true;
    let socket: WebSocket | null = null;
    let retryTimer: number | undefined;
    let refreshInFlight = false;
    let framesInFlight = false;

    const refreshFrames = async () => {
      if (framesInFlight) return;
      framesInFlight = true;
      try {
        const result = await allFrames(sessionId);
        if (active) setFrames(result);
      } catch { /* Keep the last known frame; the next update retries. */ }
      finally { framesInFlight = false; }
    };

    const accept = (value: VideoSession) => {
      if (!active || value.id !== sessionId) return;
      apply(value);
      if (value.frames_processed > 0) void refreshFrames();
      if (terminal(value.status)) {
        clearTimeout(retryTimer);
        setConnection('idle');
      }
    };

    const refresh = async () => {
      if (refreshInFlight) return;
      refreshInFlight = true;
      try {
        const value = await api.session(sessionId);
        accept(value);
      } catch (cause) {
        if (!active) return;
        const failure = cause as ApiError;
        if (failure.status === 404) {
          saveSession(null);
          sessionRef.current = null;
          setSession(null);
          setFrames([]);
          setSessionId(null);
          setConnection('idle');
          setError('That session expired or is no longer available. Start a new analysis.');
        } else setError(failure.message);
      } finally { refreshInFlight = false; }
    };

    const connect = () => {
      if (!active || terminal(sessionRef.current?.status)) return;
      setConnection('connecting');
      try {
        socket = new WebSocket(api.events(sessionId));
        socket.onopen = () => { if (active) setConnection('live'); };
        socket.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data) as { session?: VideoSession; code?: string };
            if (payload.session) accept(payload.session);
            if (payload.code === 'session_not_found') void refresh();
          } catch { if (active) setError('A live update could not be read.'); }
        };
        socket.onerror = () => { if (active) setConnection('polling'); };
        socket.onclose = () => {
          if (!active || terminal(sessionRef.current?.status)) return;
          setConnection('polling');
          retryTimer = window.setTimeout(connect, 5000);
        };
      } catch { setConnection('polling'); retryTimer = window.setTimeout(connect, 5000); }
    };

    void refresh();
    connect();
    const pollingTimer = window.setInterval(() => {
      if (!terminal(sessionRef.current?.status) && socket?.readyState !== WebSocket.OPEN) {
        void refresh();
      }
    }, 2500);
    return () => {
      active = false;
      clearTimeout(retryTimer);
      clearInterval(pollingTimer);
      socket?.close();
    };
  }, [sessionId, connectEpoch, health, apply]);

  const adopt = (value: VideoSession) => {
    sessionRef.current = null;
    setFrames([]);
    setError(null);
    apply(value);
    saveSession(value.id);
    setSessionId(value.id);
  };

  const start = async (source: { url: string } | { file: File }) => {
    if (busyRef.current || (sessionRef.current && !terminal(sessionRef.current.status))) return;
    const token = ++requestToken.current;
    busyRef.current = true; setBusy(true); setError(null);
    try {
      const value = 'url' in source ? await api.youtube(source.url) : await api.upload(source.file);
      if (token !== requestToken.current) {
        await api.cancel(value.id).catch(() => undefined);
        return;
      }
      adopt(value);
    } catch (cause) {
      if (token === requestToken.current) {
        const failure = cause as ApiError;
        setError(failure.code === 'session_capacity'
          ? 'Video capacity is full. Wait for the current analysis to finish.' : failure.message);
      }
    } finally {
      if (token === requestToken.current) { busyRef.current = false; setBusy(false); }
    }
  };

  const reconnect = (id: string) => {
    const trimmed = id.trim();
    if (!trimmed || busyRef.current) return;
    if (sessionRef.current && !terminal(sessionRef.current.status) && sessionRef.current.id !== trimmed) {
      setError('Stop the current analysis before connecting to another session.');
      return;
    }
    ++requestToken.current;
    setError(null);
    sessionRef.current = null;
    setSession(null);
    setFrames([]);
    saveSession(trimmed);
    setSessionId(trimmed);
    setConnectEpoch((value) => value + 1);
  };

  const stop = async () => {
    const current = sessionRef.current;
    if (!current || terminal(current.status)) return;
    const token = requestToken.current;
    try {
      const value = await api.cancel(current.id);
      if (token === requestToken.current && sessionRef.current?.id === current.id) apply(value);
    } catch (cause) {
      if (token === requestToken.current) setError((cause as Error).message);
    }
  };

  const reset = async () => {
    const current = sessionRef.current;
    ++requestToken.current;
    if (current && !terminal(current.status)) {
      try { await api.cancel(current.id); }
      catch (cause) { setError((cause as Error).message); return; }
    }
    saveSession(null);
    sessionRef.current = null;
    setSessionId(null);
    setSession(null);
    setFrames([]);
    setError(null);
    setBusy(false);
    busyRef.current = false;
  };

  return { session, sessionId, frames, connection, busy, error, health, ready, model,
    start, reconnect, stop, reset, clearError: () => setError(null) };
}
