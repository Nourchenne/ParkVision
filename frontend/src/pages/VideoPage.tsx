import { useRef, useState } from 'react';
import { ArrowRight, CircleAlert, Clock3, CloudUpload, Link2, Pause, Play, RotateCcw,
  ScanLine, Square, Video } from 'lucide-react';
import { useApp } from '../App';
import { VideoPlayer } from '../components/VideoPlayer';
import { InfoRow, MetricCard, PageIntro, Panel, StatusPill } from '../components/UI';
import { api } from '../services/api';

export function VideoPage() {
  const { session, sessionId, frames, connection, busy, start, stop, reset, reconnect } = useApp();
  const [mode, setMode] = useState<'upload' | 'youtube'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [reconnectId, setReconnectId] = useState('');
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const active = Boolean(session && !['completed', 'failed', 'cancelled'].includes(session.status));
  const latest = session?.latest;

  const selectFile = (value?: File) => {
    if (!value) return;
    setFile(value);
    setMode('upload');
  };

  return <>
    <PageIntro eyebrow="LIVE VIDEO ANALYSIS" title="Analyze parking spaces"
      text="Submit an MP4 or an authorized YouTube video, then play it. Analysis follows playback at one sampled frame per second and pauses with the player." />
    <div className="video-page-grid">
      <div className="video-main">
        <Panel title="Source playback" eyebrow="VIDEO MONITOR" action={<StatusPill
          state={session?.status === 'failed' ? 'bad' : active ? 'good' : 'neutral'}
          label={session?.status ?? 'No session'} />}>
          <VideoPlayer session={session} frames={frames} />
          <div className="video-under"><span><ScanLine size={17} /> Predictions follow playback and stop when the video ends.</span>
            <span>{session?.source_type === 'youtube' ? 'YouTube uses the annotated preview.' : 'MP4 boxes follow sampled timestamps.'}</span></div>
        </Panel>
        <div className="metric-grid video-metrics">
          <MetricCard label="Empty spaces" value={latest?.counts.empty ?? '—'} detail="Latest frame" tone="green" icon={<Square size={19} />} />
          <MetricCard label="Occupied spaces" value={latest?.counts.occupied ?? '—'} detail="Latest frame" tone="red" icon={<Video size={19} />} />
          <MetricCard label="Total detected" value={latest?.counts.total ?? '—'} detail="Latest frame" tone="amber" icon={<ScanLine size={19} />} />
          <MetricCard label="Inference latency" value={latest ? `${latest.inference_time_ms.toFixed(1)} ms` : '—'}
            detail="Latest frame" icon={<Clock3 size={19} />} />
        </div>
        <div className="two-column video-lower">
          <Panel title="Annotated evidence" eyebrow="LATEST ANALYZED FRAME" action={<span className="muted-small">
            {latest ? `${latest.timestamp_seconds.toFixed(2)} s` : 'Waiting for frame'}</span>}>
            <div className="preview-stage">{latest && session ? <img
              src={api.preview(session.id, session.revision)} alt={`Annotated video frame at ${latest.timestamp_seconds.toFixed(2)} seconds`} /> :
              <div className="preview-empty"><ScanLine size={32} /><p>The first annotated frame will appear here.</p></div>}</div>
          </Panel>
          <Panel title="Processing details" eyebrow="SESSION TELEMETRY">
            <div className="info-list">
              <InfoRow label="Frames processed" value={session?.frames_processed ?? 0} />
              <InfoRow label="Analyzed timestamp" value={latest ? `${latest.timestamp_seconds.toFixed(2)} s` : '—'} />
              <InfoRow label="Decode time" value={latest ? `${latest.decode_time_ms.toFixed(1)} ms` : '—'} />
              <InfoRow label="Frame analysis" value={latest ? `${latest.analysis_time_ms.toFixed(1)} ms` : '—'} />
              <InfoRow label="Model version" value={latest?.model.version ?? '—'} />
              <InfoRow label="Updates" value={connection} />
            </div>
          </Panel>
        </div>
      </div>
      <div className="video-side">
        <Panel title="Start analysis" eyebrow="VIDEO SOURCE">
          <div className="mode-switch" role="tablist" aria-label="Video source">
            <button type="button" role="tab" aria-selected={mode === 'upload'}
              className={mode === 'upload' ? 'selected' : ''} onClick={() => setMode('upload')}><CloudUpload size={17} /> MP4 upload</button>
            <button type="button" role="tab" aria-selected={mode === 'youtube'}
              className={mode === 'youtube' ? 'selected' : ''} onClick={() => setMode('youtube')}><Link2 size={17} /> YouTube URL</button>
          </div>
          {mode === 'upload' ? <div className={`drop-zone ${dragging ? 'dragging' : ''}`} role="button" tabIndex={0}
            onClick={() => fileInput.current?.click()} onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') fileInput.current?.click();
            }} onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)} onDrop={(event) => {
              event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files[0]);
            }}>
            <span className="drop-icon"><CloudUpload size={25} /></span>
            <strong>{file?.name ?? 'Drop an MP4 here'}</strong>
            <span>{file ? `${(file.size / 1048576).toFixed(1)} MB · Click to change` : 'or click to browse your files'}</span>
            <input ref={fileInput} type="file" accept="video/mp4,.mp4" aria-label="Choose MP4 file"
              onChange={(event) => selectFile(event.target.files?.[0])} hidden />
          </div> : <label className="field-label">YouTube video URL
            <input type="url" value={url} onChange={(event) => setUrl(event.target.value)}
              placeholder="https://www.youtube.com/watch?v=..." />
            <small>Public or unlisted videos only. The backend validates the URL.</small>
          </label>}
          <button className="button button-primary button-wide" disabled={busy || active || (mode === 'upload' ? !file : !url.trim())}
            onClick={() => void start(mode === 'upload' ? { file: file! } : { url: url.trim() })}>
            {busy ? 'Starting…' : 'Start analysis'} <ArrowRight size={18} />
          </button>
          {active && <p className="form-hint">Stop the current analysis before starting another.</p>}
        </Panel>
        <Panel title="Session controls" eyebrow="ACTIVE SESSION">
          <div className="session-id"><span>SESSION ID</span><code>{sessionId ?? 'No session'}</code></div>
          <div className="session-actions">
            <button className="button button-outline" onClick={() => void stop()} disabled={!active}><Pause size={17} /> Stop / Cancel</button>
            <button className="button button-outline" onClick={() => void reset()} disabled={!sessionId && !busy}><RotateCcw size={17} /> Reset</button>
          </div>
          <label className="field-label reconnect-label">Reconnect by session ID
            <input value={reconnectId} onChange={(event) => setReconnectId(event.target.value)}
              placeholder="Paste an existing session ID" />
          </label>
          <button className="button button-muted button-wide" disabled={!reconnectId.trim() || busy}
            onClick={() => reconnect(reconnectId)}><Play size={17} /> Reconnect</button>
        </Panel>
        <Panel title="Analysis progress" eyebrow="PROCESSING STATUS">
          <div className="progress-top"><strong>{session?.status ?? 'Waiting'}</strong><span>{session?.frames_processed ?? 0} frames</span></div>
          <div className="progress-track"><div style={{ width: session?.duration_seconds && latest
            ? `${Math.min(100, latest.timestamp_seconds / session.duration_seconds * 100)}%` : '0%' }} /></div>
          <p className="form-hint">{session?.duration_seconds ? `Analyzed ${latest?.timestamp_seconds.toFixed(1) ?? '0'} of ${session.duration_seconds.toFixed(1)} seconds`
            : 'Progress appears when video duration is available.'}</p>
          {session?.error && <p className="inline-error"><CircleAlert size={17} />{session.error.message}</p>}
          {!session && <div className="waiting-note"><Square size={17} /> No analysis running</div>}
        </Panel>
      </div>
    </div>
  </>;
}
