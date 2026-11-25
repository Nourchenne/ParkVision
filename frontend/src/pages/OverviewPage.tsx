import { Activity, ArrowUpRight, CircleCheck, CircleDashed, Clock3, Layers3, ParkingSquare, RadioTower } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useApp } from '../App';
import { InfoRow, MetricCard, PageIntro, Panel, Skeleton, StatusPill } from '../components/UI';

export function OverviewPage() {
  const { health, ready, model, session, connection } = useApp();
  const latest = session?.latest;
  return <>
    <PageIntro eyebrow="OPERATIONS OVERVIEW" title="Parking intelligence, at a glance."
      text="A live view of your loaded model and the current video analysis. All counts come directly from analyzed frames."
      link={{ to: '/video', label: 'Open video analysis' }} />
    <div className="summary-strip">
      <div className="summary-icon"><RadioTower size={24} /></div>
      <div><strong>{session ? `Session ${session.status}` : 'Ready for an analysis'}</strong>
        <p>{session ? `${session.source_type === 'upload' ? 'Uploaded MP4' : 'YouTube video'} · ${session.frames_processed} frames processed`
          : 'Start a video or reconnect to an existing session.'}</p></div>
      <StatusPill state={session ? session.status === 'failed' ? 'bad' : 'good' : 'neutral'}
        label={session ? session.status : 'No session'} />
    </div>
    <div className="metric-grid">
      <MetricCard label="Empty spaces" value={latest?.counts.empty ?? '—'} detail="Latest analyzed frame" tone="green" icon={<ParkingSquare size={20} />} />
      <MetricCard label="Occupied spaces" value={latest?.counts.occupied ?? '—'} detail="Latest analyzed frame" tone="red" icon={<Layers3 size={20} />} />
      <MetricCard label="Total detected" value={latest?.counts.total ?? '—'} detail="Empty + occupied" tone="amber" icon={<CircleCheck size={20} />} />
      <MetricCard label="Inference latency" value={latest ? `${latest.inference_time_ms.toFixed(1)} ms` : '—'}
        detail="Latest analyzed frame" icon={<Clock3 size={20} />} />
    </div>
    <div className="two-column">
      <Panel title="Current analysis" eyebrow="LIVE SESSION" action={<Link className="text-link" to="/video">View analysis <ArrowUpRight size={16} /></Link>}>
        {session ? <div className="info-list">
          <InfoRow label="Status" value={<StatusPill state={session.status === 'failed' ? 'bad' :
            session.status === 'completed' ? 'good' : 'wait'} label={session.status} />} />
          <InfoRow label="Frames processed" value={session.frames_processed} />
          <InfoRow label="Analyzed timestamp" value={latest ? `${latest.timestamp_seconds.toFixed(2)} s` : 'Waiting for first frame'} />
          <InfoRow label="Updates" value={connection} />
          {session.error && <p className="inline-error">{session.error.message}</p>}
        </div> : <div className="empty-panel"><CircleDashed size={31} /><strong>No session yet</strong>
          <p>Start an MP4 or YouTube analysis to see live estimates here.</p></div>}
      </Panel>
      <Panel title="Serving status" eyebrow="MODEL & API" action={<Link className="text-link" to="/model">Model details <ArrowUpRight size={16} /></Link>}>
        <div className="info-list">
          <InfoRow label="API connectivity" value={health === null ? <Skeleton className="skeleton-small" /> : health ? 'Connected' : 'Unavailable'} />
          <InfoRow label="Model readiness" value={ready === null ? <Skeleton className="skeleton-small" /> : ready ? 'Ready' : 'Unavailable'} />
          <InfoRow label="Champion" value={model?.name ?? '—'} />
          <InfoRow label="Loaded version" value={model?.version ?? '—'} />
          <InfoRow label="Alias" value={model?.alias ?? '—'} />
        </div>
      </Panel>
    </div>
    <div className="callout"><Activity size={20} /><span>Counts are sampled per-frame detections. They are not persistent parking-space IDs or historical occupancy trends.</span></div>
  </>;
}
