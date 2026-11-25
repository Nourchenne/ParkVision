import { BrainCircuit, CircleAlert, Fingerprint, ShieldCheck } from 'lucide-react';
import { useApp } from '../App';
import { InfoRow, PageIntro, Panel, Skeleton, StatusPill } from '../components/UI';

export function ModelPage() {
  const { model, ready, health } = useApp();
  return <>
    <PageIntro eyebrow="MODEL REGISTRY" title="Model information"
      text="The exact MLflow champion version loaded by FastAPI. This page reads serving metadata and does not change the registry." />
    <div className="model-hero panel"><div className="model-hero-icon"><BrainCircuit size={32} /></div>
      <div className="model-hero-text"><p className="eyebrow">ACTIVE CHAMPION</p>
        <h2>{model?.name ?? 'Model details unavailable'}</h2>
        <p>{model?.architecture ?? 'Architecture unavailable'} · alias {model?.alias ?? '—'}</p></div>
      <StatusPill state={ready === null ? 'wait' : ready ? 'good' : 'bad'} label={ready ? 'Ready for inference' : 'Unavailable'} />
    </div>
    <div className="two-column">
      <Panel title="Registry identity" eyebrow="LOADED MODEL">
        {model ? <div className="info-list">
          <InfoRow label="Registered name" value={model.name} />
          <InfoRow label="Champion alias" value={model.alias} />
          <InfoRow label="Exact version" value={model.version ?? 'Unavailable'} />
          <InfoRow label="Architecture" value={model.architecture ?? 'Unavailable'} />
          <InfoRow label="Source run ID" value={<code>{model.source_run_id ?? 'Unavailable'}</code>} />
          <InfoRow label="Model URI" value={<code>{model.model_uri ?? 'Unavailable'}</code>} />
        </div> : health === null ? <Skeleton className="skeleton-panel" /> :
          <div className="empty-panel"><CircleAlert size={30} /><strong>Metadata unavailable</strong>
            <p>Check the API connection and model startup logs.</p></div>}
      </Panel>
      <Panel title="Serving checks" eyebrow="READINESS">
        <div className="info-list">
          <InfoRow label="FastAPI health" value={health === null ? 'Checking' : health ? 'Online' : 'Offline'} />
          <InfoRow label="Champion readiness" value={ready === null ? 'Checking' : ready ? 'Ready' : 'Unavailable'} />
          <InfoRow label="Model status" value={model?.status ?? 'Unavailable'} />
        </div>
        <div className="callout callout-inner"><ShieldCheck size={19} /><span>Inference uses the backend's verified model cache. No registry controls are exposed here.</span></div>
      </Panel>
    </div>
    <div className="footnote"><Fingerprint size={17} /> Version and run ID are reported by the backend; they are never inferred from a video result.</div>
  </>;
}
