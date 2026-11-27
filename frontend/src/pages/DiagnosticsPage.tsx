import { Activity, Braces, Cable, CircleAlert, CircleCheck, Radio } from 'lucide-react';
import { useApp } from '../App';
import { InfoRow, PageIntro, Panel, StatusPill } from '../components/UI';

export function DiagnosticsPage() {
  const { health, ready, session, sessionId, connection, error } = useApp();
  return <>
    <PageIntro eyebrow="DEVELOPER TOOLS" title="API diagnostics"
      text="Current backend connectivity, session transport, and the latest real frame response. No synthetic history is shown." />
    <div className="diagnostic-grid">
      <Panel title="API health" eyebrow="GET /HEALTH" className="diagnostic-card" tone="dark">
        <div className="diagnostic-value"><Activity size={24} /><StatusPill state={health === null ? 'wait' : health ? 'good' : 'bad'}
          label={health === null ? 'Checking' : health ? 'Online' : 'Offline'} /></div>
      </Panel>
      <Panel title="Model readiness" eyebrow="GET /READY" className="diagnostic-card" tone="dark">
        <div className="diagnostic-value">{ready ? <CircleCheck size={24} /> : <CircleAlert size={24} />}
          <StatusPill state={ready === null ? 'wait' : ready ? 'good' : 'bad'}
            label={ready === null ? 'Checking' : ready ? 'Ready' : 'Unavailable'} /></div>
      </Panel>
      <Panel title="Live transport" eyebrow="SESSION EVENTS" className="diagnostic-card" tone="dark">
        <div className="diagnostic-value"><Radio size={24} /><StatusPill state={connection === 'live' ? 'good' :
          connection === 'polling' ? 'wait' : 'neutral'} label={connection} /></div>
      </Panel>
    </div>
    <div className="two-column">
      <Panel title="Session connection" eyebrow="CURRENT STATE" tone="blue">
        <div className="info-list">
          <InfoRow label="Session ID" value={<code>{sessionId ?? 'None'}</code>} />
          <InfoRow label="Status" value={session?.status ?? 'No session'} />
          <InfoRow label="Source" value={session?.source_type ?? '—'} />
          <InfoRow label="Revision" value={session?.revision ?? '—'} />
          <InfoRow label="Last backend error" value={session?.error?.message ?? error ?? 'None'} />
        </div>
      </Panel>
      <Panel title="Latest prediction JSON" eyebrow="REAL RESPONSE" action={<Braces size={20} className="muted-icon" />}>
        {session?.latest ? <pre className="json-block">{JSON.stringify(session.latest, null, 2)}</pre> :
          <div className="empty-panel"><Cable size={29} /><strong>No prediction received</strong>
            <p>Analyze a video to inspect the latest frame result.</p></div>}
      </Panel>
    </div>
  </>;
}
