import type { ReactNode } from 'react';
import { ArrowUpRight } from 'lucide-react';
import { Link } from 'react-router-dom';

export function StatusPill({ state, label }: { state: 'good' | 'bad' | 'wait' | 'neutral'; label: string }) {
  return <span className={`status-pill status-${state}`}><span className="status-dot" />{label}</span>;
}

export function Panel({ title, eyebrow, action, children, className = '' }: {
  title: string; eyebrow?: string; action?: ReactNode; children: ReactNode; className?: string;
}) {
  return <section className={`panel ${className}`}>
    <div className="panel-head">
      <div>{eyebrow && <p className="eyebrow">{eyebrow}</p>}<h2>{title}</h2></div>
      {action}
    </div>
    {children}
  </section>;
}

export function MetricCard({ label, value, detail, tone = 'neutral', icon }: {
  label: string; value: string | number; detail?: string; tone?: 'neutral' | 'green' | 'red' | 'amber'; icon: ReactNode;
}) {
  return <article className={`metric-card metric-${tone}`}>
    <div className="metric-top"><span>{label}</span><span className="metric-icon">{icon}</span></div>
    <strong>{value}</strong><small>{detail ?? '\u00a0'}</small>
  </article>;
}

export function PageIntro({ eyebrow, title, text, link }: {
  eyebrow: string; title: string; text: string; link?: { to: string; label: string };
}) {
  return <div className="page-intro">
    <div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{text}</p></div>
    {link && <Link className="button button-primary" to={link.to}>{link.label}<ArrowUpRight size={17} /></Link>}
  </div>;
}

export function InfoRow({ label, value }: { label: string; value: ReactNode }) {
  return <div className="info-row"><span>{label}</span><strong>{value}</strong></div>;
}

export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}
