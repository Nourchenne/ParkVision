import { useState } from 'react';
import { Activity, ArrowLeftToLine, ArrowRightToLine, BrainCircuit, ChartNoAxesCombined,
  CircleHelp, Image as ImageIcon, Menu, Radio, Video, X } from 'lucide-react';
import { NavLink, Outlet } from 'react-router-dom';
import { useApp } from '../App';
import { StatusPill } from '../components/UI';

const links = [
  { to: '/', label: 'Overview', icon: ChartNoAxesCombined, end: true },
  { to: '/video', label: 'Video Analysis', icon: Video, end: false },
  { to: '/image', label: 'Image Analysis', icon: ImageIcon, end: false },
  { to: '/model', label: 'Model Information', icon: BrainCircuit, end: false },
  { to: '/diagnostics', label: 'API Diagnostics', icon: Activity, end: false },
];

export function Layout() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { health, ready, session, error, clearError } = useApp();
  return <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
    {mobileOpen && <button className="mobile-scrim" aria-label="Close menu" onClick={() => setMobileOpen(false)} />}
    <aside className={`sidebar ${mobileOpen ? 'mobile-open' : ''}`}>
      <div className="brand-row">
        <span className="brand-mark" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 8V5a1 1 0 0 1 1-1h3" />
            <path d="M16 4h3a1 1 0 0 1 1 1v3" />
            <path d="M20 16v3a1 1 0 0 1-1 1h-3" />
            <path d="M8 20H5a1 1 0 0 1-1-1v-3" />
            <circle cx="12" cy="12" r="2.3" fill="currentColor" stroke="none" />
          </svg>
        </span>
        {!collapsed && <div className="brand-copy"><strong>ParkVision</strong><small>OPERATIONS CONSOLE</small></div>}
        <button className="icon-button mobile-close" aria-label="Close menu" onClick={() => setMobileOpen(false)}><X size={20} /></button>
      </div>
      <div className="nav-label">{collapsed ? '—' : 'WORKSPACE'}</div>
      <nav aria-label="Main navigation">
        {links.map(({ to, label, icon: Icon, end }) => <NavLink key={to} to={to} end={end}
          onClick={() => setMobileOpen(false)} title={collapsed ? label : undefined}
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <Icon size={20} strokeWidth={1.8} />{!collapsed && <span>{label}</span>}
        </NavLink>)}
      </nav>
      <div className="sidebar-bottom">
        {!collapsed && <div className="sidebar-note"><CircleHelp size={18} />
          <span>Local development workspace. Results are sampled estimates.</span></div>}
        <button className="collapse-button" onClick={() => setCollapsed(!collapsed)}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
          {collapsed ? <ArrowRightToLine size={18} /> : <ArrowLeftToLine size={18} />}
          {!collapsed && 'Collapse sidebar'}
        </button>
      </div>
    </aside>
    <div className="main-column">
      <header className="topbar">
        <div className="topbar-left"><button className="icon-button mobile-menu" aria-label="Open menu"
          onClick={() => setMobileOpen(true)}><Menu size={21} /></button>
          <span className="breadcrumb">WORKSPACE <span>/</span> PARKING MONITOR</span></div>
        <div className="topbar-right">
          <StatusPill state={health === null ? 'wait' : health ? 'good' : 'bad'}
            label={health === null ? 'Checking API' : health ? 'API online' : 'API offline'} />
          <StatusPill state={ready === null ? 'wait' : ready ? 'good' : 'bad'}
            label={ready === null ? 'Checking model' : ready ? 'Model ready' : 'Model unavailable'} />
        </div>
      </header>
      <main className="page-content"><Outlet /></main>
      <footer className="app-footer"><span>ParkVision · local monitoring</span>
        <span>{session ? `Session ${session.id.slice(0, 8)} · ${session.status}` : 'No active session'}</span></footer>
    </div>
    {error && <div className="toast" role="alert"><Radio size={19} /><span>{error}</span>
      <button aria-label="Dismiss message" onClick={clearError}><X size={17} /></button></div>}
  </div>;
}
