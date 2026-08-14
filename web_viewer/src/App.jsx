import MonitorView from "./components/MonitorView.jsx";

function StatusDot({ state }) {
  return <span className={`status-dot ${state}`} aria-hidden="true" />;
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span />
      <span />
      <span />
      <span />
      <span />
      <span />
    </div>
  );
}

export default function App() {
  return (
    <main className="app-shell">
      <aside className="control-rail">
        <div className="brand-block">
          <BrandMark />
          <div>
            <div className="brand-name">HFF-Net</div>
            <div className="brand-subtitle">Training monitor</div>
          </div>
        </div>

        <div className="rail-scroll">
          <section className="rail-section rail-status-card">
            <StatusDot state="ready" />
            <strong>Viewer ready</strong>
          </section>

          <section className="rail-section monitor-rail-note">
            <div className="rail-section-title"><span className="rail-icon bars" aria-hidden="true"><i /><i /><i /></span>Training telemetry</div>
            <p>Live and completed fold resource usage.</p>
          </section>
        </div>
      </aside>

      <section className="workspace monitor-workspace">
        <MonitorView />
      </section>
    </main>
  );
}
