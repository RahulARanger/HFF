import { useState } from "react";
import MonitorView from "./components/MonitorView.jsx";

function StatusDot({ state }) {
  return <span className={`status-dot ${state}`} aria-hidden="true" />;
}

export default function App() {
  const [railCollapsed, setRailCollapsed] = useState(false);

  return (
    <main className={`app-shell ${railCollapsed ? "rail-collapsed" : ""}`}>
      <aside className="control-rail">
        <div className="brand-block">
          <div className="brand-mark">H</div>
          <div>
            <div className="brand-name">HFF-Net</div>
            <div className="brand-subtitle">BraTS viewer</div>
          </div>
          <button
            className="rail-toggle"
            type="button"
            onClick={() => setRailCollapsed((current) => !current)}
            aria-label={railCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={railCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {railCollapsed ? "›" : "‹"}
          </button>
        </div>

        <div className="rail-scroll">
          <section className="rail-section monitor-rail-note">
            <div className="section-kicker">Training telemetry</div>
            <p>Process-scoped RAM and accelerator samples from cross-validation folds.</p>
            <div className="selected-path">Logs refresh automatically while a fold is running.</div>
          </section>

          <section className="rail-section legend-section">
            <div className="section-kicker">Segmentation labels</div>
            <div className="legend-row"><span className="legend-swatch core" /> Necrotic / core</div>
            <div className="legend-row"><span className="legend-swatch edema" /> Edema</div>
            <div className="legend-row"><span className="legend-swatch enhancing" /> Enhancing tumour</div>
          </section>
        </div>

        <div className="rail-footer">
          <StatusDot state="ready" />
          <span>Viewer ready</span>
        </div>
      </aside>

      <section className="workspace monitor-workspace">
        <header className="topbar">
          <div className="breadcrumb"><span>Research workspace</span><span className="crumb-divider">/</span><strong>Training monitor</strong></div>
        </header>
        <MonitorView />
      </section>
    </main>
  );
}
