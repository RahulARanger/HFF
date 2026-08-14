import { useState } from "react";

import MonitorView from "./components/MonitorView.jsx";
import EvaluationView from "./components/EvaluationView.jsx";
import ValidationMonitorView from "./components/ValidationMonitorView.jsx";

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
  const [activeView, setActiveView] = useState("monitor");

  return (
    <main className="app-shell">
      <aside className="control-rail">
        <div className="brand-block">
          <BrandMark />
          <div>
            <div className="brand-name">HFF-Net</div>
            <div className="brand-subtitle">Research control room</div>
          </div>
        </div>

        <div className="rail-scroll">
          <section className="rail-section rail-status-card">
            <StatusDot state="ready" />
            <strong>Viewer ready</strong>
          </section>

          <nav className="rail-section rail-navigation" aria-label="Viewer sections">
            <button type="button" className={`rail-nav-button ${activeView === "monitor" ? "active" : ""}`} onClick={() => setActiveView("monitor")}><span className="rail-icon bars" aria-hidden="true"><i /><i /><i /></span><span><strong>Training telemetry</strong><small>Live and completed folds</small></span></button>
            <button type="button" className={`rail-nav-button ${activeView === "evaluation" ? "active" : ""}`} onClick={() => setActiveView("evaluation")}><span className="rail-icon target" aria-hidden="true">◎</span><span><strong>Run evaluation</strong><small>Launch checkpoint inference</small></span></button>
            <button type="button" className={`rail-nav-button ${activeView === "validation" ? "active" : ""}`} onClick={() => setActiveView("validation")}><span className="rail-icon pulse" aria-hidden="true">◌</span><span><strong>Validation monitor</strong><small>RAM, VRAM, CPU, GPU telemetry</small></span></button>
          </nav>
        </div>
      </aside>

      <section className="workspace monitor-workspace">
        {activeView === "evaluation" ? <EvaluationView /> : activeView === "validation" ? <ValidationMonitorView /> : <MonitorView />}
      </section>
    </main>
  );
}
