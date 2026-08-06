import { useCallback, useEffect, useState } from "react";

import { fetchMonitorRun, fetchMonitorRuns } from "../api.js";

const POLL_INTERVAL_MS = 5000;

function formatBytes(value) {
  if (value === null || value === undefined) return "Unavailable";
  const number = Number(value);
  if (!Number.isFinite(number)) return "Unavailable";
  if (number < 1024) return `${number} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let scaled = number;
  let unit = "B";
  for (const nextUnit of units) {
    scaled /= 1024;
    unit = nextUnit;
    if (scaled < 1024) break;
  }
  return `${scaled.toFixed(scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2)} ${unit}`;
}

function formatTime(value) {
  if (!value) return "No samples yet";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown time" : date.toLocaleString([], { dateStyle: "medium", timeStyle: "medium" });
}

function StatusBadge({ status }) {
  return <span className={`monitor-status ${status}`}><span className="monitor-status-dot" />{status}</span>;
}

function Sparkline({ samples, field, color }) {
  const values = samples.map((sample) => sample[field]).filter((value) => value !== null && value !== undefined).map(Number).filter(Number.isFinite);
  if (values.length < 2) return <div className="monitor-chart-empty">Waiting for samples</div>;
  const maximum = Math.max(...values, 1);
  const minimum = Math.min(...values, 0);
  const range = Math.max(maximum - minimum, 1);
  const points = values.map((value, index) => `${(index / (values.length - 1)) * 100},${92 - ((value - minimum) / range) * 78}`).join(" ");
  return (
    <svg className="monitor-sparkline" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${field} over time`}>
      <line x1="0" y1="92" x2="100" y2="92" stroke="rgba(128, 155, 173, 0.2)" strokeWidth="1" />
      <polyline points={points} fill="none" stroke={color} strokeWidth="2.2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function MetricCard({ label, value, detail, children, tone = "blue" }) {
  return (
    <article className={`monitor-metric-card ${tone}`}>
      <div className="monitor-metric-label">{label}</div>
      <div className="monitor-metric-value">{value}</div>
      <div className="monitor-metric-detail">{detail}</div>
      {children}
    </article>
  );
}

function RunCard({ run, selected, onClick }) {
  const latest = run.latest || {};
  return (
    <button className={`monitor-run-card ${selected ? "selected" : ""}`} onClick={onClick}>
      <div className="monitor-run-card-topline">
        <span className="monitor-run-label">{run.label}</span>
        <StatusBadge status={run.status} />
      </div>
      <div className="monitor-run-card-meta">
        <span>{String(run.backend).toUpperCase()}</span>
        <span>{run.sample_count} samples</span>
        <span>{formatTime(run.updated_at)}</span>
      </div>
      <div className="monitor-run-card-values">
        <span>RAM <strong>{formatBytes(latest.ram_rss_bytes)}</strong></span>
        <span>VRAM <strong>{formatBytes(latest.gpu_memory_bytes)}</strong></span>
      </div>
    </button>
  );
}

export default function MonitorView() {
  const [runs, setRuns] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const refreshRuns = useCallback(async () => {
    const payload = await fetchMonitorRuns();
    const nextRuns = payload.runs || [];
    setRuns(nextRuns);
    setLastUpdated(payload.updated_at || new Date().toISOString());
    setSelectedId((current) => current && nextRuns.some((run) => run.id === current) ? current : nextRuns[0]?.id || "");
    return nextRuns;
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        await refreshRuns();
        if (!cancelled) setError("");
      } catch (pollError) {
        if (!cancelled) setError(pollError.message);
      }
    };
    poll();
    const timer = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [refreshRuns]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return undefined;
    }
    let cancelled = false;
    const pollDetail = async () => {
      try {
        const nextDetail = await fetchMonitorRun(selectedId);
        if (!cancelled) {
          setDetail(nextDetail);
          setError("");
        }
      } catch (pollError) {
        if (!cancelled) setError(pollError.message);
      }
    };
    pollDetail();
    const timer = window.setInterval(pollDetail, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedId]);

  const activeCount = runs.filter((run) => run.status === "running").length;
  const completedCount = runs.filter((run) => run.status === "completed").length;
  const samples = detail?.samples || [];
  const latest = detail?.latest || {};
  const gpuUnavailable = latest.gpu_memory_bytes === null || latest.gpu_memory_bytes === undefined;
  const peakGpu = detail?.peak?.gpu_memory_bytes ?? latest.gpu_memory_bytes;
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await refreshRuns();
      if (selectedId) setDetail(await fetchMonitorRun(selectedId));
      setError("");
    } catch (refreshError) {
      setError(refreshError.message);
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <section className="monitor-page" aria-label="Training monitor">
      <header className="monitor-header">
        <div>
          <div className="monitor-kicker">Cross-validation telemetry</div>
          <h1>Training monitor</h1>
          <p>Live and completed fold resource usage from the process-scoped monitors.</p>
        </div>
        <div className="monitor-header-actions">
          <span className="monitor-refresh-state"><span className="monitor-live-pulse" /> Auto-refresh · 5s</span>
          <button className="monitor-refresh-button" onClick={handleRefresh} disabled={refreshing}>{refreshing ? "Refreshing…" : "Refresh now"}</button>
        </div>
      </header>

      <div className="monitor-summary-strip">
        <div><span className="monitor-summary-label">Active folds</span><strong>{activeCount}</strong></div>
        <div><span className="monitor-summary-label">Completed folds</span><strong>{completedCount}</strong></div>
        <div><span className="monitor-summary-label">Tracked runs</span><strong>{runs.length}</strong></div>
        <div><span className="monitor-summary-label">Last index</span><strong>{lastUpdated ? formatTime(lastUpdated) : "Waiting"}</strong></div>
      </div>

      {error && <div className="app-alert monitor-alert">{error}</div>}

      <div className="monitor-layout">
        <aside className="monitor-run-list">
          <div className="monitor-panel-heading"><span>Runs</span><span>{runs.length}</span></div>
          {runs.length ? runs.map((run) => <RunCard key={run.id} run={run} selected={run.id === selectedId} onClick={() => setSelectedId(run.id)} />) : <div className="monitor-empty"><strong>No monitor logs found</strong><span>Start cross_train or train.py to populate this view.</span></div>}
        </aside>

        <section className="monitor-detail-panel">
          {detail ? (
            <>
              <header className="monitor-detail-header">
                <div>
                  <div className="monitor-detail-eyebrow">{detail.run_name} / {detail.fold}</div>
                  <h2>{detail.label}</h2>
                </div>
                <div className="monitor-detail-status"><StatusBadge status={detail.status} /><span>Updated {formatTime(detail.updated_at)}</span></div>
              </header>

              <div className="monitor-metrics-grid">
                <MetricCard label="RAM RSS" value={formatBytes(latest.ram_rss_bytes)} detail={`Peak ${formatBytes(detail.peak?.ram_rss_bytes)}`} tone="blue">
                  <Sparkline samples={samples} field="ram_rss_bytes" color="#43b7e8" />
                </MetricCard>
                <MetricCard label="RAM USS" value={formatBytes(latest.ram_uss_bytes)} detail={`Peak ${formatBytes(detail.peak?.ram_uss_bytes)}`} tone="violet">
                  <Sparkline samples={samples} field="ram_uss_bytes" color="#9b8cff" />
                </MetricCard>
                <MetricCard label="VRAM" value={formatBytes(latest.gpu_memory_bytes)} detail={gpuUnavailable ? "NVML telemetry unavailable" : `Peak ${formatBytes(peakGpu)}`} tone="orange">
                  <Sparkline samples={samples} field="gpu_memory_bytes" color="#f0a35b" />
                </MetricCard>
              </div>

              <div className="monitor-detail-grid">
                <section className="monitor-data-panel">
                  <div className="monitor-panel-heading"><span>Latest sample</span><span>{detail.sample_count} total</span></div>
                  <dl className="monitor-data-list">
                    <div><dt>Backend</dt><dd>{String(detail.backend).toUpperCase()}</dd></div>
                    <div><dt>Root PID</dt><dd>{detail.root_pid || "—"}{detail.process_visible ? " · visible" : ""}</dd></div>
                    <div><dt>Tracked processes</dt><dd>{latest.tracked_pids?.length ?? "—"}</dd></div>
                    <div><dt>Interval</dt><dd>{detail.interval_seconds}s</dd></div>
                    <div><dt>Last sample</dt><dd>{formatTime(latest.timestamp_utc)}</dd></div>
                    <div><dt>Log file</dt><dd className="monitor-path" title={detail.resource_log}>{detail.resource_log}</dd></div>
                  </dl>
                </section>
                <section className="monitor-data-panel monitor-note-panel">
                  <div className="monitor-panel-heading"><span>Telemetry note</span><span className="monitor-note-mark">i</span></div>
                  <p>{gpuUnavailable ? "RAM samples are available, but the NVIDIA process query did not return VRAM for this run. Check the training job output for the NVML warning and verify nvidia-ml-py is installed in the job environment." : "VRAM is limited to the selected training process tree. Other processes sharing the device are excluded."}</p>
                  <div className="monitor-process-line"><span className="monitor-live-pulse" /> {detail.process_visible ? "Training process visible on this host" : detail.status === "completed" ? "Training process exited normally" : "Process is not visible from this viewer host"}</div>
                </section>
              </div>
            </>
          ) : (
            <div className="monitor-empty monitor-detail-empty"><strong>Select a run</strong><span>Choose a fold from the list to inspect its resource history.</span></div>
          )}
        </section>
      </div>
    </section>
  );
}
