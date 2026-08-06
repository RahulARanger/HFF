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

const RESOURCE_SERIES = [
  { field: "ram_rss_bytes", color: "#43b7e8", label: "RAM RSS", formatValue: formatBytes },
  { field: "ram_uss_bytes", color: "#9b8cff", label: "RAM USS", formatValue: formatBytes },
  { field: "gpu_memory_bytes", color: "#f0a35b", label: "VRAM", formatValue: formatBytes },
  { field: "cpu_utilization_percent", color: "#51d48a", label: "CPU tree", formatValue: (value) => `${Number(value).toFixed(1)}%` },
  { field: "gpu_utilization_percent", color: "#f17c86", label: "GPU util", formatValue: (value) => `${Number(value).toFixed(1)}%` },
];

function formatClock(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function ResourceLane({ item, samples }) {
  const values = samples.map((sample) => Number(sample[item.field])).filter(Number.isFinite);
  if (!values.length) {
    return <div className="resource-lane resource-lane-empty"><div className="resource-lane-header"><span className="resource-lane-label"><i style={{ background: item.color }} />{item.label}</span><span>Unavailable</span></div></div>;
  }
  const maximum = Math.max(...values, 1);
  const formatValue = item.formatValue || formatBytes;
  const latest = [...samples].reverse().map((sample) => Number(sample[item.field])).find(Number.isFinite);
  const pointFor = (sample, index) => {
    const value = Number(sample[item.field]);
    if (!Number.isFinite(value)) return null;
    const x = samples.length === 1 ? 50 : (index / (samples.length - 1)) * 100;
    return { x, y: 90 - (value / maximum) * 72 };
  };
  const points = samples
    .map((sample, index) => pointFor(sample, index))
    .filter(Boolean)
    .map(({ x, y }) => `${x},${y}`)
    .join(" ");
  const lastIndex = samples.map((sample) => Number(sample[item.field])).reduce((last, value, index) => Number.isFinite(value) ? index : last, -1);
  const lastPoint = lastIndex >= 0 ? pointFor(samples[lastIndex], lastIndex) : null;
  return (
    <div className="resource-lane">
      <div className="resource-lane-header"><span className="resource-lane-label"><i style={{ background: item.color }} />{item.label}</span><span>{formatValue(latest)} · peak {formatValue(maximum)}</span></div>
      <div className="resource-lane-plot">
        <div className="resource-lane-axis"><span>{formatValue(maximum)}</span><span>{item.formatValue === formatBytes ? "0 B" : "0%"}</span></div>
        <svg className="resource-lane-svg" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${item.label} over time`}>
          {[18, 54, 90].map((y) => <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="rgba(128, 155, 173, 0.18)" strokeWidth="1" />)}
          <polyline points={points} fill="none" stroke={item.color} strokeWidth="2.4" vectorEffect="non-scaling-stroke" />
          {lastPoint && <circle cx={lastPoint.x} cy={lastPoint.y} r="2.6" fill={item.color} vectorEffect="non-scaling-stroke" />}
        </svg>
      </div>
    </div>
  );
}

function ResourceLineChart({ samples }) {
  if (!samples.some((sample) => RESOURCE_SERIES.some(({ field }) => Number.isFinite(Number(sample[field]))))) {
    return <div className="monitor-chart-empty resource-chart-empty">Waiting for resource samples.</div>;
  }
  return (
    <div className="resource-chart-shell">
      {RESOURCE_SERIES.map((item) => <ResourceLane key={item.field} item={item} samples={samples} />)}
      <div className="resource-chart-time-axis"><span>{formatClock(samples[0]?.timestamp_utc)}</span><span>{formatClock(samples.at(-1)?.timestamp_utc)}</span></div>
    </div>
  );
}

function ResourceDrilldown({ samples }) {
  return (
    <section className="resource-drilldown">
      <div className="monitor-panel-heading"><span>Resource timeline</span><span>{samples.length} samples</span></div>
      <div className="resource-chart-wrap"><ResourceLineChart samples={samples} /></div>
    </section>
  );
}

function EpochLineChart({ history }) {
  const rows = history.filter((row) => Number.isFinite(Number(row.epoch)));
  if (!rows.length) return <div className="monitor-chart-empty epoch-chart-empty">Epoch metrics will appear after the first validation.</div>;
  const series = [
    { field: "train_loss_total", color: "#43b7e8", label: "Train loss" },
    { field: "validation_loss_branch_1", color: "#f0a35b", label: "Validation loss" },
  ];
  const values = series.flatMap(({ field }) => rows.map((row) => Number(row[field])).filter(Number.isFinite));
  const maximum = Math.max(...values, 1);
  const minimum = Math.min(...values, 0);
  const range = Math.max(maximum - minimum, 1);
  const pointFor = (field, row, index) => {
    const value = Number(row[field]);
    if (!Number.isFinite(value)) return null;
    const x = rows.length === 1 ? 50 : (index / (rows.length - 1)) * 100;
    return { x, y: 92 - ((value - minimum) / range) * 78 };
  };
  const pointsFor = (field) => rows
    .map((row, index) => pointFor(field, row, index))
    .filter(Boolean)
    .map(({ x, y }) => `${x},${y}`)
    .filter(Boolean)
    .join(" ");
  return (
    <div>
      <div className="epoch-chart-legend">{series.map((item) => <span key={item.field}><i style={{ background: item.color }} />{item.label}</span>)}</div>
      <svg className="epoch-chart" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Training and validation loss by epoch">
        <line x1="0" y1="92" x2="100" y2="92" stroke="rgba(128, 155, 173, 0.2)" strokeWidth="1" />
        {series.map((item) => (
          <g key={item.field}>
            <polyline points={pointsFor(item.field)} fill="none" stroke={item.color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
            {rows.map((row, index) => {
              const point = pointFor(item.field, row, index);
              return point ? <circle key={`${item.field}-${index}`} cx={point.x} cy={point.y} r="2.2" fill={item.color} vectorEffect="non-scaling-stroke" /> : null;
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}

const DIAGNOSTIC_SERIES = [
  { field: "learning_rate", color: "#51d48a", label: "Learning rate", formatValue: (value) => Number(value).toExponential(2) },
  { field: "gradient_norm_last_step", color: "#d889ff", label: "Gradient norm", formatValue: (value) => Number(value).toFixed(3) },
  { field: "train_samples_per_second", color: "#43b7e8", label: "Samples / sec", formatValue: (value) => Number(value).toFixed(2) },
  { field: "train_data_wait_seconds", color: "#f0a35b", label: "Data wait / epoch", formatValue: (value) => `${Number(value).toFixed(1)}s` },
  { field: "epoch_seconds", color: "#f17c86", label: "Epoch duration", formatValue: (value) => `${Number(value).toFixed(1)}s` },
];

function EpochDiagnosticLane({ item, rows }) {
  const values = rows.map((row) => Number(row[item.field])).filter(Number.isFinite);
  if (!values.length) return null;
  const maximum = Math.max(...values, 1);
  const latest = [...rows].reverse().map((row) => Number(row[item.field])).find(Number.isFinite);
  const pointFor = (row, index) => {
    const value = Number(row[item.field]);
    if (!Number.isFinite(value)) return null;
    const x = rows.length === 1 ? 50 : (index / (rows.length - 1)) * 100;
    return { x, y: 90 - (value / maximum) * 72 };
  };
  const points = rows
    .map((row, index) => pointFor(row, index))
    .filter(Boolean)
    .map(({ x, y }) => `${x},${y}`)
    .join(" ");
  const lastIndex = rows.map((row) => Number(row[item.field])).reduce((last, value, index) => Number.isFinite(value) ? index : last, -1);
  const lastPoint = lastIndex >= 0 ? pointFor(rows[lastIndex], lastIndex) : null;
  return (
    <article className="epoch-diagnostic-lane">
      <div className="epoch-diagnostic-header"><span><i style={{ background: item.color }} />{item.label}</span><strong>{item.formatValue(latest)}</strong></div>
      <div className="epoch-diagnostic-plot">
        <span>{item.formatValue(maximum)}</span>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${item.label} by epoch`}>
          {[18, 54, 90].map((y) => <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="rgba(128, 155, 173, 0.18)" strokeWidth="1" />)}
          <polyline points={points} fill="none" stroke={item.color} strokeWidth="2.4" vectorEffect="non-scaling-stroke" />
          {lastPoint && <circle cx={lastPoint.x} cy={lastPoint.y} r="2.6" fill={item.color} vectorEffect="non-scaling-stroke" />}
        </svg>
        <span>0</span>
      </div>
    </article>
  );
}

function EpochDiagnostics({ history }) {
  const rows = history.filter((row) => Number.isFinite(Number(row.epoch)));
  const available = DIAGNOSTIC_SERIES.filter((item) => rows.some((row) => Number.isFinite(Number(row[item.field]))));
  if (!available.length) return null;
  const nonFiniteBatches = rows.reduce((total, row) => total + (Number(row.non_finite_batches) || 0), 0);
  return (
    <section className="epoch-diagnostics">
      <div className="monitor-panel-heading"><span>Training diagnostics</span><span>{nonFiniteBatches ? `${nonFiniteBatches} non-finite batches` : "No NaN / Inf batches"}</span></div>
      <div className="epoch-diagnostics-grid">{available.map((item) => <EpochDiagnosticLane key={item.field} item={item} rows={rows} />)}</div>
    </section>
  );
}

function EpochProgress({ training }) {
  const completed = training?.completed_epochs ?? 0;
  const total = training?.total_epochs;
  const pending = training?.pending_epochs;
  const percent = training?.progress_percent ?? 0;
  return (
    <section className="epoch-progress-card">
      <div className="epoch-progress-heading">
        <div><div className="monitor-metric-label">Epoch progress</div><strong>{completed} / {total ?? "—"} epochs</strong></div>
        <div className="epoch-progress-pending">{pending === null || pending === undefined ? "Pending unavailable" : `${pending} pending`}<span>{percent}%</span></div>
      </div>
      <div className="epoch-progress-track"><span style={{ width: `${Math.min(Math.max(percent, 0), 100)}%` }} /></div>
      <div className="epoch-progress-caption">Progress is based on completed validation epochs recorded in <code>training_metrics.json</code>.</div>
    </section>
  );
}

function EpochDrilldown({ training }) {
  const history = training?.history || [];
  return (
    <section className="epoch-drilldown">
      <div className="monitor-panel-heading"><span>Epoch timeline</span><span>{history.length} recorded</span></div>
      <div className="epoch-chart-wrap"><EpochLineChart history={history} /></div>
      <EpochDiagnostics history={history} />
      {!history.length && <div className="monitor-empty epoch-empty">No completed validation epochs have been written yet.</div>}
    </section>
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
  const training = run.training || {};
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
      <div className="monitor-run-card-epochs"><span>Epochs <strong>{training.completed_epochs ?? 0} / {training.total_epochs ?? "—"}</strong></span><span>{training.pending_epochs ?? "—"} pending</span></div>
    </button>
  );
}

export default function MonitorView() {
  const [runs, setRuns] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [showResourceDetail, setShowResourceDetail] = useState(false);
  const [showEpochDetail, setShowEpochDetail] = useState(false);

  const refreshRuns = useCallback(async () => {
    const payload = await fetchMonitorRuns();
    const nextRuns = payload.runs || [];
    setRuns(nextRuns);
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
  const epochsCompleted = runs.reduce((total, run) => total + (run.training?.completed_epochs || 0), 0);
  const epochsPending = runs.reduce((total, run) => total + (run.training?.pending_epochs || 0), 0);
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
        <div><span className="monitor-summary-label">Epochs completed</span><strong>{epochsCompleted}</strong></div>
        <div><span className="monitor-summary-label">Epochs pending</span><strong>{epochsPending}</strong></div>
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

              <div className="monitor-drilldown-toggle"><span>RAM / VRAM resource history</span><button className="monitor-refresh-button" onClick={() => setShowResourceDetail((current) => !current)}>{showResourceDetail ? "Hide resource drilldown" : "Open resource drilldown"}</button></div>
              {showResourceDetail && <ResourceDrilldown samples={samples} />}

              <EpochProgress training={detail.training} />
              <div className="monitor-drilldown-toggle"><span>Epoch-level training history</span><button className="monitor-refresh-button" onClick={() => setShowEpochDetail((current) => !current)}>{showEpochDetail ? "Hide drilldown" : "Open drilldown"}</button></div>
              {showEpochDetail && <EpochDrilldown training={detail.training} />}

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
