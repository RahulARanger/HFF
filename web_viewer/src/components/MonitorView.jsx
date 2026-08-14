import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { fetchMonitorRun, fetchMonitorRuns } from "../api.js";
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Progress } from "./watermelon-ui.jsx";

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

function formatRelativeTime(value) {
  if (!value) return "Timing unavailable";
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return "Timing unavailable";
  const elapsedSeconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (elapsedSeconds < 60) return "just now";
  const minutes = Math.floor(elapsedSeconds / 60);
  if (minutes < 60) return `${minutes} min${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hr${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function formatDuration(seconds, fallback = "—") {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return fallback;
  const total = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes) return `${minutes}m ${String(remainder).padStart(2, "0")}s`;
  return `${remainder}s`;
}

function formatCompactTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown" : date.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function StatusBadge({ status }) {
  return <Badge variant="outline" className={`monitor-status ${status}`}><span className="monitor-status-dot" />{status}</Badge>;
}

function nearestPoint(event, count) {
  const bounds = event.currentTarget.getBoundingClientRect();
  const ratio = Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0), 1);
  return Math.round(ratio * Math.max(count - 1, 0));
}

function Sparkline({ samples, field, color }) {
  const points = samples.map((sample, index) => ({ sample, index, value: Number(sample[field]) })).filter(({ value }) => Number.isFinite(value));
  const [hoveredIndex, setHoveredIndex] = useState(null);
  if (points.length < 2) return <div className="monitor-chart-empty">Waiting for samples</div>;
  const values = points.map(({ value }) => value);
  const maximum = Math.max(...values, 1);
  const minimum = Math.min(...values, 0);
  const range = Math.max(maximum - minimum, 1);
  const polyline = points.map(({ value }, index) => `${(index / (points.length - 1)) * 100},${92 - ((value - minimum) / range) * 78}`).join(" ");
  const hovered = hoveredIndex === null ? null : points[hoveredIndex];
  return (
    <div className="sparkline-wrap">
      <svg className="monitor-sparkline" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${field} over time`} onMouseMove={(event) => setHoveredIndex(nearestPoint(event, points.length))} onMouseLeave={() => setHoveredIndex(null)}>
        <line x1="0" y1="92" x2="100" y2="92" stroke="rgba(128, 155, 173, 0.2)" strokeWidth="1" />
        <polyline points={polyline} fill="none" stroke={color} strokeWidth="2.2" vectorEffect="non-scaling-stroke" />
      </svg>
      {hovered && <div className="graph-tooltip sparkline-tooltip" style={{ left: `${(hoveredIndex / (points.length - 1)) * 100}%` }}>
        <strong>{formatBytes(hovered.value)}</strong>
        <span>{formatClock(hovered.sample.timestamp_utc)}</span>
      </div>}
    </div>
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
  const [hoveredIndex, setHoveredIndex] = useState(null);
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
  const hoveredSample = hoveredIndex === null ? null : samples[hoveredIndex];
  const hoveredValue = hoveredSample ? Number(hoveredSample[item.field]) : null;
  const hoveredPoint = hoveredSample ? pointFor(hoveredSample, hoveredIndex) : null;
  return (
    <div className="resource-lane">
      <div className="resource-lane-header"><span className="resource-lane-label"><i style={{ background: item.color }} />{item.label}</span><span>{formatValue(latest)} · peak {formatValue(maximum)}</span></div>
      <div className="resource-lane-plot">
        <div className="resource-lane-axis"><span>{formatValue(maximum)}</span><span>{item.formatValue === formatBytes ? "0 B" : "0%"}</span></div>
        <svg className="resource-lane-svg" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${item.label} over time`} onMouseMove={(event) => setHoveredIndex(nearestPoint(event, samples.length))} onMouseLeave={() => setHoveredIndex(null)}>
          {[18, 54, 90].map((y) => <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="rgba(128, 155, 173, 0.18)" strokeWidth="1" />)}
          <polyline points={points} fill="none" stroke={item.color} strokeWidth="2.4" vectorEffect="non-scaling-stroke" />
          {lastPoint && <circle cx={lastPoint.x} cy={lastPoint.y} r="2.6" fill={item.color} vectorEffect="non-scaling-stroke" />}
          {hoveredPoint && <line x1={hoveredPoint.x} y1="10" x2={hoveredPoint.x} y2="92" stroke={item.color} strokeOpacity="0.55" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />}
        </svg>
        {hoveredSample && Number.isFinite(hoveredValue) && <div className="graph-tooltip resource-tooltip" style={{ left: `${hoveredPoint.x}%` }}>
          <strong>{formatValue(hoveredValue)}</strong>
          <span>{formatClock(hoveredSample.timestamp_utc)}</span>
        </div>}
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
  const [hoveredIndex, setHoveredIndex] = useState(null);
  if (!rows.length) return <div className="monitor-chart-empty epoch-chart-empty">Epoch metrics will appear after the first validation.</div>;
  const trainLossField = rows.some((row) => Number.isFinite(Number(row.train_loss_supervised_branch_1)))
    ? "train_loss_supervised_branch_1"
    : "train_loss_total";
  const series = [
    { field: trainLossField, color: "#43b7e8", label: "Train loss" },
    { field: "validation_loss_branch_1", color: "#f0a35b", label: "Validation loss" },
  ];
  const values = series.flatMap(({ field }) => rows.map((row) => Number(row[field])).filter(Number.isFinite));
  const rawMaximum = values.length ? Math.max(...values) : 1;
  const rawMinimum = values.length ? Math.min(...values) : 0;
  const rawRange = rawMaximum - rawMinimum;
  const padding = rawRange > 0 ? rawRange * 0.12 : Math.max(Math.abs(rawMaximum) * 0.12, 0.01);
  const maximum = rawMaximum + padding;
  const minimum = Math.max(0, rawMinimum - padding);
  const range = Math.max(maximum - minimum, Number.EPSILON);
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
  const hoveredRow = hoveredIndex === null ? null : rows[hoveredIndex];
  const hoveredX = hoveredIndex === null ? null : (rows.length === 1 ? 50 : (hoveredIndex / (rows.length - 1)) * 100);
  return (
    <div className="epoch-chart-shell">
      <div className="epoch-chart-legend">{series.map((item) => <span key={item.field}><i style={{ background: item.color }} />{item.label}</span>)}</div>
      <div className="epoch-chart-plot">
      <svg className="epoch-chart" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Training and validation loss by epoch" onMouseMove={(event) => setHoveredIndex(nearestPoint(event, rows.length))} onMouseLeave={() => setHoveredIndex(null)}>
        {[20, 56, 92].map((y) => <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="rgba(128, 155, 173, 0.18)" strokeWidth="1" />)}
        {hoveredX !== null && <line x1={hoveredX} y1="8" x2={hoveredX} y2="92" stroke="#50e3c2" strokeOpacity="0.65" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />}
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
      {hoveredRow && <div className="graph-tooltip epoch-tooltip" style={{ left: `${hoveredX}%` }}>
        <strong>Epoch {hoveredRow.epoch}</strong>
        {series.map((item) => Number.isFinite(Number(hoveredRow[item.field])) && <span key={item.field}><i style={{ background: item.color }} />{item.label}: {Number(hoveredRow[item.field]).toFixed(4)}</span>)}
      </div>}
      </div>
      <div className="epoch-chart-axis" aria-hidden="true">{rows.map((row) => <span key={row.epoch}>Epoch {row.epoch}</span>)}</div>
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
    <Card className="epoch-progress-card">
      <div className="epoch-progress-heading">
        <div><div className="monitor-metric-label">Epoch progress</div><strong>{completed} / {total ?? "—"} epochs</strong></div>
        <div className="epoch-progress-pending">{pending === null || pending === undefined ? "Pending unavailable" : `${pending} pending`}<span>{percent}%</span></div>
      </div>
      <Progress value={percent} className="epoch-progress-track" />
      <div className="epoch-progress-caption">Progress is based on completed validation epochs recorded in <code>training_metrics.json</code>.</div>
    </Card>
  );
}

function SummaryCard({ label, value, total, tone, progress }) {
  return (
    <Card className={`summary-card ${tone}`}>
      <div className="summary-card-label"><span className="summary-dot" />{label}</div>
      <div className="summary-card-value"><strong>{value}</strong><span>/ {total ?? "—"}</span></div>
      <Progress value={progress} className="summary-progress" />
    </Card>
  );
}

function RunTiming({ detail }) {
  const timing = detail.timing || {};
  const isRunning = detail.status === "running";
  return (
    <Card className="run-timing-card">
      <CardHeader className="run-timing-header">
        <div>
          <CardTitle>Run timing</CardTitle>
          <div className="wm-card-description">{isRunning ? "Live estimate based on completed epochs" : "Recorded execution window"}</div>
        </div>
        <Badge variant={isRunning ? "default" : "secondary"}>{isRunning ? "ETA" : "Completed"}</Badge>
      </CardHeader>
      <CardContent className="run-timing-grid">
        <div><span>Started</span><strong title={timing.started_at ? formatTime(timing.started_at) : "—"}>{timing.started_at ? formatCompactTime(timing.started_at) : "—"}</strong></div>
        <div><span>Ended</span><strong title={timing.ended_at ? formatTime(timing.ended_at) : "In progress"}>{timing.ended_at ? formatCompactTime(timing.ended_at) : "In progress"}</strong></div>
        <div><span>Elapsed</span><strong>{timing.elapsed_display || formatDuration(timing.elapsed_seconds)}</strong></div>
        <div className={isRunning ? "run-timing-estimate" : ""}><span>{isRunning ? "Estimated remaining" : "Estimated total"}</span><strong>{isRunning ? (timing.estimated_remaining_display || formatDuration(timing.estimated_remaining_seconds, "Calculating…")) : (timing.elapsed_display || formatDuration(timing.elapsed_seconds))}</strong></div>
      </CardContent>
      {isRunning && timing.estimated_total_display && <div className="run-timing-footnote">Estimated total duration: {timing.estimated_total_display} · based on {timing.estimate_source || "current progress"}</div>}
    </Card>
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

function MetricCard({ label, value, children, tone = "blue" }) {
  return (
    <Card className={`monitor-metric-card ${tone}`}>
      <div className="monitor-metric-card-header">
        <div className="monitor-metric-label">{label}</div>
        <div className="monitor-metric-value">{value}</div>
      </div>
      {children}
    </Card>
  );
}

function RunCard({ run, selected, onClick }) {
  const latest = run.latest || {};
  const training = run.training || {};
  const completed = Number(training.completed_epochs) || 0;
  const total = Number(training.total_epochs) || 0;
  const progress = Number(training.progress_percent) || (total ? (completed / total) * 100 : 0);
  const runId = String(run.id || "").split("/").at(-1)?.slice(0, 6) || "—";
  return (
    <button className={`monitor-run-card ${selected ? "selected" : ""}`} type="button" onClick={onClick}>
      <div className="monitor-run-card-topline">
        <span className="monitor-run-label">{run.fold}</span>
        <StatusBadge status={run.status} />
      </div>
      <div className="monitor-run-card-meta"><span>{runId}</span><span className="run-arrow">›</span></div>
      <div className="run-progress-line"><span style={{ width: `${Math.min(Math.max(progress, 0), 100)}%` }} /></div>
      <div className="monitor-run-card-epochs"><span>{completed} / {total || "—"} epochs</span><span>{Math.round(progress)}%</span></div>
      <div className="monitor-run-card-values">
        <span>RAM <strong>{formatBytes(latest.ram_rss_bytes)}</strong></span>
        <span>VRAM <strong>{formatBytes(latest.gpu_memory_bytes)}</strong></span>
      </div>
    </button>
  );
}

function groupStatus(runs) {
  if (runs.some((run) => run.status === "running")) return "running";
  if (runs.some((run) => run.status === "failed")) return "failed";
  if (runs.some((run) => run.status === "stale")) return "stale";
  return "completed";
}

function RunGroup({ group, expanded, selectedId, onToggle, onSelect }) {
  const status = groupStatus(group.runs);
  return (
    <section className="monitor-run-group">
      <button
        className={`monitor-run-group-heading ${expanded ? "expanded" : ""}`}
        type="button"
        aria-expanded={expanded}
        onClick={() => onToggle(group.key)}
      >
        <span className="monitor-run-group-chevron" aria-hidden="true">{expanded ? "⌄" : "›"}</span>
        <div><strong>{group.label}</strong></div>
        <StatusBadge status={status} />
      </button>
      {expanded && <div className="monitor-run-group-folds">{group.runs.map((run) => <RunCard key={run.id} run={run} selected={run.id === selectedId} onClick={() => onSelect(run.id)} />)}</div>}
    </section>
  );
}

export default function MonitorView() {
  const [runs, setRuns] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [showResourceDetail, setShowResourceDetail] = useState(true);
  const [showEpochDetail, setShowEpochDetail] = useState(true);
  const [expandedGroups, setExpandedGroups] = useState(() => new Set());
  const groupsInitialized = useRef(false);
  const groups = useMemo(() => {
    const grouped = new Map();
    runs.forEach((run) => {
      const key = run.group_id || run.run_name || run.id;
      const group = grouped.get(key) || {
        key,
        label: run.group_label || run.run_name || "Standalone run",
        startedAt: run.group_started_at || run.started_at,
        completedAt: run.group_completed_at,
        runs: [],
      };
      group.runs.push(run);
      group.startedAt = group.startedAt || run.started_at;
      group.completedAt = group.completedAt || run.group_completed_at;
      grouped.set(key, group);
    });
    return [...grouped.values()].sort((left, right) => new Date(right.startedAt || 0) - new Date(left.startedAt || 0));
  }, [runs]);

  useEffect(() => {
    if (!groups.length) return;
    setExpandedGroups((current) => {
      const available = new Set(groups.map((group) => group.key));
      return new Set([...current].filter((key) => available.has(key)));
    });
  }, [groups]);

  const toggleGroup = useCallback((groupKey) => {
    setExpandedGroups((current) => {
      const next = new Set(current);
      if (next.has(groupKey)) next.delete(groupKey);
      else next.add(groupKey);
      return next;
    });
  }, []);

  const refreshRuns = useCallback(async () => {
    const payload = await fetchMonitorRuns();
    const nextRuns = payload.runs || [];
    setRuns(nextRuns);
    if (!groupsInitialized.current && nextRuns.length) {
      const firstGroupKey = nextRuns[0].group_id || nextRuns[0].run_name || nextRuns[0].id;
      setExpandedGroups(new Set([firstGroupKey]));
      groupsInitialized.current = true;
    }
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
  const totalEpochs = runs.reduce((total, run) => total + (Number(run.training?.total_epochs) || 0), 0);
  const totalRuns = Math.max(runs.length, 1);
  const epochDenominator = Math.max(totalEpochs, epochsCompleted + epochsPending, 1);
  const samples = detail?.samples || [];
  const latest = detail?.latest || {};
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
        <div className="monitor-heading-copy">
          <h1>Training monitor</h1>
        </div>
        <div className="monitor-header-actions">
          <span className="monitor-refresh-state"><span className="monitor-live-pulse" /> Auto-refresh · 5s</span>
          <Button variant="outline" className="monitor-refresh-button" onClick={handleRefresh} disabled={refreshing}>{refreshing ? "Refreshing…" : "Refresh now"}</Button>
        </div>
      </header>

      <div className="monitor-summary-strip">
        <SummaryCard label="Active folds" value={activeCount} total={runs.length || "—"} progress={(activeCount / totalRuns) * 100} tone="blue" />
        <SummaryCard label="Completed folds" value={completedCount} total={runs.length || "—"} progress={(completedCount / totalRuns) * 100} tone="teal" />
        <SummaryCard label="Epochs completed" value={epochsCompleted} total={totalEpochs || "—"} progress={(epochsCompleted / epochDenominator) * 100} tone="teal" />
        <SummaryCard label="Epochs pending" value={epochsPending} total={totalEpochs || "—"} progress={(epochsPending / epochDenominator) * 100} tone="muted" />
      </div>

      {error && <div className="app-alert monitor-alert">{error}</div>}

      <div className="monitor-layout">
        <aside className="monitor-run-list">
          <div className="monitor-panel-heading"><span>Training runs</span><span>{groups.length} groups · {runs.length} folds</span></div>
          {groups.length ? groups.map((group) => <RunGroup key={group.key} group={group} expanded={expandedGroups.has(group.key)} selectedId={selectedId} onToggle={toggleGroup} onSelect={setSelectedId} />) : <div className="monitor-empty"><strong>No monitor logs found</strong><span>Start cross_train or train.py to populate this view.</span></div>}
        </aside>

        <section className="monitor-detail-panel">
          {detail ? (
            <>
              <header className="monitor-detail-header">
                <div>
                  <div className="monitor-detail-eyebrow">{detail.run_name} / {detail.fold}</div>
                  <h2>{detail.label}</h2>
                </div>
                <div className="monitor-detail-status">
                  <StatusBadge status={detail.status} />
                  <span>Duration {detail.timing?.elapsed_display || formatDuration(detail.timing?.elapsed_seconds)}</span>
                  <span>Started {formatRelativeTime(detail.started_at)}</span>
                  <span>Updated {formatTime(detail.updated_at)}</span>
                </div>
              </header>

              <div className="monitor-metrics-grid">
                <MetricCard label="RAM RSS" value={formatBytes(latest.ram_rss_bytes)} tone="blue">
                  <Sparkline samples={samples} field="ram_rss_bytes" color="#43b7e8" />
                </MetricCard>
                <MetricCard label="RAM USS" value={formatBytes(latest.ram_uss_bytes)} tone="violet">
                  <Sparkline samples={samples} field="ram_uss_bytes" color="#9b8cff" />
                </MetricCard>
                <MetricCard label="VRAM" value={formatBytes(latest.gpu_memory_bytes)} tone="orange">
                  <Sparkline samples={samples} field="gpu_memory_bytes" color="#f0a35b" />
                </MetricCard>
              </div>

              <RunTiming detail={detail} />

              <div className="monitor-drilldown-toggle"><span>Resource timeline</span><Button variant="outline" className="monitor-refresh-button" onClick={() => setShowResourceDetail((current) => !current)}>{showResourceDetail ? "Hide timeline" : "Open timeline"}</Button></div>
              {showResourceDetail && <ResourceDrilldown samples={samples} />}

              <EpochProgress training={detail.training} />
              <div className="monitor-drilldown-toggle"><span>Epoch timeline</span><Button variant="outline" className="monitor-refresh-button" onClick={() => setShowEpochDetail((current) => !current)}>{showEpochDetail ? "Hide timeline" : "Open timeline"}</Button></div>
              {showEpochDetail && <EpochDrilldown training={detail.training} />}

              <div className="monitor-detail-grid">
                <section className="monitor-data-panel">
                  <div className="monitor-panel-heading"><span>Latest sample</span><span>{detail.sample_count} total</span></div>
                  <dl className="monitor-data-list">
                    <div><dt>Backend</dt><dd>{String(detail.backend).toUpperCase()}</dd></div>
                    <div><dt>Root PID</dt><dd>{detail.root_pid || "—"}{detail.process_visible ? " · visible" : ""}</dd></div>
                    <div><dt>Tracked processes</dt><dd>{latest.tracked_pids?.length ?? "—"}</dd></div>
                    <div><dt>Interval</dt><dd>{detail.interval_seconds}s</dd></div>
                    <div><dt>Training started</dt><dd>{detail.started_at ? formatTime(detail.started_at) : "Timing unavailable"}</dd></div>
                    <div><dt>Training completed</dt><dd>{detail.completed_at ? formatTime(detail.completed_at) : detail.status === "completed" ? "Timing unavailable" : "Still running"}</dd></div>
                    <div><dt>Last sample</dt><dd>{formatTime(latest.timestamp_utc)}</dd></div>
                    <div><dt>Log file</dt><dd className="monitor-path" title={detail.resource_log}>{detail.resource_log}</dd></div>
                  </dl>
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
