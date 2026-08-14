import { useCallback, useEffect, useState } from "react";

import { deleteValidationRun, fetchValidationRun, fetchValidationRuns } from "../api.js";
import { EvaluationResultTabs } from "./EvaluationResultTables.jsx";
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Progress } from "./watermelon-ui.jsx";

const POLL_INTERVAL_MS = 3000;

function formatBytes(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "Unavailable";
  const number = Number(value);
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
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown" : date.toLocaleString([], { dateStyle: "medium", timeStyle: "medium" });
}

function formatClock(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDuration(seconds) {
  if (!Number.isFinite(Number(seconds))) return "—";
  const total = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes) return `${minutes}m ${String(remainder).padStart(2, "0")}s`;
  return `${remainder}s`;
}

function shortPath(value) {
  if (!value) return "—";
  const parts = String(value).split("/");
  return parts.length > 3 ? `…/${parts.slice(-3).join("/")}` : value;
}

function StatusBadge({ status }) {
  const tone = status === "completed" ? "completed" : status === "failed" ? "failed" : "running";
  return <Badge variant="outline" className={`monitor-status ${tone}`}><span className="monitor-status-dot" />{status}</Badge>;
}

function SummaryCard({ label, value, total, progress, tone = "blue" }) {
  return <Card className={`summary-card ${tone}`}><div className="summary-card-label"><span className="summary-dot" />{label}</div><div className="summary-card-value"><strong>{value}</strong><span>/ {total ?? "—"}</span></div><Progress value={progress} className="summary-progress" /></Card>;
}

const RESOURCE_SERIES = [
  { field: "ram_rss_bytes", color: "#43b7e8", label: "RAM RSS", format: formatBytes },
  { field: "ram_uss_bytes", color: "#9b8cff", label: "RAM USS", format: formatBytes },
  { field: "gpu_memory_bytes", color: "#f0a35b", label: "VRAM", format: formatBytes },
  { field: "cpu_utilization_percent", color: "#51d48a", label: "CPU tree", format: (value) => `${Number(value).toFixed(1)}%` },
  { field: "gpu_utilization_percent", color: "#f17c86", label: "GPU util", format: (value) => `${Number(value).toFixed(1)}%` },
];

function ResourceLane({ item, samples }) {
  const values = samples.map((sample) => Number(sample[item.field])).filter(Number.isFinite);
  if (!values.length) return <div className="resource-lane resource-lane-empty"><div className="resource-lane-header"><span className="resource-lane-label"><i style={{ background: item.color }} />{item.label}</span><span>Unavailable</span></div></div>;
  const maximum = Math.max(...values, 1);
  const points = samples.map((sample, index) => {
    const value = Number(sample[item.field]);
    if (!Number.isFinite(value)) return null;
    const x = samples.length === 1 ? 50 : (index / (samples.length - 1)) * 100;
    return `${x},${90 - (value / maximum) * 72}`;
  }).filter(Boolean).join(" ");
  const latest = [...samples].reverse().map((sample) => Number(sample[item.field])).find(Number.isFinite);
  return <div className="resource-lane"><div className="resource-lane-header"><span className="resource-lane-label"><i style={{ background: item.color }} />{item.label}</span><span>{item.format(latest)} · peak {item.format(maximum)}</span></div><div className="resource-lane-plot"><div className="resource-lane-axis"><span>{item.format(maximum)}</span><span>{item.format === formatBytes ? "0 B" : "0%"}</span></div><svg className="resource-lane-svg" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${item.label} over time`}><line x1="0" y1="90" x2="100" y2="90" stroke="rgba(128, 155, 173, 0.18)" strokeWidth="1" /><polyline points={points} fill="none" stroke={item.color} strokeWidth="2.4" vectorEffect="non-scaling-stroke" /></svg></div></div>;
}

function ResourceTimeline({ samples }) {
  const hasValues = samples.some((sample) => RESOURCE_SERIES.some(({ field }) => Number.isFinite(Number(sample[field]))));
  return <section className="resource-drilldown"><div className="monitor-panel-heading"><span>Validation resource timeline</span><span>{samples.length} samples</span></div>{hasValues ? <div className="resource-chart-wrap"><div className="resource-chart-shell">{RESOURCE_SERIES.map((item) => <ResourceLane key={item.field} item={item} samples={samples} />)}<div className="resource-chart-time-axis"><span>{formatClock(samples[0]?.timestamp_utc)}</span><span>{formatClock(samples.at(-1)?.timestamp_utc)}</span></div></div></div> : <div className="monitor-chart-empty resource-chart-empty">Waiting for validation resource samples.</div>}</section>;
}

function ResourceMetric({ label, value, peak, tone }) {
  return <article className={`monitor-metric-card ${tone}`}><div className="monitor-metric-card-header"><span className="monitor-metric-label">{label}</span><div className="monitor-metric-values"><strong className="monitor-metric-value">{value}</strong><span className="monitor-metric-peak">peak {peak}</span></div></div></article>;
}

function ValidationRunCard({ run, selected, onSelect, onDelete, deleting }) {
  const progress = run.status === "completed" ? 100 : run.status === "running" ? 50 : 0;
  const isActive = ["queued", "running"].includes(run.status);
  return <div className={`monitor-run-card-shell ${selected ? "selected" : ""}`}><button type="button" className="monitor-run-card" onClick={() => onSelect(run.id)}><div className="monitor-run-card-topline"><span className="monitor-run-label">{run.label}</span><StatusBadge status={run.status} /></div><div className="monitor-run-card-meta"><span>{run.backend} · {run.sample_count || 0} samples</span><span className="run-arrow">›</span></div><div className="run-progress-line"><span style={{ width: `${progress}%` }} /></div><div className="monitor-run-card-epochs"><span>{run.request?.checkpoints?.length || 0} checkpoints</span><span>{run.timing?.elapsed_display || formatDuration(run.timing?.elapsed_seconds)}</span></div><div className="monitor-run-card-values"><span>RAM <strong>{formatBytes(run.latest?.ram_rss_bytes)}</strong></span><span>VRAM <strong>{formatBytes(run.latest?.gpu_memory_bytes)}</strong></span></div></button><button type="button" className="monitor-run-delete-button" onClick={() => onDelete(run)} disabled={isActive || deleting} title={isActive ? "Running validations cannot be deleted" : "Delete monitor telemetry"} aria-label={`Delete ${run.label}`}>{deleting ? "Deleting…" : "Delete"}</button></div>;
}

export default function ValidationMonitorView() {
  const [runs, setRuns] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState(null);
  const [deletingId, setDeletingId] = useState("");
  const [error, setError] = useState("");

  const refreshRuns = useCallback(async () => {
    const payload = await fetchValidationRuns();
    const nextRuns = payload.runs || [];
    setRuns(nextRuns);
    setSelectedId((current) => current && nextRuns.some((run) => run.id === current) ? current : nextRuns[0]?.id || "");
  }, []);

  useEffect(() => {
    const poll = async () => {
      try { await refreshRuns(); setError(""); } catch (pollError) { setError(pollError.message); }
    };
    poll();
    const timer = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [refreshRuns]);

  useEffect(() => {
    if (!selectedId) { setDetail(null); return undefined; }
    let cancelled = false;
    const poll = async () => {
      try { const nextDetail = await fetchValidationRun(selectedId); if (!cancelled) { setDetail(nextDetail); setError(""); } } catch (pollError) { if (!cancelled) setError(pollError.message); }
    };
    poll();
    const timer = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [selectedId]);

  const activeCount = runs.filter((run) => ["queued", "running"].includes(run.status)).length;
  const completedCount = runs.filter((run) => run.status === "completed").length;
  const sampleCount = runs.reduce((total, run) => total + (run.sample_count || 0), 0);
  const samples = detail?.samples || [];
  const request = detail?.request || {};
  const peak = detail?.peak || {};
  const latest = detail?.latest || {};
  const evaluationSummary = detail?.evaluation_summary || {};
  const handleRefresh = async () => { try { await refreshRuns(); if (selectedId) setDetail(await fetchValidationRun(selectedId)); setError(""); } catch (refreshError) { setError(refreshError.message); } };
  const handleDelete = async (run) => {
    if (["queued", "running"].includes(run.status) || deletingId) return;
    const confirmed = window.confirm(`Delete validation telemetry for "${run.label}"? This removes the monitor trigger and resource logs, but keeps evaluation outputs and checkpoints.`);
    if (!confirmed) return;
    setDeletingId(run.id);
    setError("");
    try {
      await deleteValidationRun(run.id);
      setRuns((current) => current.filter((item) => item.id !== run.id));
      if (selectedId === run.id) {
        setSelectedId("");
        setDetail(null);
      }
    } catch (deleteError) {
      setError(deleteError.message);
    } finally {
      setDeletingId("");
    }
  };

  return <section className="monitor-page validation-page" aria-label="Validation monitor"><header className="monitor-header"><div className="monitor-heading-copy"><div><div className="evaluation-eyebrow">HFF-Net / process telemetry</div><h1>Validation monitor</h1></div></div><div className="monitor-header-actions"><span className="monitor-refresh-state"><span className="monitor-live-pulse" /> Auto-refresh · 3s</span><Button type="button" variant="outline" className="monitor-refresh-button" onClick={handleRefresh}>Refresh now</Button></div></header><p className="validation-intro">Process-scoped RAM, VRAM, CPU, and GPU utilization for web-triggered checkpoint evaluation jobs. Delete removes monitor telemetry only; evaluation outputs remain.</p>{error && <div className="app-alert monitor-alert">{error}</div>}<div className="monitor-summary-strip"><SummaryCard label="Active validations" value={activeCount} total={runs.length || "—"} progress={(activeCount / Math.max(runs.length, 1)) * 100} tone="blue" /><SummaryCard label="Completed validations" value={completedCount} total={runs.length || "—"} progress={(completedCount / Math.max(runs.length, 1)) * 100} tone="teal" /><SummaryCard label="Telemetry samples" value={sampleCount} total={sampleCount || "—"} progress={sampleCount ? 100 : 0} tone="teal" /><SummaryCard label="GPU tracked" value={runs.filter((run) => ["cuda", "mps", "nvidia"].includes(run.backend)).length} total={runs.length || "—"} progress={(runs.filter((run) => ["cuda", "mps", "nvidia"].includes(run.backend)).length / Math.max(runs.length, 1)) * 100} tone="muted" /></div><div className="monitor-layout"><aside className="monitor-run-list"><div className="monitor-panel-heading"><span>Validation runs</span><span>{runs.length} jobs</span></div>{runs.length ? runs.map((run) => <ValidationRunCard key={run.id} run={run} selected={run.id === selectedId} onSelect={setSelectedId} onDelete={handleDelete} deleting={deletingId === run.id} />) : <div className="monitor-empty"><strong>No validation telemetry found</strong><span>Start a web evaluation job to populate this view.</span></div>}</aside><section className="monitor-detail-panel">{detail ? <><header className="monitor-detail-header"><div><div className="monitor-detail-eyebrow">{detail.backend} / {detail.id}</div><h2>{detail.label}</h2></div><div className="monitor-detail-status"><StatusBadge status={detail.status} /><span>Duration {detail.timing?.elapsed_display || formatDuration(detail.timing?.elapsed_seconds)}</span><span>Updated {formatTime(detail.updated_at)}</span></div></header><div className="monitor-metrics-grid"><ResourceMetric label="RAM RSS" value={formatBytes(latest.ram_rss_bytes)} peak={formatBytes(peak.ram_rss_bytes)} tone="blue" /><ResourceMetric label="RAM USS" value={formatBytes(latest.ram_uss_bytes)} peak={formatBytes(peak.ram_uss_bytes)} tone="violet" /><ResourceMetric label="VRAM" value={formatBytes(latest.gpu_memory_bytes)} peak={formatBytes(peak.gpu_memory_bytes)} tone="orange" /><ResourceMetric label="CPU tree" value={latest.cpu_utilization_percent == null ? "Unavailable" : `${Number(latest.cpu_utilization_percent).toFixed(1)}%`} peak={peak.cpu_utilization_percent == null ? "Unavailable" : `${Number(peak.cpu_utilization_percent).toFixed(1)}%`} tone="blue" /><ResourceMetric label="GPU util" value={latest.gpu_utilization_percent == null ? "Unavailable" : `${Number(latest.gpu_utilization_percent).toFixed(1)}%`} peak={latest.gpu_utilization_percent == null ? "Unavailable" : `${Number(peak.gpu_utilization_percent ?? latest.gpu_utilization_percent).toFixed(1)}%`} tone="orange" /></div><ResourceTimeline samples={samples} />{evaluationSummary.results?.length > 0 && <EvaluationResultTabs summary={evaluationSummary} className="validation-evaluation-results" />}<div className="monitor-detail-grid"><section className="monitor-data-panel"><div className="monitor-panel-heading"><span>Validation configuration</span><span>{detail.sample_count} samples</span></div><dl className="monitor-data-list"><div><dt>Dataset / region</dt><dd>{request.dataset_name || "—"} / {request.class_type || "—"}</dd></div><div><dt>Checkpoints</dt><dd>{request.checkpoints?.length || 0}</dd></div><div><dt>Test list</dt><dd className="monitor-path" title={request.test_list}>{shortPath(request.test_list)}</dd></div><div><dt>Output directory</dt><dd className="monitor-path" title={request.output_dir}>{shortPath(request.output_dir)}</dd></div><div><dt>Backend</dt><dd>{String(detail.backend).toUpperCase()}</dd></div><div><dt>Monitor interval</dt><dd>{detail.interval_seconds}s</dd></div><div><dt>Root PID</dt><dd>{detail.root_pid || "—"}{detail.process_visible ? " · visible" : ""}</dd></div><div><dt>Resource log</dt><dd className="monitor-path" title={detail.resource_log}>{shortPath(detail.resource_log)}</dd></div></dl></section></div>{detail.resource_monitor_error && <div className="app-alert monitor-alert">Resource monitor warning: {detail.resource_monitor_error}</div>}</> : <div className="monitor-empty monitor-detail-empty"><strong>Select a validation run</strong><span>Choose an evaluation job to inspect process-scoped resource history.</span></div>}</section></div></section>;
}
