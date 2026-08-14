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

function evaluationSampleProgress(run) {
  const progress = run?.progress || {};
  const processedValue = progress.overall_processed_samples ?? progress.processed_samples;
  const totalValue = progress.overall_total_samples ?? progress.total_samples;
  const processed = Number.isFinite(Number(processedValue)) ? Math.max(0, Math.round(Number(processedValue))) : null;
  const total = Number.isFinite(Number(totalValue)) ? Math.max(0, Math.round(Number(totalValue))) : null;
  const percent = total > 0 && processed !== null ? Math.min(100, (processed / total) * 100) : null;
  return {
    processed,
    total,
    percent,
    label: processed !== null && total !== null ? `${processed} / ${total} samples` : "Waiting for inference progress",
  };
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

function nearestPoint(event, count) {
  const bounds = event.currentTarget.getBoundingClientRect();
  const ratio = Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0), 1);
  return Math.round(ratio * Math.max(count - 1, 0));
}

function ResourceLane({ item, samples }) {
  const values = samples.map((sample) => Number(sample[item.field])).filter(Number.isFinite);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  if (!values.length) return <div className="resource-lane resource-lane-empty"><div className="resource-lane-header"><span className="resource-lane-label"><i style={{ background: item.color }} />{item.label}</span><span>Unavailable</span></div></div>;
  const maximum = Math.max(...values, 1);
  const pointFor = (sample, index) => {
    const value = Number(sample[item.field]);
    if (!Number.isFinite(value)) return null;
    const x = samples.length === 1 ? 50 : (index / (samples.length - 1)) * 100;
    return { x, y: 90 - (value / maximum) * 72 };
  };
  const points = samples.map((sample, index) => pointFor(sample, index)).filter(Boolean).map(({ x, y }) => `${x},${y}`).join(" ");
  const latest = [...samples].reverse().map((sample) => Number(sample[item.field])).find(Number.isFinite);
  const lastIndex = samples.map((sample) => Number(sample[item.field])).reduce((last, value, index) => Number.isFinite(value) ? index : last, -1);
  const lastPoint = lastIndex >= 0 ? pointFor(samples[lastIndex], lastIndex) : null;
  const hoveredSample = hoveredIndex === null ? null : samples[hoveredIndex];
  const hoveredValue = hoveredSample ? Number(hoveredSample[item.field]) : null;
  const hoveredPoint = hoveredSample ? pointFor(hoveredSample, hoveredIndex) : null;
  return <div className="resource-lane"><div className="resource-lane-header"><span className="resource-lane-label"><i style={{ background: item.color }} />{item.label}</span><span>{item.format(latest)} · peak {item.format(maximum)}</span></div><div className="resource-lane-plot"><div className="resource-lane-axis"><span>{item.format(maximum)}</span><span>{item.format === formatBytes ? "0 B" : "0%"}</span></div><svg className="resource-lane-svg" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${item.label} over time`} onMouseMove={(event) => setHoveredIndex(nearestPoint(event, samples.length))} onMouseLeave={() => setHoveredIndex(null)}><line x1="0" y1="90" x2="100" y2="90" stroke="rgba(128, 155, 173, 0.18)" strokeWidth="1" /><polyline points={points} fill="none" stroke={item.color} strokeWidth="2.4" vectorEffect="non-scaling-stroke" />{lastPoint && <circle cx={lastPoint.x} cy={lastPoint.y} r="2.6" fill={item.color} vectorEffect="non-scaling-stroke" />}{hoveredPoint && <line x1={hoveredPoint.x} y1="10" x2={hoveredPoint.x} y2="92" stroke={item.color} strokeOpacity="0.55" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />}</svg>{hoveredSample && Number.isFinite(hoveredValue) && hoveredPoint && <div className="graph-tooltip resource-tooltip" style={{ left: `${hoveredPoint.x}%` }}><strong>{item.format(hoveredValue)}</strong><span>{formatClock(hoveredSample.timestamp_utc)}</span></div>}</div></div>;
}

function ResourceTimeline({ samples }) {
  const hasValues = samples.some((sample) => RESOURCE_SERIES.some(({ field }) => Number.isFinite(Number(sample[field]))));
  return <section className="resource-drilldown"><div className="monitor-panel-heading"><span>Validation resource timeline</span><span>{samples.length} samples</span></div>{hasValues ? <div className="resource-chart-wrap"><div className="resource-chart-shell">{RESOURCE_SERIES.map((item) => <ResourceLane key={item.field} item={item} samples={samples} />)}<div className="resource-chart-time-axis"><span>{formatClock(samples[0]?.timestamp_utc)}</span><span>{formatClock(samples.at(-1)?.timestamp_utc)}</span></div></div></div> : <div className="monitor-chart-empty resource-chart-empty">Waiting for validation resource samples.</div>}</section>;
}

function ResourceMetric({ label, value, peak, tone }) {
  const displayPeak = typeof peak === "number" ? formatBytes(peak) : peak;
  return <article className={`monitor-metric-card ${tone}`}><div className="monitor-metric-card-header"><span className="monitor-metric-label">{label}</span><div className="monitor-metric-values"><strong className="monitor-metric-value">{value}</strong><span className="monitor-metric-peak">peak {displayPeak}</span></div></div></article>;
}

function ValidationRunCard({ run, selected, onSelect, onDelete, deleting }) {
  const sampleProgress = evaluationSampleProgress(run);
  const progress = sampleProgress.percent ?? (run.status === "completed" ? 100 : run.status === "running" ? 50 : 0);
  const isActive = ["queued", "running"].includes(run.status);
  return <div className={`monitor-run-card-shell ${selected ? "selected" : ""}`}><button type="button" className="monitor-run-card" onClick={() => onSelect(run.id)}><div className="monitor-run-card-topline"><span className="monitor-run-label">{run.label}</span><StatusBadge status={run.status} /></div><div className="monitor-run-card-meta"><span>{run.backend} · {run.sample_count || 0} telemetry samples</span><span className="run-arrow">›</span></div><div className="monitor-run-log-path" title={run.resource_log}>Log: {shortPath(run.resource_log)}</div><div className="run-progress-line" aria-label={`Inference progress: ${sampleProgress.label}`}><span style={{ width: `${progress}%` }} /></div><div className="monitor-run-card-epochs"><span>{sampleProgress.label}</span><span>{run.request?.checkpoints?.length || 0} checkpoints · {run.timing?.elapsed_display || formatDuration(run.timing?.elapsed_seconds)}</span></div><div className="monitor-run-card-values"><span>RAM <strong>{formatBytes(run.latest?.ram_rss_bytes)}</strong></span><span>VRAM <strong>{formatBytes(run.latest?.gpu_memory_bytes)}</strong></span></div></button><button type="button" className="monitor-run-delete-button" onClick={() => onDelete(run)} disabled={isActive || deleting} title={isActive ? "Running validations cannot be deleted" : "Delete monitor telemetry"} aria-label={`Delete ${run.label}`}>{deleting ? "Deleting…" : "Delete"}</button></div>;
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
  const evaluationProgress = evaluationSampleProgress(detail);
  const detailProgressPercent = evaluationProgress.percent ?? (detail?.status === "completed" ? 100 : detail?.status === "running" ? 50 : 0);
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

 return <section className="monitor-page validation-page" aria-label="Validation monitor"><header className="monitor-header"><div className="monitor-heading-copy"><div><div className="evaluation-eyebrow">HFF-Net / process telemetry</div><h1>Validation monitor</h1></div></div><div className="monitor-header-actions"><span className="monitor-refresh-state"><span className="monitor-live-pulse" /> Auto-refresh · 3s</span><Button type="button" variant="outline" className="monitor-refresh-button" onClick={handleRefresh}>Refresh now</Button></div></header><p className="validation-intro">Process-scoped RAM, VRAM, CPU, and GPU utilization for web-triggered or PBS-submitted checkpoint evaluation jobs. Delete removes monitor telemetry only; evaluation outputs remain.</p>{error && <div className="app-alert monitor-alert">{error}</div>}<div className="monitor-summary-strip"><SummaryCard label="Active validations" value={activeCount} total={runs.length || "—"} progress={(activeCount / Math.max(runs.length, 1)) * 100} tone="blue" /><SummaryCard label="Completed validations" value={completedCount} total={runs.length || "—"} progress={(completedCount / Math.max(runs.length, 1)) * 100} tone="teal" /><SummaryCard label="Telemetry samples" value={sampleCount} total={sampleCount || "—"} progress={sampleCount ? 100 : 0} tone="teal" /><SummaryCard label="GPU tracked" value={runs.filter((run) => ["cuda", "mps", "nvidia"].includes(run.backend)).length} total={runs.length || "—"} progress={(runs.filter((run) => ["cuda", "mps", "nvidia"].includes(run.backend)).length / Math.max(runs.length, 1)) * 100} tone="muted" /></div><div className="monitor-layout"><aside className="monitor-run-list"><div className="monitor-panel-heading"><span>Validation runs</span><span>{runs.length} jobs</span></div>{runs.length ? runs.map((run) => <ValidationRunCard key={run.id} run={run} selected={run.id === selectedId} onSelect={setSelectedId} onDelete={handleDelete} deleting={deletingId === run.id} />) : <div className="monitor-empty"><strong>No validation telemetry found</strong><span>Start an evaluation from the viewer or submit the generated PBS command to populate this view.</span></div>}</aside><section className="monitor-detail-panel">{detail ? <><header className="monitor-detail-header"><div><div className="monitor-detail-eyebrow">{detail.backend} / {detail.id}</div><h2>{detail.label}</h2></div><div className="monitor-detail-status"><StatusBadge status={detail.status} /><span>Duration {detail.timing?.elapsed_display || formatDuration(detail.timing?.elapsed_seconds)}</span><span>Updated {formatTime(detail.updated_at)}</span></div></header><section className="evaluation-sample-progress" aria-label="Inference sample progress"><div className="evaluation-sample-progress-header"><strong>Inference progress</strong><strong>{evaluationProgress.label}</strong></div><div className="evaluation-sample-progress-track"><span style={{ width: `${detailProgressPercent}%` }} /></div><div className="evaluation-sample-progress-meta"><span>{detail.progress?.phase || "Waiting for progress updates"}</span><span>{evaluationProgress.percent === null ? "" : `${evaluationProgress.percent.toFixed(0)}%`}</span></div></section><div className="monitor-metrics-grid"><ResourceMetric label="RAM RSS" value={formatBytes(latest.ram_rss_bytes)} peak={formatBytes(peak.ram_rss_bytes)} tone="blue" /><ResourceMetric label="RAM USS" value={formatBytes(latest.ram_uss_bytes)} peak={peak.ram_uss_bytes} tone="violet" /><ResourceMetric label="VRAM" value={formatBytes(latest.gpu_memory_bytes)} peak={formatBytes(peak.gpu_memory_bytes)} tone="orange" /><ResourceMetric label="CPU tree" value={latest.cpu_utilization_percent == null ? "Unavailable" : `${Number(latest.cpu_utilization_percent).toFixed(1)}%`} peak={peak.cpu_utilization_percent == null ? "Unavailable" : `${Number(peak.cpu_utilization_percent).toFixed(1)}%`} tone="blue" /><ResourceMetric label="GPU util" value={latest.gpu_utilization_percent == null ? "Unavailable" : `${Number(latest.gpu_utilization_percent).toFixed(1)}%`} peak={latest.gpu_utilization_percent == null ? "Unavailable" : `${Number(peak.gpu_utilization_percent ?? latest.gpu_utilization_percent).toFixed(1)}%`} tone="orange" /></div><ResourceTimeline samples={samples} />{(evaluationSummary.results?.length > 0 || request.checkpoints?.length > 0) && <EvaluationResultTabs summary={evaluationSummary} checkpointPaths={request.checkpoints} className="validation-evaluation-results" />}<div className="monitor-detail-grid"><section className="monitor-data-panel"><div className="monitor-panel-heading"><span>Validation configuration</span><span>{detail.sample_count} samples</span></div><dl className="monitor-data-list"><div><dt>Evaluation name</dt><dd>{request.name || detail.label || "—"}</dd></div><div><dt>Training run</dt><dd>{request.training_run || "—"}</dd></div><div><dt>Fold</dt><dd>{request.fold || "All folds"}</dd></div><div><dt>Dataset / labels</dt><dd>{request.dataset_name || "—"} / {request.class_type || "—"}</dd></div><div><dt>Checkpoints</dt><dd>{request.checkpoints?.length || 0}</dd></div><div><dt>Checkpoint paths</dt><dd className="monitor-config-paths">{(request.checkpoints || []).map((checkpoint) => <code key={checkpoint} title={checkpoint}>{shortPath(checkpoint)}</code>)}</dd></div><div><dt>Test list</dt><dd className="monitor-path" title={request.test_list}>{shortPath(request.test_list)}</dd></div><div><dt>Output directory</dt><dd className="monitor-path" title={request.output_dir}>{shortPath(request.output_dir)}</dd></div><div><dt>Batch size</dt><dd>{request.batch_size ?? "—"}</dd></div><div><dt>Workers</dt><dd>{request.num_workers ?? "—"}</dd></div><div><dt>Backend</dt><dd>{String(detail.backend).toUpperCase()}</dd></div><div><dt>Monitor interval</dt><dd>{detail.interval_seconds}s</dd></div><div><dt>Root PID</dt><dd>{detail.root_pid || "—"}{detail.process_visible ? " · visible" : ""}</dd></div><div><dt>Resource log</dt><dd className="monitor-path" title={detail.resource_log}>{shortPath(detail.resource_log)}</dd></div></dl></section></div>{detail.resource_monitor_error && <div className="app-alert monitor-alert">Resource monitor warning: {detail.resource_monitor_error}</div>}</> : <div className="monitor-empty monitor-detail-empty"><strong>Select a validation run</strong><span>Choose an evaluation job to inspect process-scoped resource history.</span></div>}</section></div></section>;
}
