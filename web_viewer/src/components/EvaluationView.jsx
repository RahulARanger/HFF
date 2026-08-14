import { useCallback, useEffect, useMemo, useState } from "react";

import {
  fetchEvaluationJobs,
  fetchEvaluationOptions,
  startEvaluation,
} from "../api.js";
import { Badge, Button, Card, CardContent, CardHeader, CardTitle } from "./watermelon-ui.jsx";

const POLL_INTERVAL_MS = 3000;

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown" : date.toLocaleString([], { dateStyle: "medium", timeStyle: "medium" });
}

function shortPath(value) {
  if (!value) return "—";
  const parts = String(value).split("/");
  return parts.length > 3 ? `…/${parts.slice(-3).join("/")}` : value;
}

function StatusBadge({ status }) {
  const tone = status === "completed" ? "completed" : status === "failed" ? "failed" : "running";
  return <Badge variant="outline" className={`monitor-status ${tone}`}><span className="monitor-status-dot" />{status || "queued"}</Badge>;
}

const STRUCTURE_ROWS = [
  { label: "ET", metricIndex: 0 },
  { label: "TC", metricIndex: 1 },
  { label: "WT", metricIndex: 2 },
];

function formatMetric(value) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—"
    : Number(value).toFixed(4);
}

function metricRows(metrics) {
  const maxLength = Math.max(
    metrics?.branch_1?.length || 0,
    metrics?.branch_2?.length || 0,
  );
  if (maxLength >= 6) return STRUCTURE_ROWS;
  if (maxLength >= 2) return [{ label: "Tumour", metricIndex: 0 }];
  return [];
}

function branchLabel(branch) {
  return branch === "branch_1" ? "LF" : "HF";
}

function MetricResultsTable({ metrics }) {
  const rows = metricRows(metrics);
  return (
    <div className="evaluation-results-table-wrap">
      <table className="evaluation-results-table">
        <caption>Segmentation metrics · Dice and HD95</caption>
        <thead>
          <tr>
            <th scope="col">Region</th>
            <th scope="col">{branchLabel("branch_1")} Dice</th>
            <th scope="col">{branchLabel("branch_1")} HD95</th>
            <th scope="col">{branchLabel("branch_2")} Dice</th>
            <th scope="col">{branchLabel("branch_2")} HD95</th>
          </tr>
        </thead>
        <tbody>
          {rows.length ? rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">{row.label}</th>
              <td>{formatMetric(metrics?.branch_1?.[row.metricIndex * 2])}</td>
              <td>{formatMetric(metrics?.branch_1?.[row.metricIndex * 2 + 1])}</td>
              <td>{formatMetric(metrics?.branch_2?.[row.metricIndex * 2])}</td>
              <td>{formatMetric(metrics?.branch_2?.[row.metricIndex * 2 + 1])}</td>
            </tr>
          )) : (
            <tr><td colSpan="5">No metric values were written for this result.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function LossResultsTable({ losses }) {
  return (
    <div className="evaluation-loss-table-wrap">
      <table className="evaluation-loss-table">
        <caption>Validation losses retained by eval.py</caption>
        <thead><tr><th scope="col">Branch</th><th scope="col">Loss</th></tr></thead>
        <tbody>
          <tr><th scope="row">LF</th><td>{formatMetric(losses?.validation_loss_branch_1)}</td></tr>
          <tr><th scope="row">HF</th><td>{formatMetric(losses?.validation_loss_branch_2)}</td></tr>
        </tbody>
      </table>
    </div>
  );
}

function ResultTabs({ summary }) {
  const [activeTab, setActiveTab] = useState("average");
  const results = summary.results || [];
  const tabs = [
    { id: "average", label: "Average across checkpoints", metrics: summary.average_metrics, losses: summary.average_validation_losses },
    ...results.map((result, index) => ({
      id: `checkpoint-${index}`,
      label: `Checkpoint ${index + 1}`,
      checkpoint: result.checkpoint,
      metrics: result.metrics,
      losses: {
        validation_loss_branch_1: result.validation_loss_branch_1,
        validation_loss_branch_2: result.validation_loss_branch_2,
      },
    })),
  ];
  const selected = tabs.find((tab) => tab.id === activeTab) || tabs[0];

  return (
    <section className="evaluation-results" aria-label="Evaluation results">
      <div className="evaluation-results-heading">
        <div><h3>Results by checkpoint</h3><span>{results.length} checkpoint result{results.length === 1 ? "" : "s"}</span></div>
        <span className="evaluation-results-note">Dice ↑ · HD95 ↓</span>
      </div>
      <div className="evaluation-result-tabs" role="tablist" aria-label="Checkpoint evaluation results">
        {tabs.map((tab) => (
          <button
            type="button"
            role="tab"
            aria-selected={tab.id === selected.id}
            className={`evaluation-result-tab ${tab.id === selected.id ? "active" : ""}`}
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            title={tab.checkpoint || tab.label}
          >
            <strong>{tab.label}</strong>
            {tab.checkpoint && <small>{shortPath(tab.checkpoint)}</small>}
          </button>
        ))}
      </div>
      <div className="evaluation-result-panel" role="tabpanel">
        <div className="evaluation-result-panel-heading">
          <strong>{selected.label}</strong>
          {selected.checkpoint && <code title={selected.checkpoint}>{selected.checkpoint}</code>}
        </div>
        <MetricResultsTable metrics={selected.metrics} />
        <LossResultsTable losses={selected.losses} />
      </div>
    </section>
  );
}

function CheckpointPicker({ checkpoints, selected, onChange, disabled }) {
  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const toggle = (path) => {
    const next = new Set(selectedSet);
    if (next.has(path)) next.delete(path);
    else next.add(path);
    onChange([...next]);
  };

  return (
    <div className="evaluation-checkpoint-list">
      {checkpoints.length ? checkpoints.map((checkpoint) => (
        <label className={`evaluation-checkpoint ${selectedSet.has(checkpoint.path) ? "selected" : ""}`} key={checkpoint.path}>
          <input type="checkbox" checked={selectedSet.has(checkpoint.path)} onChange={() => toggle(checkpoint.path)} disabled={disabled} />
          <span><strong>{checkpoint.label}</strong><small>{checkpoint.path}</small></span>
        </label>
      )) : <div className="evaluation-empty">No .pth checkpoints were found below the configured results directory.</div>}
    </div>
  );
}

function JobCard({ job }) {
  const request = job.request || {};
  const summary = job.summary || {};
  return (
    <Card className="evaluation-job-card">
      <CardHeader>
        <div>
          <CardTitle>Evaluation job <code>{job.id}</code></CardTitle>
          <div className="wm-card-description">{formatTime(job.created_at)} · {request.checkpoints?.length || 0} checkpoint(s)</div>
        </div>
        <StatusBadge status={job.status} />
      </CardHeader>
      <CardContent>
        <div className="evaluation-job-meta">
          <span><label>Dataset</label><strong>{request.dataset_name || "—"}</strong></span>
          <span><label>Test list</label><strong title={request.test_list}>{shortPath(request.test_list)}</strong></span>
          <span><label>Output</label><strong title={request.output_dir}>{shortPath(request.output_dir)}</strong></span>
        </div>
        {job.error && <div className="app-alert evaluation-alert">{job.error}</div>}
        {job.status === "completed" && <ResultTabs summary={summary} />}
        {job.status !== "completed" && job.log_tail && <pre className="evaluation-log">{job.log_tail}</pre>}
        {job.status === "completed" && <div className="evaluation-job-files"><span>Summary</span><code title={job.summary_file}>{shortPath(job.summary_file)}</code></div>}
      </CardContent>
    </Card>
  );
}

export default function EvaluationView() {
  const [options, setOptions] = useState({ checkpoints: [], test_lists: [], defaults: {} });
  const [selectedCheckpoints, setSelectedCheckpoints] = useState([]);
  const [testList, setTestList] = useState("");
  const [datasetName, setDatasetName] = useState("brats19");
  const [outputDir, setOutputDir] = useState("");
  const [monitorInterval, setMonitorInterval] = useState("5");
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const loadOptions = useCallback(async () => {
    const payload = await fetchEvaluationOptions();
    setOptions(payload);
    setTestList((current) => current || payload.defaults?.test_list || payload.test_lists?.[0]?.path || "");
    setOutputDir((current) => current || payload.defaults?.output_dir || "");
    setSelectedCheckpoints((current) => current.filter((path) => payload.checkpoints?.some((checkpoint) => checkpoint.path === path)));
  }, []);

  const loadJobs = useCallback(async () => {
    const payload = await fetchEvaluationJobs();
    const nextJobs = payload.jobs || [];
    setJobs(nextJobs);
    return nextJobs;
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([loadOptions(), loadJobs()]).then(() => {
      if (!cancelled) {
        setLoading(false);
        setError("");
      }
    }).catch((loadError) => {
      if (!cancelled) {
        setLoading(false);
        setError(loadError.message);
      }
    });
    return () => { cancelled = true; };
  }, [loadJobs, loadOptions]);

  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        await loadJobs();
      } catch (pollError) {
        setError(pollError.message);
      }
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [loadJobs]);

  const activeJob = jobs.find((job) => ["queued", "running"].includes(job.status));
  const canSubmit = selectedCheckpoints.length > 0 && testList.trim() && outputDir.trim() && !activeJob && !submitting;

  const handleRefresh = async () => {
    try {
      await Promise.all([loadOptions(), loadJobs()]);
      setError("");
    } catch (refreshError) {
      setError(refreshError.message);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError("");
    try {
      const job = await startEvaluation({
        checkpoints: selectedCheckpoints,
        test_list: testList.trim(),
        output_dir: outputDir.trim(),
        dataset_name: datasetName,
        class_type: "all",
        resource_monitor_interval: Number(monitorInterval),
      });
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="evaluation-page" aria-label="Model evaluation">
      <header className="evaluation-header">
        <div><div className="evaluation-eyebrow">HFF-Net / inference control</div><h1>Run evaluation</h1><p>Evaluate selected checkpoints on the same test manifest and keep the per-checkpoint results plus the aggregate average.</p></div>
        <div className="evaluation-header-actions"><span className="monitor-refresh-state"><span className="monitor-live-pulse" /> Job status · 3s</span><Button type="button" variant="outline" className="monitor-refresh-button" onClick={handleRefresh}>Refresh</Button></div>
      </header>

      {error && <div className="app-alert evaluation-alert">{error}</div>}

      <div className="evaluation-layout">
        <form className="evaluation-form" onSubmit={handleSubmit}>
          <Card className="evaluation-form-card">
            <CardHeader><div><CardTitle>Evaluation setup</CardTitle><div className="wm-card-description">The backend launches <code>cross_eval.py</code> with these exact selections.</div></div></CardHeader>
            <CardContent>
              <div className="evaluation-field"><label>Checkpoints <span>{selectedCheckpoints.length} selected</span></label><CheckpointPicker checkpoints={options.checkpoints || []} selected={selectedCheckpoints} onChange={setSelectedCheckpoints} disabled={Boolean(activeJob) || submitting} /></div>
              <div className="evaluation-field"><label htmlFor="test-list">Test list</label><select id="test-list" value={testList} onChange={(event) => setTestList(event.target.value)} disabled={Boolean(activeJob) || submitting}><option value="">Select a discovered test list</option>{(options.test_lists || []).map((item) => <option key={item.path} value={item.path}>{item.label}</option>)}</select><input aria-label="Custom test list path" value={testList} onChange={(event) => setTestList(event.target.value)} placeholder="Or enter an absolute/custom test-list path" disabled={Boolean(activeJob) || submitting} /></div>
              <div className="evaluation-field-row"><div className="evaluation-field"><label htmlFor="dataset-name">Dataset</label><select id="dataset-name" value={datasetName} onChange={(event) => setDatasetName(event.target.value)} disabled={Boolean(activeJob) || submitting}><option value="brats19">BraTS 2019</option><option value="brats20">BraTS 2020</option><option value="brats23men">BraTS 2023 meningioma</option><option value="msdbts">MSD BraTS</option></select></div><div className="evaluation-field"><label htmlFor="class-type">Labels</label><input id="class-type" value="All labels" readOnly aria-describedby="class-type-note" /><small id="class-type-note">HFF-Net checkpoints are evaluated with all trained classes.</small></div></div>
              <div className="evaluation-field"><label htmlFor="output-dir">Output directory</label><input id="output-dir" value={outputDir} onChange={(event) => setOutputDir(event.target.value)} placeholder="result/cross_eval" disabled={Boolean(activeJob) || submitting} /><small>Relative paths are resolved from the HFF project root.</small></div>
              <div className="evaluation-field"><label htmlFor="monitor-interval">Telemetry interval (seconds)</label><input id="monitor-interval" type="number" min="1" max="300" step="1" value={monitorInterval} onChange={(event) => setMonitorInterval(event.target.value)} disabled={Boolean(activeJob) || submitting} /><small>Samples are written to JSONL and summarized at the end of the job.</small></div>
              <Button type="submit" className="evaluation-submit" disabled={!canSubmit}>{submitting ? "Starting…" : activeJob ? "Evaluation running…" : "Start evaluation"}</Button>
              {activeJob && <div className="evaluation-running-note"><span className="topbar-pulse" /> Job <code>{activeJob.id}</code> is running. Only one GPU evaluation is allowed at a time.</div>}
            </CardContent>
          </Card>
        </form>

        <section className="evaluation-history"><div className="evaluation-section-heading"><div><h2>Evaluation history</h2><p>Results remain in each selected output directory.</p></div><span>{jobs.length} jobs</span></div>{loading ? <div className="evaluation-empty">Loading evaluation options…</div> : jobs.length ? jobs.map((job) => <JobCard key={job.id} job={job} />) : <div className="evaluation-empty"><strong>No evaluation jobs yet</strong><span>Select one or more checkpoints to begin.</span></div>}</section>
      </div>
    </section>
  );
}
