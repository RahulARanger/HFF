import { useCallback, useEffect, useMemo, useState } from "react";

import {
  fetchEvaluationJobs,
  fetchEvaluationOptions,
  renameEvaluation,
  startEvaluation,
} from "../api.js";
import { Badge, Button, Card, CardContent, CardHeader, CardTitle } from "./watermelon-ui.jsx";
import { EvaluationResultTabs } from "./EvaluationResultTables.jsx";

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

function evaluationDisplayName(job) {
  const name = typeof job.name === "string" ? job.name.trim() : "";
  return name || `Evaluation job ${job.id}`;
}

function formatCheckpointScore(value) {
  if (value === null || value === undefined || value === "") return "—";
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue.toFixed(4) : "—";
}

function CheckpointPicker({ checkpoints, selected, onChange, disabled, emptyMessage }) {
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
          <span className="evaluation-checkpoint-content">
            <span className="evaluation-checkpoint-heading">
              <strong>{checkpoint.name || checkpoint.label || checkpoint.path}</strong>
              {checkpoint.is_last_save && <em>Last save</em>}
              {checkpoint.fold_name && <small>{checkpoint.fold_name}</small>}
            </span>
            <span className="evaluation-checkpoint-scores" aria-label="Dice scores">
              {[["ET", checkpoint.scores?.et], ["TC", checkpoint.scores?.tc], ["WT", checkpoint.scores?.wt], ["Avg", checkpoint.average_dice]].map(([label, value]) => (
                <span key={label}><small>{label}</small><strong>{formatCheckpointScore(value)}</strong></span>
              ))}
            </span>
            {checkpoint.is_last_save && <small className="evaluation-last-save-note">mJc {formatCheckpointScore(checkpoint.last_save_metric)} · final training save</small>}
          </span>
        </label>
      )) : <div className="evaluation-empty">{emptyMessage}</div>}
    </div>
  );
}

function JobCard({ job, onRenamed }) {
  const [editingName, setEditingName] = useState(false);
  const [draftName, setDraftName] = useState(job.name || "");
  const [savingName, setSavingName] = useState(false);
  const [renameError, setRenameError] = useState("");
  const request = job.request || {};
  const summary = job.summary || {};

  useEffect(() => {
    if (!editingName) setDraftName(job.name || "");
  }, [editingName, job.name]);

  const handleRename = async () => {
    setSavingName(true);
    setRenameError("");
    try {
      const updatedJob = await renameEvaluation(job.id, draftName.trim() || null);
      onRenamed(updatedJob);
      setEditingName(false);
    } catch (renameRequestError) {
      setRenameError(renameRequestError.message);
    } finally {
      setSavingName(false);
    }
  };

  return (
    <Card className="evaluation-job-card">
      <CardHeader>
        <div>
          <CardTitle>{evaluationDisplayName(job)}</CardTitle>
          <div className="wm-card-description">{formatTime(job.created_at)} · {request.checkpoints?.length || 0} checkpoint(s) · ID <code>{job.id}</code></div>
        </div>
        <div className="evaluation-job-actions">
          <StatusBadge status={job.status} />
          <Button type="button" variant="outline" className="evaluation-rename-button" onClick={() => { setRenameError(""); setEditingName(true); }} disabled={savingName}>{editingName ? "Editing…" : "Rename"}</Button>
        </div>
      </CardHeader>
      <CardContent>
        {editingName && <div className="evaluation-name-editor"><label htmlFor={`rename-evaluation-${job.id}`}>Evaluation name</label><div><input id={`rename-evaluation-${job.id}`} value={draftName} maxLength={120} autoFocus onChange={(event) => setDraftName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); handleRename(); } if (event.key === "Escape") setEditingName(false); }} placeholder="Leave blank to use the generated ID" disabled={savingName} /><Button type="button" className="evaluation-name-save" onClick={handleRename} disabled={savingName}>{savingName ? "Saving…" : "Save"}</Button><Button type="button" variant="outline" className="evaluation-name-cancel" onClick={() => setEditingName(false)} disabled={savingName}>Cancel</Button></div><small>Up to 120 characters. This changes the display label only.</small></div>}
        {renameError && <div className="app-alert evaluation-alert">{renameError}</div>}
        <div className="evaluation-job-meta">
          <span><label>Dataset</label><strong>{request.dataset_name || "—"}</strong></span>
          <span><label>Test list</label><strong title={request.test_list}>{shortPath(request.test_list)}</strong></span>
          <span><label>Output</label><strong title={request.output_dir}>{shortPath(request.output_dir)}</strong></span>
        </div>
        {job.error && <div className="app-alert evaluation-alert">{job.error}</div>}
        {job.status === "completed" && <EvaluationResultTabs summary={summary} />}
        {job.status !== "completed" && job.log_tail && <pre className="evaluation-log">{job.log_tail}</pre>}
        {job.status === "completed" && <div className="evaluation-job-files"><span>Summary</span><code title={job.summary_file}>{shortPath(job.summary_file)}</code></div>}
      </CardContent>
    </Card>
  );
}

export default function EvaluationView() {
  const [options, setOptions] = useState({ checkpoints: [], checkpoint_groups: [], test_lists: [], defaults: {} });
  const [selectedRunName, setSelectedRunName] = useState("");
  const [showLastSave, setShowLastSave] = useState(false);
  const [selectedCheckpoints, setSelectedCheckpoints] = useState([]);
  const [testList, setTestList] = useState("");
  const [datasetName, setDatasetName] = useState("brats19");
  const [evaluationName, setEvaluationName] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [monitorInterval, setMonitorInterval] = useState("5");
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const loadOptions = useCallback(async () => {
    const payload = await fetchEvaluationOptions();
    const groups = payload.checkpoint_groups || [];
    const availableCheckpointPaths = new Set(groups.flatMap((group) => (group.checkpoints || []).map((checkpoint) => checkpoint.path)));
    setOptions(payload);
    setSelectedRunName((current) => groups.some((group) => group.name === current) ? current : "");
    setTestList((current) => current || payload.defaults?.test_list || payload.test_lists?.[0]?.path || "");
    setOutputDir((current) => current || payload.defaults?.output_dir || "");
    setSelectedCheckpoints((current) => current.filter((path) => availableCheckpointPaths.has(path)));
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
  const selectedRun = (options.checkpoint_groups || []).find((group) => group.name === selectedRunName);
  const runCheckpoints = selectedRun?.checkpoints || [];
  const visibleCheckpoints = showLastSave ? runCheckpoints : runCheckpoints.filter((checkpoint) => !checkpoint.is_last_save);
  const canSubmit = selectedCheckpoints.length > 0 && testList.trim() && outputDir.trim() && !activeJob && !submitting;

  const handleRunChange = (event) => {
    setSelectedRunName(event.target.value);
    setSelectedCheckpoints([]);
    setShowLastSave(false);
  };

  const handleShowLastSaveChange = (event) => {
    const nextValue = event.target.checked;
    setShowLastSave(nextValue);
    if (!nextValue) {
      setSelectedCheckpoints((current) => current.filter((path) => !runCheckpoints.some((checkpoint) => checkpoint.path === path && checkpoint.is_last_save)));
    }
  };

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
        name: evaluationName.trim() || null,
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

  const handleJobRenamed = (updatedJob) => {
    setJobs((current) => current.map((job) => job.id === updatedJob.id ? updatedJob : job));
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
              <div className="evaluation-field"><label htmlFor="evaluation-run">Training run</label><select id="evaluation-run" value={selectedRunName} onChange={handleRunChange} disabled={Boolean(activeJob) || submitting}><option value="">Select a training run</option>{(options.checkpoint_groups || []).map((group) => <option key={group.name} value={group.name}>{group.label || group.name}</option>)}</select><small>Choose the training configuration first, then select its checkpoints.</small></div>
              <div className="evaluation-field"><label>Checkpoints <span>{selectedCheckpoints.length} selected</span></label><div className="evaluation-checkpoint-toolbar"><span>{selectedRun ? `${visibleCheckpoints.length} shown of ${runCheckpoints.length} checkpoint(s) · newest saves first` : "Select a training run to see checkpoints"}</span><label className="evaluation-switch"><input type="checkbox" checked={showLastSave} onChange={handleShowLastSaveChange} disabled={!selectedRun || Boolean(activeJob) || submitting} /><span className="evaluation-switch-track" aria-hidden="true" /><strong>Show Last Save</strong></label></div><CheckpointPicker checkpoints={visibleCheckpoints} selected={selectedCheckpoints} onChange={setSelectedCheckpoints} disabled={Boolean(activeJob) || submitting} emptyMessage={selectedRun ? "No score-named checkpoints were found for this training run." : "Select a training run to see checkpoints."} /></div>
              <div className="evaluation-field"><label htmlFor="evaluation-name">Evaluation name <span>Optional</span></label><input id="evaluation-name" value={evaluationName} maxLength={120} onChange={(event) => setEvaluationName(event.target.value)} placeholder="e.g. BraTS 2019 baseline" disabled={Boolean(activeJob) || submitting} /><small>Use a memorable label, or leave blank to keep the generated job ID.</small></div>
              <div className="evaluation-field"><label htmlFor="test-list">Test list</label><select id="test-list" value={testList} onChange={(event) => setTestList(event.target.value)} disabled={Boolean(activeJob) || submitting}><option value="">Select a discovered test list</option>{(options.test_lists || []).map((item) => <option key={item.path} value={item.path}>{item.label}</option>)}</select><input aria-label="Custom test list path" value={testList} onChange={(event) => setTestList(event.target.value)} placeholder="Or enter an absolute/custom test-list path" disabled={Boolean(activeJob) || submitting} /></div>
              <div className="evaluation-field-row"><div className="evaluation-field"><label htmlFor="dataset-name">Dataset</label><select id="dataset-name" value={datasetName} onChange={(event) => setDatasetName(event.target.value)} disabled={Boolean(activeJob) || submitting}><option value="brats19">BraTS 2019</option><option value="brats20">BraTS 2020</option><option value="brats23men">BraTS 2023 meningioma</option><option value="msdbts">MSD BraTS</option></select></div><div className="evaluation-field"><label htmlFor="class-type">Labels</label><input id="class-type" value="All labels" readOnly aria-describedby="class-type-note" /><small id="class-type-note">HFF-Net checkpoints are evaluated with all trained classes.</small></div></div>
              <div className="evaluation-field"><label htmlFor="output-dir">Output directory</label><input id="output-dir" value={outputDir} onChange={(event) => setOutputDir(event.target.value)} placeholder="result/cross_eval" disabled={Boolean(activeJob) || submitting} /><small>Relative paths are resolved from the HFF project root.</small></div>
              <div className="evaluation-field"><label htmlFor="monitor-interval">Telemetry interval (seconds)</label><input id="monitor-interval" type="number" min="1" max="300" step="1" value={monitorInterval} onChange={(event) => setMonitorInterval(event.target.value)} disabled={Boolean(activeJob) || submitting} /><small>Samples are written to JSONL and summarized at the end of the job.</small></div>
              <Button type="submit" className="evaluation-submit" disabled={!canSubmit}>{submitting ? "Starting…" : activeJob ? "Evaluation running…" : "Start evaluation"}</Button>
              {activeJob && <div className="evaluation-running-note"><span className="topbar-pulse" /> Job <code>{activeJob.id}</code> is running. Only one GPU evaluation is allowed at a time.</div>}
            </CardContent>
          </Card>
        </form>

        <section className="evaluation-history"><div className="evaluation-section-heading"><div><h2>Evaluation history</h2><p>Results remain in each selected output directory.</p></div><span>{jobs.length} jobs</span></div>{loading ? <div className="evaluation-empty">Loading evaluation options…</div> : jobs.length ? jobs.map((job) => <JobCard key={job.id} job={job} onRenamed={handleJobRenamed} />) : <div className="evaluation-empty"><strong>No evaluation jobs yet</strong><span>Select one or more checkpoints to begin.</span></div>}</section>
      </div>
    </section>
  );
}
