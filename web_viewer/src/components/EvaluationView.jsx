import { useCallback, useEffect, useMemo, useState } from "react";

import {
  fetchEvaluationJobs,
  fetchEvaluationOptions,
  renameEvaluation,
} from "../api.js";
import { Badge, Button, Card, CardContent, CardHeader, CardTitle } from "./watermelon-ui.jsx";
import { EvaluationResultTabs } from "./EvaluationResultTables.jsx";

const POLL_INTERVAL_MS = 3000;
const EVALUATION_CONFIGURATION_KEY = "hff-net:evaluation-configuration:v1";

function readSavedEvaluationConfiguration() {
  const defaults = {
    selectedRunName: "",
    selectedFoldName: "all",
    showLastSave: false,
    selectedCheckpoints: [],
    testList: "",
    datasetName: "brats19",
    evaluationName: "",
    outputDir: "",
    gpuDevice: "",
    condaBase: "/apps/compilers/anaconda3",
    condaEnv: "hffnet",
    savedAt: "",
  };
  if (typeof window === "undefined") return defaults;
  try {
    const saved = JSON.parse(window.localStorage.getItem(EVALUATION_CONFIGURATION_KEY) || "null");
    if (!saved || typeof saved !== "object") return defaults;
    return {
      ...defaults,
      ...saved,
      selectedCheckpoints: Array.isArray(saved.selectedCheckpoints)
        ? saved.selectedCheckpoints.filter((path) => typeof path === "string")
        : defaults.selectedCheckpoints,
    };
  } catch {
    return defaults;
  }
}

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

function shellQuote(value) {
  return `'${String(value).replaceAll("'", "'\"'\"'")}'`;
}

function trimTrailingSlashes(value) {
  return String(value).replace(/\/+$/, "");
}

function checkpointListPath(outputDir, evaluationName) {
  const name = evaluationName.trim().replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^\.+/, "");
  const suffix = name || "manual";
  return `${trimTrailingSlashes(outputDir)}/checkpoint_list_eval_${suffix}.txt`;
}

function evaluationJobName(evaluationName) {
  const name = evaluationName.trim().replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^\.+/, "");
  return `hff_eval_${(name || "manual").slice(0, 40)}`;
}

function buildCheckpointListPreparation(outputDir, listPath, checkpoints) {
  return [
    `mkdir -p ${shellQuote(outputDir)}`,
    `printf '%s\\n' ${checkpoints.map(shellQuote).join(" ")} > ${shellQuote(listPath)}`,
  ].join(" && \\\n");
}

function buildEvaluationArguments({ listPath, testList, datasetName, outputDir, progressPath }) {
  return [
    `--checkpoint_list ${shellQuote(listPath)}`,
    `--test_list ${shellQuote(testList)}`,
    `--dataset_name ${shellQuote(datasetName)}`,
    "--class_type all",
    "--batch_size 1",
    "--num_workers 3",
    `--output_dir ${shellQuote(outputDir)}`,
    `--progress_file ${shellQuote(progressPath)}`,
  ].join(" \\\n  ");
}

function CommandBox({ title, description, command, note }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setCopied(false);
  }, [command]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <section className="evaluation-command-box" aria-label={title}>
      <div className="evaluation-command-heading">
        <div><h3>{title}</h3><p>{description}</p></div>
        <button type="button" className="evaluation-copy-button" onClick={handleCopy} disabled={!command}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className={`evaluation-command ${command ? "" : "empty"}`}><code>{command || "Select checkpoints, a test list, and an output directory to generate this command."}</code></pre>
      {note && <small className="evaluation-command-note">{note}</small>}
    </section>
  );
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
  const progress = job.progress || {};
  const processedSamples = Number(progress.overall_processed_samples ?? progress.processed_samples ?? 0);
  const totalSamples = Number(progress.overall_total_samples ?? progress.total_samples ?? 0);
  const progressPercent = totalSamples > 0 ? Math.min(100, (processedSamples / totalSamples) * 100) : 0;

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
        {job.progress && <div className="evaluation-progress"><div className="evaluation-progress-heading"><span>Inference progress</span><strong>{totalSamples > 0 ? `${processedSamples.toLocaleString()} / ${totalSamples.toLocaleString()} samples` : "Preparing samples…"}</strong></div><div className="evaluation-progress-track"><span style={{ width: `${progressPercent}%` }} /></div><small>{progress.checkpoint_count ? `Checkpoint ${progress.checkpoint_index || 0} / ${progress.checkpoint_count}` : ""}{progress.checkpoint_name ? ` · ${progress.checkpoint_name}` : ""}</small></div>}
        {job.error && <div className="app-alert evaluation-alert">{job.error}</div>}
        {job.status === "completed" && <EvaluationResultTabs summary={summary} />}
        {job.status !== "completed" && job.log_tail && <pre className="evaluation-log">{job.log_tail}</pre>}
        {job.log_file && <div className="evaluation-job-files"><span>Backend log</span><code title={job.log_file}>{shortPath(job.log_file)}</code></div>}
        {job.status === "completed" && <div className="evaluation-job-files"><span>Summary</span><code title={job.summary_file}>{shortPath(job.summary_file)}</code></div>}
      </CardContent>
    </Card>
  );
}

export default function EvaluationView() {
  const [initialConfiguration] = useState(readSavedEvaluationConfiguration);
  const [options, setOptions] = useState({ checkpoints: [], checkpoint_groups: [], test_lists: [], defaults: {} });
  const [selectedRunName, setSelectedRunName] = useState(initialConfiguration.selectedRunName);
  const [selectedFoldName, setSelectedFoldName] = useState(initialConfiguration.selectedFoldName);
  const [showLastSave, setShowLastSave] = useState(initialConfiguration.showLastSave);
  const [selectedCheckpoints, setSelectedCheckpoints] = useState(initialConfiguration.selectedCheckpoints);
  const [testList, setTestList] = useState(initialConfiguration.testList);
  const [datasetName, setDatasetName] = useState(initialConfiguration.datasetName);
  const [evaluationName, setEvaluationName] = useState(initialConfiguration.evaluationName);
  const [outputDir, setOutputDir] = useState(initialConfiguration.outputDir);
  const [gpuDevice, setGpuDevice] = useState(initialConfiguration.gpuDevice);
  const [condaBase, setCondaBase] = useState(initialConfiguration.condaBase);
  const [condaEnv, setCondaEnv] = useState(initialConfiguration.condaEnv);
  const [configurationSavedAt, setConfigurationSavedAt] = useState(initialConfiguration.savedAt);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
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

  const selectedRun = (options.checkpoint_groups || []).find((group) => group.name === selectedRunName);
  const runCheckpoints = selectedRun?.checkpoints || [];
  const availableFolds = [...new Set(runCheckpoints.map((checkpoint) => checkpoint.fold_name).filter(Boolean))].sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
  const foldCheckpoints = selectedFoldName === "all" ? runCheckpoints : runCheckpoints.filter((checkpoint) => checkpoint.fold_name === selectedFoldName);
  const visibleCheckpoints = showLastSave ? foldCheckpoints : foldCheckpoints.filter((checkpoint) => !checkpoint.is_last_save);
  const commandsReady = Boolean(selectedCheckpoints.length > 0 && testList.trim() && outputDir.trim());
  const commandPaths = useMemo(() => {
    const listPath = checkpointListPath(outputDir || "result/cross_eval", evaluationName);
    const progressPath = `${trimTrailingSlashes(outputDir || "result/cross_eval")}/cross_eval_progress_manual.json`;
    const projectRoot = trimTrailingSlashes(options.project_root || ".");
    const pbsScript = `${projectRoot}/scripts/submit_eval_gpu.pbs`;
    const qsubEnvironment = [
      `HFF_CONDA_BASE=${condaBase.trim() || "/apps/compilers/anaconda3"}`,
      `HFF_GPU_DEVICE=${gpuDevice.trim() || "REPLACE_WITH_MIG_UUID"}`,
      "WANDB_MODE=offline",
      `HFF_CONDA_ENV=${condaEnv.trim() || "hffnet"}`,
      `HFF_EVAL_NAME=${evaluationJobName(evaluationName)}`,
      "HFF_RESOURCE_MONITOR_INTERVAL=5",
      `HFF_REPO_ROOT=${projectRoot}`,
    ].join(",");
    const preparation = commandsReady
      ? buildCheckpointListPreparation(outputDir.trim(), listPath, selectedCheckpoints)
      : "";
    const argumentsBlock = commandsReady
      ? buildEvaluationArguments({
        listPath,
        testList: testList.trim(),
        datasetName,
        outputDir: outputDir.trim(),
        progressPath,
      })
      : "";
    return {
      direct: preparation ? `${preparation} && \\\npython cross_eval.py \\\n  ${argumentsBlock}` : "",
      pbs: preparation ? `${preparation} && \\\nqsub \\\n  -q workq \\\n  -N ${shellQuote(evaluationJobName(evaluationName))} \\\n  -l select=1:ncpus=12:mem=32gb:ngpus=1 \\\n  -l walltime=48:00:00 \\\n  -j oe \\\n  -v ${shellQuote(qsubEnvironment)} \\\n  -- \\\n  ${shellQuote(pbsScript)} \\\n  ${argumentsBlock}` : "",
    };
  }, [commandsReady, condaBase, condaEnv, datasetName, evaluationName, gpuDevice, options.project_root, outputDir, selectedCheckpoints, testList]);

  const handleRunChange = (event) => {
    setSelectedRunName(event.target.value);
    setSelectedFoldName("all");
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

  const handleSaveConfiguration = () => {
    const savedAt = new Date().toISOString();
    const configuration = {
      selectedRunName,
      selectedFoldName,
      showLastSave,
      selectedCheckpoints,
      testList: testList.trim(),
      datasetName,
      evaluationName: evaluationName.trim(),
      outputDir: outputDir.trim(),
      gpuDevice: gpuDevice.trim(),
      condaBase: condaBase.trim(),
      condaEnv: condaEnv.trim(),
      savedAt,
    };
    try {
      window.localStorage.setItem(EVALUATION_CONFIGURATION_KEY, JSON.stringify(configuration));
      setConfigurationSavedAt(savedAt);
      setError("");
    } catch (saveError) {
      setError(`Could not save the evaluation configuration locally: ${saveError.message}`);
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
        <div className="evaluation-form">
          <Card className="evaluation-form-card">
            <CardHeader><div><CardTitle>Evaluation setup</CardTitle><div className="wm-card-description">Select the inputs below, then copy one of the generated commands. This page does not launch evaluation directly.</div></div></CardHeader>
            <CardContent>
              <div className="evaluation-field"><label htmlFor="evaluation-run">Training run</label><select id="evaluation-run" value={selectedRunName} onChange={handleRunChange}><option value="">Select a training run</option>{(options.checkpoint_groups || []).map((group) => <option key={group.name} value={group.name}>{group.label || group.name}</option>)}</select><small>Choose the training configuration first, then select its checkpoints.</small></div>
              <div className="evaluation-field"><label>Checkpoints <span>{selectedCheckpoints.length} selected</span></label><div className="evaluation-checkpoint-toolbar"><label className="evaluation-fold-filter" htmlFor="checkpoint-fold"><span>Fold</span><select id="checkpoint-fold" value={selectedFoldName} onChange={(event) => setSelectedFoldName(event.target.value)} disabled={!selectedRun}><option value="all">All folds</option>{availableFolds.map((foldName) => <option key={foldName} value={foldName}>{foldName}</option>)}</select></label><span>{selectedRun ? `${visibleCheckpoints.length} shown of ${runCheckpoints.length} checkpoint(s) · newest saves first` : "Select a training run to see checkpoints"}</span><label className="evaluation-switch"><input type="checkbox" checked={showLastSave} onChange={handleShowLastSaveChange} disabled={!selectedRun} /><span className="evaluation-switch-track" aria-hidden="true" /><strong>Show Last Save</strong></label></div><CheckpointPicker checkpoints={visibleCheckpoints} selected={selectedCheckpoints} onChange={setSelectedCheckpoints} emptyMessage={selectedRun ? "No score-named checkpoints were found for this training run." : "Select a training run to see checkpoints."} /></div>
              <div className="evaluation-field"><label htmlFor="evaluation-name">Evaluation name <span>Optional</span></label><input id="evaluation-name" value={evaluationName} maxLength={120} onChange={(event) => setEvaluationName(event.target.value)} placeholder="e.g. BraTS 2019 baseline" /><small>Used to make the generated checkpoint-list filename easier to recognize.</small></div>
              <div className="evaluation-field"><label htmlFor="test-list">Test list</label><select id="test-list" value={testList} onChange={(event) => setTestList(event.target.value)}><option value="">Select a discovered test list</option>{(options.test_lists || []).map((item) => <option key={item.path} value={item.path}>{item.label}</option>)}</select><input aria-label="Custom test list path" value={testList} onChange={(event) => setTestList(event.target.value)} placeholder="Or enter an absolute/custom test-list path" /></div>
              <div className="evaluation-field-row"><div className="evaluation-field"><label htmlFor="dataset-name">Dataset</label><select id="dataset-name" value={datasetName} onChange={(event) => setDatasetName(event.target.value)}><option value="brats19">BraTS 2019</option><option value="brats20">BraTS 2020</option><option value="brats23men">BraTS 2023 meningioma</option><option value="msdbts">MSD BraTS</option></select></div><div className="evaluation-field"><label htmlFor="class-type">Labels</label><input id="class-type" value="All labels" readOnly aria-describedby="class-type-note" /><small id="class-type-note">HFF-Net checkpoints are evaluated with all trained classes.</small></div></div>
              <div className="evaluation-field"><label htmlFor="output-dir">Output directory</label><input id="output-dir" value={outputDir} onChange={(event) => setOutputDir(event.target.value)} placeholder="result/cross_eval" /><small>Relative paths are resolved from the HFF project root.</small></div>
              <div className="evaluation-field"><label htmlFor="gpu-device">GPU / MIG device <span>Required for PBS</span></label><input id="gpu-device" value={gpuDevice} onChange={(event) => setGpuDevice(event.target.value)} placeholder="e.g. MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx or 0" autoComplete="off" /><small>Use one GPU index or MIG UUID from <code>nvidia-smi -L</code>. This value is used by the PBS command.</small></div>
              <div className="evaluation-field-row"><div className="evaluation-field"><label htmlFor="conda-base">Conda base <span>PBS</span></label><input id="conda-base" value={condaBase} onChange={(event) => setCondaBase(event.target.value)} placeholder="/apps/compilers/anaconda3" /><small>Passed as <code>HFF_CONDA_BASE</code>.</small></div><div className="evaluation-field"><label htmlFor="conda-env">Conda environment</label><input id="conda-env" value={condaEnv} onChange={(event) => setCondaEnv(event.target.value)} placeholder="hffnet" /><small>Passed as <code>HFF_CONDA_ENV</code>.</small></div></div>
              <div className="evaluation-configuration-actions"><Button type="button" className="evaluation-save-configuration" onClick={handleSaveConfiguration}>Save configuration</Button><span>{configurationSavedAt ? `Saved locally ${formatTime(configurationSavedAt)}` : "Configuration not saved yet"}</span></div>
              <div className="evaluation-command-section"><div className="evaluation-command-section-heading"><h2>Run commands</h2><p>Run these from the HFF repository root. Neither box starts a job from the viewer.</p></div><CommandBox title="1. Direct command" description="Runs cross_eval.py in the current shell." command={commandPaths.direct} /><CommandBox title="2. PBS / workq command" description="Submits one GPU job to workq with explicit GPU and Conda settings." command={commandPaths.pbs} note={gpuDevice.trim() ? "The command uses the GPU / MIG device and Conda base entered above." : "Enter a GPU / MIG device above; the command currently contains a placeholder."} /></div>
            </CardContent>
          </Card>
        </div>

        <section className="evaluation-history"><div className="evaluation-section-heading"><div><h2>Evaluation history</h2><p>Results remain in each selected output directory.</p></div><span>{jobs.length} jobs</span></div>{loading ? <div className="evaluation-empty">Loading evaluation options…</div> : jobs.length ? jobs.map((job) => <JobCard key={job.id} job={job} onRenamed={handleJobRenamed} />) : <div className="evaluation-empty"><strong>No evaluation jobs yet</strong><span>Select one or more checkpoints to begin.</span></div>}</section>
      </div>
    </section>
  );
}
