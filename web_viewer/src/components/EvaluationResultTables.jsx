import { useState } from "react";

const STRUCTURE_ROWS = [
  { label: "ET", metricIndex: 0 },
  { label: "TC", metricIndex: 1 },
  { label: "WT", metricIndex: 2 },
];

export function formatMetric(value) {
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

function MetricHeader({ tone, children }) {
  return <span className="evaluation-metric-header"><i className={`evaluation-metric-dot ${tone}`} />{children}</span>;
}

export function MetricResultsTable({ metrics, jaccard }) {
  const rows = metricRows(metrics);
  return (
    <div className="evaluation-results-table-wrap">
      <table className="evaluation-results-table">
        <caption>Segmentation metrics · Dice, Jaccard, and HD95</caption>
        <thead>
          <tr>
            <th scope="col">Region</th>
            <th scope="col"><MetricHeader tone="purple">{branchLabel("branch_1")} Dice</MetricHeader></th>
            <th scope="col"><MetricHeader tone="orange">{branchLabel("branch_1")} Jaccard</MetricHeader></th>
            <th scope="col"><MetricHeader tone="purple">{branchLabel("branch_1")} HD95</MetricHeader></th>
            <th scope="col"><MetricHeader tone="purple">{branchLabel("branch_2")} Dice</MetricHeader></th>
            <th scope="col"><MetricHeader tone="orange">{branchLabel("branch_2")} Jaccard</MetricHeader></th>
            <th scope="col"><MetricHeader tone="purple">{branchLabel("branch_2")} HD95</MetricHeader></th>
          </tr>
        </thead>
        <tbody>
          {rows.length ? rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">{row.label}</th>
              <td>{formatMetric(metrics?.branch_1?.[row.metricIndex * 2])}</td>
              <td>{formatMetric(jaccard?.branch_1?.[row.metricIndex])}</td>
              <td>{formatMetric(metrics?.branch_1?.[row.metricIndex * 2 + 1])}</td>
              <td>{formatMetric(metrics?.branch_2?.[row.metricIndex * 2])}</td>
              <td>{formatMetric(jaccard?.branch_2?.[row.metricIndex])}</td>
              <td>{formatMetric(metrics?.branch_2?.[row.metricIndex * 2 + 1])}</td>
            </tr>
          )) : (
            <tr><td colSpan="7">No metric values were written for this result.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export function LossResultsTable({ losses }) {
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

function shortPath(value) {
  if (!value) return "—";
  const parts = String(value).split("/");
  return parts.length > 3 ? `…/${parts.slice(-3).join("/")}` : value;
}

export function EvaluationResultTabs({ summary, className = "" }) {
  const [activeTab, setActiveTab] = useState("average");
  const results = summary?.results || [];
  const tabs = [
    {
      id: "average",
      label: "Average across checkpoints",
      metrics: summary?.average_metrics,
      jaccard: summary?.average_jaccard,
      losses: summary?.average_validation_losses,
    },
    ...results.map((result, index) => ({
      id: `checkpoint-${index}`,
      label: `Checkpoint ${index + 1}`,
      checkpoint: result.checkpoint,
      metrics: result.metrics,
      jaccard: result.jaccard,
      losses: {
        validation_loss_branch_1: result.validation_loss_branch_1,
        validation_loss_branch_2: result.validation_loss_branch_2,
      },
    })),
  ];
  const selected = tabs.find((tab) => tab.id === activeTab) || tabs[0];

  return (
    <section className={`evaluation-results ${className}`} aria-label="Evaluation results">
      <div className="evaluation-results-heading">
        <div><h3>Results by checkpoint</h3><span>{results.length} checkpoint result{results.length === 1 ? "" : "s"}</span></div>
        <span className="evaluation-results-note"><span><i className="evaluation-metric-dot purple" />Purple: author scores</span><span><i className="evaluation-metric-dot orange" />Orange: added Jaccard</span></span>
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
        <MetricResultsTable metrics={selected.metrics} jaccard={selected.jaccard} />
        <LossResultsTable losses={selected.losses} />
      </div>
    </section>
  );
}
