"""Evaluate the same test split with several HFF-Net checkpoints.

The inference implementation lives in ``eval.py``. This wrapper only owns
checkpoint-list parsing, per-checkpoint result persistence, and aggregation, so
the single-checkpoint and multi-checkpoint workflows remain mathematically
aligned.

The checkpoint list is a UTF-8 text file with one checkpoint path per line.
Blank lines and lines beginning with ``#`` are ignored. Relative checkpoint
paths are resolved relative to the checkpoint-list file, not the current
working directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from eval import add_evaluation_arguments, evaluate_checkpoint


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HFF-Net eval.py inference for every checkpoint in a list and average the results."
    )
    parser.add_argument(
        '--checkpoint_list',
        type=Path,
        required=True,
        help='Text file containing one checkpoint path per line.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('./result/cross_eval'),
        help='Directory for per-checkpoint JSON files and the aggregate summary.',
    )
    parser.add_argument(
        '--progress_file',
        type=Path,
        default=None,
        help='Optional JSON file updated with sample-level inference progress.',
    )
    add_evaluation_arguments(
        parser,
        include_checkpoint=False,
        include_output_dir=False,
    )
    return parser


def read_checkpoint_list(list_path: Path) -> list[Path]:
    """Read and validate checkpoint paths from a plain-text checkpoint list."""
    list_path = list_path.expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f'Checkpoint list does not exist: {list_path}')

    checkpoints: list[Path] = []
    for line_number, line in enumerate(list_path.read_text(encoding='utf-8').splitlines(), start=1):
        checkpoint_text = line.strip()
        if not checkpoint_text or checkpoint_text.startswith('#'):
            continue

        checkpoint = Path(checkpoint_text).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = list_path.parent / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f'Checkpoint on line {line_number} does not exist: {checkpoint}'
            )
        checkpoints.append(checkpoint)

    if not checkpoints:
        raise ValueError(f'Checkpoint list contains no checkpoint paths: {list_path}')
    return checkpoints


def finite_float(value: Any) -> float | None:
    """Convert a scalar to JSON-safe float, representing NaN/Inf as null."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def json_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize NumPy/PyTorch scalar outputs before writing JSON."""
    safe_result = dict(result)
    safe_result['validation_loss_branch_1'] = finite_float(
        result['validation_loss_branch_1']
    )
    safe_result['validation_loss_branch_2'] = finite_float(
        result['validation_loss_branch_2']
    )
    safe_result['metrics'] = {
        branch: [finite_float(metric) for metric in values]
        for branch, values in result['metrics'].items()
    }
    safe_result['jaccard'] = {
        branch: [finite_float(metric) for metric in values]
        for branch, values in result.get('jaccard', {}).items()
    }
    return safe_result


def mean(values: Iterable[Any]) -> tuple[float | None, int]:
    finite_values = [value for value in (finite_float(item) for item in values) if value is not None]
    if not finite_values:
        return None, 0
    return sum(finite_values) / len(finite_values), len(finite_values)


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute element-wise means across successful checkpoint evaluations."""
    branch_names = sorted({
        branch
        for result in results
        for branch in result['metrics']
    })
    average_metrics: dict[str, list[float | None]] = {}
    metric_counts: dict[str, list[int]] = {}
    for branch in branch_names:
        metric_count = max(len(result['metrics'].get(branch, [])) for result in results)
        averages = []
        counts = []
        for metric_index in range(metric_count):
            values = [
                result['metrics'][branch][metric_index]
                for result in results
                if metric_index < len(result['metrics'].get(branch, []))
            ]
            metric_average, metric_count = mean(values)
            averages.append(metric_average)
            counts.append(metric_count)
        average_metrics[branch] = averages
        metric_counts[branch] = counts

    jaccard_branches = sorted({
        branch
        for result in results
        for branch in result.get('jaccard', {})
    })
    average_jaccard: dict[str, list[float | None]] = {}
    jaccard_counts: dict[str, list[int]] = {}
    for branch in jaccard_branches:
        metric_count = max(len(result.get('jaccard', {}).get(branch, [])) for result in results)
        averages = []
        counts = []
        for metric_index in range(metric_count):
            values = [
                result.get('jaccard', {}).get(branch, [])[metric_index]
                for result in results
                if metric_index < len(result.get('jaccard', {}).get(branch, []))
            ]
            metric_average, metric_count = mean(values)
            averages.append(metric_average)
            counts.append(metric_count)
        average_jaccard[branch] = averages
        jaccard_counts[branch] = counts

    loss_averages = {}
    loss_counts = {}
    for field in ('validation_loss_branch_1', 'validation_loss_branch_2'):
        loss_averages[field], loss_counts[field] = mean(result[field] for result in results)

    return {
        'checkpoint_count': len(results),
        'average_validation_losses': loss_averages,
        'average_metrics': average_metrics,
        'average_jaccard': average_jaccard,
        'valid_value_counts': {
            'validation_losses': loss_counts,
            'metrics': metric_counts,
            'jaccard': jaccard_counts,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        delete=False,
    ) as temporary_file:
        json.dump(payload, temporary_file, indent=2, allow_nan=False)
        temporary_file.write('\n')
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    """Persist progress so a parent process can expose it to the web UI."""
    if path is None:
        return
    write_json(path.expanduser().resolve(), payload)


def checkpoint_result_path(output_dir: Path, index: int, checkpoint: Path) -> Path:
    safe_stem = re.sub(r'[^A-Za-z0-9._-]+', '_', checkpoint.stem)
    return output_dir / f'checkpoint_{index:03d}_{safe_stem}.json'


def build_summary(args: argparse.Namespace, checkpoint_list: Path, results: list[dict[str, Any]], result_files: list[str]) -> dict[str, Any]:
    """Build the current summary so the monitor can expose completed checkpoints early."""
    return {
        'checkpoint_list': str(checkpoint_list),
        'test_list': str(Path(args.test_list).expanduser().resolve()),
        'dataset_name': args.dataset_name,
        'class_type': args.class_type,
        'result_files': result_files,
        'results': results,
        'partial': False,
        **aggregate_results(results),
    }


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_list = args.checkpoint_list.expanduser().resolve()
    checkpoints = read_checkpoint_list(checkpoint_list)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = args.progress_file.expanduser().resolve() if args.progress_file else None

    results: list[dict[str, Any]] = []
    result_files: list[str] = []
    write_progress(progress_file, {
        'phase': 'starting',
        'checkpoint_index': 0,
        'checkpoint_count': len(checkpoints),
        'processed_samples': 0,
        'total_samples': None,
        'overall_processed_samples': 0,
        'overall_total_samples': None,
    })
    for index, checkpoint in enumerate(checkpoints, start=1):
        LOGGER.info('Evaluating checkpoint %d/%d: %s', index, len(checkpoints), checkpoint)

        def report_progress(update: dict[str, Any]) -> None:
            total_samples = update.get('total_samples')
            processed_samples = update.get('processed_samples', 0)
            overall_total = total_samples * len(checkpoints) if isinstance(total_samples, int) else None
            overall_processed = (
                (index - 1) * total_samples + processed_samples
                if isinstance(total_samples, int)
                else 0
            )
            progress_payload = {
                'phase': 'inference',
                'checkpoint_index': index,
                'checkpoint_count': len(checkpoints),
                'checkpoint_name': checkpoint.name,
                'processed_samples': processed_samples,
                'total_samples': total_samples,
                'overall_processed_samples': overall_processed,
                'overall_total_samples': overall_total,
            }
            if update.get('metrics') and update.get('jaccard'):
                live_result = json_safe_result({
                    'checkpoint': str(checkpoint),
                    'test_list': str(Path(args.test_list).expanduser().resolve()),
                    'dataset_name': args.dataset_name,
                    'class_type': args.class_type,
                    'num_samples': processed_samples,
                    'validation_loss_branch_1': update.get('validation_loss_branch_1'),
                    'validation_loss_branch_2': update.get('validation_loss_branch_2'),
                    'metrics': update['metrics'],
                    'jaccard': update['jaccard'],
                    'partial': True,
                })
                progress_payload.update({
                    'validation_loss_branch_1': live_result['validation_loss_branch_1'],
                    'validation_loss_branch_2': live_result['validation_loss_branch_2'],
                    'metrics': live_result['metrics'],
                    'jaccard': live_result['jaccard'],
                })
                live_summary = build_summary(args, checkpoint_list, [*results, live_result], result_files)
                live_summary['partial'] = True
                live_summary['completed_checkpoint_count'] = len(results)
                write_json(output_dir / 'cross_eval_summary.json', live_summary)
            write_progress(progress_file, progress_payload)

        result = json_safe_result(
            evaluate_checkpoint(
                args,
                checkpoint_path=checkpoint,
                progress_callback=report_progress,
            )
        )
        result_path = checkpoint_result_path(output_dir, index, checkpoint)
        write_json(result_path, result)
        results.append(result)
        result_files.append(str(result_path))
        LOGGER.info('Saved checkpoint result to %s', result_path)
        checkpoint_summary = build_summary(args, checkpoint_list, results, result_files)
        checkpoint_summary['partial'] = index < len(checkpoints)
        checkpoint_summary['completed_checkpoint_count'] = index
        write_json(output_dir / 'cross_eval_summary.json', checkpoint_summary)
        write_progress(progress_file, {
            'phase': 'checkpoint_complete',
            'checkpoint_index': index,
            'checkpoint_count': len(checkpoints),
            'checkpoint_name': checkpoint.name,
            'processed_samples': result.get('num_samples', 0),
            'total_samples': result.get('num_samples', 0),
            'overall_processed_samples': index * result.get('num_samples', 0),
            'overall_total_samples': len(checkpoints) * result.get('num_samples', 0),
        })

    summary = build_summary(args, checkpoint_list, results, result_files)
    summary_path = output_dir / 'cross_eval_summary.json'
    write_json(summary_path, summary)
    completed_samples = results[-1].get('num_samples', 0) if results else 0
    write_progress(progress_file, {
        'phase': 'completed',
        'checkpoint_index': len(checkpoints),
        'checkpoint_count': len(checkpoints),
        'processed_samples': completed_samples,
        'total_samples': completed_samples,
        'overall_processed_samples': len(results) * completed_samples,
        'overall_total_samples': len(results) * completed_samples,
    })

    LOGGER.info('Saved cross-checkpoint summary to %s', summary_path)
    print('\nAverage across %d checkpoints:' % len(results))
    print(json.dumps(summary['average_metrics'], indent=2))
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    raise SystemExit(main())
