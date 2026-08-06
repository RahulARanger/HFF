"""Process-scoped CPU and accelerator memory monitoring for training runs.

The monitor is intentionally owned by ``train.py`` rather than scanning the
whole machine.  It samples the training process and its descendants, which
includes PyTorch DataLoader workers without attributing other users' work to
the experiment.

CUDA uses NVIDIA's NVML bindings (provided by ``nvidia-ml-py``).  MPS uses the
memory counters exposed by PyTorch because Apple does not provide an NVML-like
per-process API for Metal devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import atexit
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable

import psutil
import torch


LOGGER = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sum_optional(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _tracked_processes(root_pid: int) -> list[psutil.Process]:
    """Return the root process and currently living descendants."""
    try:
        root = psutil.Process(root_pid)
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _ram_usage(processes: Iterable[psutil.Process]) -> tuple[int, int | None, list[int]]:
    """Return RSS, optional USS, and the PIDs successfully sampled."""
    rss_bytes = 0
    uss_values: list[int] = []
    pids: list[int] = []
    for process in processes:
        try:
            with process.oneshot():
                memory = process.memory_info()
                rss_bytes += memory.rss
                pids.append(process.pid)
                try:
                    # USS is more precise but slower and may require extra
                    # privileges, so RSS remains the always-available metric.
                    uss_values.append(int(process.memory_full_info().uss))
                except (AttributeError, psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return rss_bytes, (_sum_optional(uss_values)), sorted(set(pids))


def _visible_device_token() -> str | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return None
    tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if not tokens:
        return None
    try:
        logical_index = torch.cuda.current_device()
    except (RuntimeError, AssertionError):
        logical_index = 0
    return tokens[logical_index] if logical_index < len(tokens) else tokens[0]


def _nvml_handles(nvml: Any) -> list[tuple[Any, str]]:
    """Discover physical and MIG handles, tolerating older driver bindings."""
    def decode_uuid(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    handles: list[tuple[Any, str]] = []
    for index in range(nvml.nvmlDeviceGetCount()):
        physical_handle = nvml.nvmlDeviceGetHandleByIndex(index)
        handles.append((physical_handle, decode_uuid(nvml.nvmlDeviceGetUUID(physical_handle))))

        get_mig_mode = getattr(nvml, "nvmlDeviceGetMigMode", None)
        get_max_mig_count = getattr(nvml, "nvmlDeviceGetMaxMigDeviceCount", None)
        get_mig_handle = getattr(nvml, "nvmlDeviceGetMigDeviceHandleByIndex", None)
        if not all((get_mig_mode, get_max_mig_count, get_mig_handle)):
            continue
        try:
            current_mode, _ = get_mig_mode(physical_handle)
            if not current_mode:
                continue
            for mig_index in range(get_max_mig_count(physical_handle)):
                try:
                    mig_handle = get_mig_handle(physical_handle, mig_index)
                    handles.append((mig_handle, decode_uuid(nvml.nvmlDeviceGetUUID(mig_handle))))
                except nvml.NVMLError:
                    continue
        except nvml.NVMLError:
            continue
    return handles


def _select_nvml_handles(nvml: Any, visible_token: str | None) -> list[tuple[Any, str]]:
    handles = _nvml_handles(nvml)
    if visible_token is None:
        try:
            logical_index = torch.cuda.current_device()
            physical_handles = [item for item in handles if not item[1].startswith("MIG-")]
            return physical_handles[logical_index : logical_index + 1] or handles[:1]
        except (RuntimeError, AssertionError):
            return handles[:1]

    if visible_token.isdigit():
        physical_index = int(visible_token)
        physical_handles = [item for item in handles if not item[1].startswith("MIG-")]
        return physical_handles[physical_index : physical_index + 1]

    matching = [item for item in handles if item[1] == visible_token]
    return matching


def _query_nvml_processes(nvml: Any, handle: Any) -> list[Any]:
    query_v3 = getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v3", None)
    query_v2 = getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v2", None)
    query_legacy = getattr(nvml, "nvmlDeviceGetComputeRunningProcesses", None)
    for query in (query_v3, query_v2, query_legacy):
        if query is None:
            continue
        try:
            return list(query(handle))
        except nvml.NVMLError:
            continue
    return []


@dataclass
class _PeakValues:
    ram_rss_bytes: int = 0
    ram_uss_bytes: int = 0
    gpu_memory_bytes: int = 0
    samples: int = 0

    def update(self, sample: dict[str, Any]) -> None:
        self.samples += 1
        self.ram_rss_bytes = max(self.ram_rss_bytes, int(sample.get("ram_rss_bytes", 0)))
        self.ram_uss_bytes = max(self.ram_uss_bytes, int(sample.get("ram_uss_bytes") or 0))
        self.gpu_memory_bytes = max(self.gpu_memory_bytes, int(sample.get("gpu_memory_bytes") or 0))


@dataclass
class ResourceMonitor:
    """Background monitor for one training process and its descendants."""

    device: torch.device
    output_directory: Path
    interval_seconds: float = 5.0
    root_pid: int = field(default_factory=os.getpid)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _file: Any = field(default=None, init=False)
    _started_at: float = field(default_factory=time.monotonic, init=False)
    _started_at_utc: str = field(default_factory=_utc_timestamp, init=False)
    _peak: _PeakValues = field(default_factory=_PeakValues, init=False)
    _sample: Callable[[], dict[str, Any]] | None = field(default=None, init=False)
    _stopped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("Resource monitor interval must be positive.")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.backend = self._backend_name()
        self.output_path = self.output_directory / f"{self.backend}_resource_usage.jsonl"
        self.summary_path = self.output_directory / f"{self.backend}_resource_summary.json"
        self._sample = self._make_sampler()

    def _backend_name(self) -> str:
        if self.device.type == "cuda":
            return "nvidia"
        if self.device.type == "mps":
            return "mps"
        return "cpu"

    def _make_sampler(self) -> Callable[[], dict[str, Any]]:
        if self.backend == "nvidia":
            try:
                import pynvml as nvml

                nvml.nvmlInit()
                return self._make_nvidia_sampler(nvml)
            except Exception as error:
                LOGGER.warning("NVIDIA monitoring unavailable; continuing with RAM monitoring: %s", error)
        if self.backend == "mps":
            return self._make_mps_sampler()
        return self._make_cpu_sampler()

    def _base_sample(self) -> dict[str, Any]:
        processes = _tracked_processes(self.root_pid)
        ram_rss, ram_uss, pids = _ram_usage(processes)
        return {
            "timestamp_utc": _utc_timestamp(),
            "elapsed_seconds": time.monotonic() - self._started_at,
            "root_pid": self.root_pid,
            "tracked_pids": pids,
            "backend": self.backend,
            "ram_rss_bytes": ram_rss,
            "ram_uss_bytes": ram_uss,
            "gpu_memory_bytes": None,
        }

    def _make_cpu_sampler(self) -> Callable[[], dict[str, Any]]:
        return self._base_sample

    def _make_mps_sampler(self) -> Callable[[], dict[str, Any]]:
        def sample() -> dict[str, Any]:
            result = self._base_sample()
            current = getattr(torch.mps, "current_allocated_memory", None)
            driver = getattr(torch.mps, "driver_allocated_memory", None)
            result["mps_current_allocated_bytes"] = int(current()) if current else None
            result["mps_driver_allocated_bytes"] = int(driver()) if driver else None
            result["gpu_memory_bytes"] = result["mps_driver_allocated_bytes"]
            return result

        return sample

    def _make_nvidia_sampler(self, nvml: Any) -> Callable[[], dict[str, Any]]:
        handles = _select_nvml_handles(nvml, _visible_device_token())
        if not handles:
            raise RuntimeError("Could not resolve CUDA_VISIBLE_DEVICES to an NVML device or MIG handle.")

        def sample() -> dict[str, Any]:
            result = self._base_sample()
            tracked_pids = set(result["tracked_pids"])
            process_memory: dict[str, int] = {}
            device_memory: dict[str, int] = {}
            for handle, uuid in handles:
                used_on_device = 0
                for process in _query_nvml_processes(nvml, handle):
                    pid = int(process.pid)
                    used_memory_value = getattr(process, "usedGpuMemory", None)
                    if used_memory_value is None:
                        continue
                    used_memory = int(used_memory_value)
                    if pid not in tracked_pids or used_memory < 0:
                        continue
                    process_memory[str(pid)] = process_memory.get(str(pid), 0) + used_memory
                    used_on_device += used_memory
                device_memory[uuid] = used_on_device

            result.update(
                {
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu_memory_bytes": sum(process_memory.values()),
                    "gpu_memory_by_pid_bytes": process_memory,
                    "gpu_memory_by_device_bytes": device_memory,
                }
            )
            try:
                result["torch_cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(self.device))
                result["torch_cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(self.device))
                result["torch_cuda_max_memory_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated(self.device)
                )
            except (RuntimeError, AssertionError):
                result["torch_cuda_memory_allocated_bytes"] = None
                result["torch_cuda_memory_reserved_bytes"] = None
                result["torch_cuda_max_memory_allocated_bytes"] = None
            return result

        return sample

    def start(self) -> "ResourceMonitor":
        if self._thread is not None:
            return self
        self._file = self.output_path.open("w", encoding="utf-8", buffering=1)
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()
        atexit.register(self.stop)
        LOGGER.info("Resource monitor enabled (%s); writing to %s", self.backend, self.output_path)
        return self

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                sample = self._sample() if self._sample else self._base_sample()
                self._peak.update(sample)
                if self._file is not None:
                    self._file.write(json.dumps(sample, allow_nan=False) + "\n")
            except Exception:  # monitoring must never interrupt training
                LOGGER.exception("Resource monitor sample failed")
            self._stop_event.wait(self.interval_seconds)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        if self._file is not None:
            self._file.close()
        summary = {
            "backend": self.backend,
            "root_pid": self.root_pid,
            "started_at_utc": self._started_at_utc,
            "sample_count": self._peak.samples,
            "peak_ram_rss_bytes": self._peak.ram_rss_bytes,
            "peak_ram_uss_bytes": self._peak.ram_uss_bytes or None,
            "peak_gpu_memory_bytes": self._peak.gpu_memory_bytes or None,
            "usage_log": str(self.output_path),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def start_resource_monitor(
    device: torch.device,
    output_directory: str | Path,
    interval_seconds: float = 5.0,
) -> ResourceMonitor:
    """Start the backend appropriate for ``device`` and return its controller."""
    return ResourceMonitor(
        device=device,
        output_directory=Path(output_directory),
        interval_seconds=interval_seconds,
    ).start()
