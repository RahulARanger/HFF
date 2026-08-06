"""Shared BraTS viewer data loading and HFF-Net inference helpers.

This module deliberately contains no GUI dependencies.  The web viewer and any
future research tooling can use the same preprocessing and checkpoint logic
without importing a desktop GUI toolkit or browser code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from model.HFF import HFFNet
from utils.utils import get_device


MODALITY_PRIORITY = ("t1", "t1ce", "t2", "flair")
MODEL_LOW_MODALITIES = ("flair_L", "t1_L", "t1ce_L", "t2_L")
MODEL_HIGH_MODALITIES = tuple(
    f"{modality}_H{band}"
    for modality in ("flair", "t1", "t1ce", "t2")
    for band in range(1, 5)
)
DISPLAY_MODALITIES = ("FLAIR", "T1", "T1CE", "T2")

BRATS_SEGMENTATION_COLORS = {
    1: (0.95, 0.15, 0.20, 1.0),  # necrotic / non-enhancing tumour core
    2: (0.10, 0.55, 1.00, 1.0),  # peritumoral edema
    3: (1.00, 0.85, 0.05, 1.0),  # enhancing tumour in the canonical viewer labels
}
SEGMENTATION_LABELS = {
    0: "Background",
    1: "Necrotic / non-enhancing core",
    2: "Edema",
    3: "Enhancing tumour",
}


def strip_nifti_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def is_nifti_file(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def is_frequency_file(path: Path) -> bool:
    stem = strip_nifti_suffix(path).lower()
    return stem.rsplit("_", 1)[-1] in {"l", "h", "h1", "h2", "h3", "h4"}


def modality_sort_key(path: Path) -> tuple[int, str]:
    stem = strip_nifti_suffix(path).lower()
    for index, modality in enumerate(MODALITY_PRIORITY):
        if stem.endswith(f"_{modality}"):
            return index, stem
    return 99, stem


def subject_files(subject_dir: Path) -> tuple[list[Path], Path | None]:
    files = [path for path in subject_dir.iterdir() if path.is_file() and is_nifti_file(path)]
    scans = sorted(
        [path for path in files if "_seg" not in path.name and not is_frequency_file(path)],
        key=modality_sort_key,
    )
    segmentation = next((path for path in files if "_seg" in path.name), None)
    return scans[:4], segmentation


@lru_cache(maxsize=12)
def load_volume(path: Path) -> np.ndarray:
    """Read a volume once per backend process to avoid duplicate startup I/O."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def load_subject(subject_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    scans, segmentation = subject_files(subject_dir)
    volumes = {
        strip_nifti_suffix(path).rsplit("_", 1)[-1].upper(): load_volume(path)
        for path in scans
    }
    mask = load_volume(segmentation).astype(np.uint8) if segmentation is not None else None
    return volumes, mask


def contrast_limits(volume: np.ndarray) -> tuple[float, float]:
    values = volume[np.isfinite(volume)]
    if values.size == 0:
        return 0.0, 1.0
    lower, upper = (float(value) for value in np.percentile(values, [1, 99]))
    if upper <= lower:
        return lower, lower + 1.0
    return lower, upper


def resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Resize labels with nearest-neighbour sampling for frequency data."""
    if mask.shape == target_shape:
        return mask
    if mask.ndim != len(target_shape):
        return mask

    source_indices = [
        np.minimum(
            (np.arange(target_size) * source_size / target_size).astype(np.intp),
            source_size - 1,
        )
        for source_size, target_size in zip(mask.shape, target_shape)
    ]
    return mask[np.ix_(*source_indices)]


def resize_volume_to_shape(volume: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Resample a frequency volume to the source scan display shape."""
    if volume.shape == target_shape:
        return volume

    image = sitk.GetImageFromArray(volume)
    source_size = image.GetSize()
    target_size = tuple(reversed(target_shape))
    source_spacing = image.GetSpacing()
    target_spacing = tuple(
        source_spacing[index] * source_size[index] / target_size[index]
        for index in range(len(target_size))
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return sitk.GetArrayFromImage(resampler.Execute(image)).astype(np.float32)


def suppress_background(volume: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove transform padding outside the original scan's foreground."""
    foreground = resize_mask_to_shape(reference > 0, volume.shape)
    cleaned = volume.copy()
    cleaned[~foreground] = 0
    return cleaned


def canonical_segmentation_labels(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Use 1/core, 2/edema, and 3/ET for targets and model output."""
    labels = resize_mask_to_shape(mask, target_shape).astype(np.uint8, copy=True)
    # BraTS targets encode enhancing tumour as 4; HFF-Net output uses class 3.
    labels[labels == 4] = 3
    return labels


def normalize_for_model(volume: np.ndarray) -> np.ndarray:
    """Mirror the existing loader's per-volume [-1, 1] normalization exactly."""
    minimum = float(np.min(volume))
    maximum = float(np.max(volume))
    if maximum <= minimum:
        return np.zeros_like(volume, dtype=np.float32)
    return (2.0 * ((volume - minimum) / (maximum - minimum)) - 1.0).astype(np.float32)


def restrict_mask_to_scan_foreground(mask: np.ndarray, reference_scan: np.ndarray) -> np.ndarray:
    """Remove predictions in transform padding outside the acquired MRI volume."""
    foreground = reference_scan != reference_scan.flat[0]
    cleaned = np.asarray(mask, dtype=np.uint8).copy()
    cleaned[~foreground] = 0
    return cleaned


def foreground_center_crop(
    volumes: list[np.ndarray], crop_size: int = 128
) -> tuple[list[np.ndarray], tuple[slice, slice, slice]]:
    """Reproduce the deterministic validation crop in ``loader/dataload3d.py``."""
    working = [volume[3:, :, :] for volume in volumes]
    foreground = np.zeros_like(working[0], dtype=bool)
    for volume in working:
        foreground |= volume != volume[0, 0, 0]
    coordinates = np.where(foreground)
    if coordinates[0].size == 0:
        raise ValueError("Cannot crop an empty scan foreground.")

    centre = tuple((int(axis.min()) + int(axis.max())) // 2 for axis in coordinates)
    bounds = (152, 240, 240)
    slices: list[slice] = []
    for coordinate, upper_bound in zip(centre, bounds):
        start = coordinate - crop_size // 2
        end = coordinate + crop_size // 2
        if start < 0:
            start, end = 0, crop_size
        elif end >= upper_bound:
            end, start = upper_bound - 1, upper_bound - 1 - crop_size
        slices.append(slice(start, end))
    crop = tuple(slices)
    return [volume[crop] for volume in working], crop


def find_frequency_file(scan_path: Path, band: str) -> Path | None:
    base = strip_nifti_suffix(scan_path)
    band = band.upper()
    names = [band]
    if band in {"H1", "H2", "H3", "H4"}:
        names.append("H")
    elif band == "H":
        names = ["H1", "H2", "H3", "H4", "H"]
    candidates = [
        scan_path.with_name(f"{base}_{name}{suffix}")
        for name in names
        for suffix in (".nii.gz", ".nii")
    ]
    return next((path for path in candidates if path.exists()), None)


def load_frequency_volume(
    scan_path: Path, band: str, fallback: np.ndarray
) -> tuple[np.ndarray, bool]:
    frequency_path = find_frequency_file(scan_path, band)
    if frequency_path is None:
        return np.zeros_like(fallback), False
    return resize_volume_to_shape(load_volume(frequency_path), fallback.shape), True


def discover_checkpoints(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        (path for path in results_root.rglob("*.pth") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def checkpoint_output_path(results_root: Path, checkpoint: Path, subject_dir: Path) -> Path:
    return results_root / "eval" / checkpoint.stem / f"{subject_dir.name}_OSeg.nii.gz"


def resolve_subject(dataset_root: Path, subject_id: str) -> Path:
    """Resolve a listed subject while preventing path traversal."""
    candidate = (dataset_root / subject_id).resolve()
    root = dataset_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Subject path is outside the configured dataset root.")
    if not candidate.is_dir() or not subject_files(candidate)[0]:
        raise FileNotFoundError(f"Not a scan folder: {subject_id}")
    return candidate


@lru_cache(maxsize=8)
def discover_subject_ids(dataset_root_string: str) -> tuple[str, ...]:
    dataset_root = Path(dataset_root_string).resolve()
    subject_ids = {
        scan.parent.relative_to(dataset_root).as_posix()
        for scan in dataset_root.rglob("*")
        if scan.is_file()
        and is_nifti_file(scan)
        and "_seg" not in scan.name
        and not is_frequency_file(scan)
    }
    return tuple(sorted(subject_ids))


def find_scan_path(subject_dir: Path, modality: str) -> Path:
    modality = modality.lower()
    scans, _ = subject_files(subject_dir)
    for path in scans:
        if strip_nifti_suffix(path).rsplit("_", 1)[-1].lower() == modality:
            return path
    raise FileNotFoundError(f"Missing {modality} scan in {subject_dir}")


def load_display_volume(subject_dir: Path, modality: str) -> tuple[np.ndarray, bool]:
    """Load an actual, low-frequency, or high-frequency display volume."""
    actual_path = find_scan_path(subject_dir, modality)
    actual = load_volume(actual_path)
    if modality.upper() in DISPLAY_MODALITIES:
        return actual, True
    raise ValueError(f"Unsupported display modality: {modality}")


def generate_output_segmentation(
    checkpoint: Path, subject_dir: Path, results_root: Path
) -> Path:
    """Run the selected HFF-Net checkpoint with the existing validation transform."""
    modal_paths = {
        strip_nifti_suffix(path).rsplit("_", 1)[-1].lower(): path
        for path in subject_files(subject_dir)[0]
    }
    required = [*MODEL_LOW_MODALITIES, *MODEL_HIGH_MODALITIES]
    volumes: list[np.ndarray] = []
    for modality in required:
        base, band = modality.rsplit("_", 1)
        path = find_frequency_file(modal_paths[base], band)
        if path is None:
            raise FileNotFoundError(
                f"Missing required frequency volume for {modality} in {subject_dir}"
            )
        volumes.append(load_volume(path))

    cropped_volumes, crop = foreground_center_crop(volumes)
    inputs = [
        torch.from_numpy(normalize_for_model(volume)).unsqueeze(0).unsqueeze(0)
        for volume in cropped_volumes
    ]
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    num_classes = int(state_dict["l1_b1_f.weight"].shape[0])
    device = get_device()
    model = HFFNet(4, 16, num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        low = torch.cat(inputs[:4], dim=1).to(device)
        high = torch.cat(inputs[4:], dim=1).to(device)
        output_low, output_high, _, _ = model(low, high)
        output = output_low if "result1" in checkpoint.stem.lower() else output_high
        prediction = (
            torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        )

    target_crop_shape = tuple(axis_slice.stop - axis_slice.start for axis_slice in crop)
    prediction = resize_mask_to_shape(prediction, target_crop_shape).astype(np.uint8)

    segmentation_path = subject_files(subject_dir)[1]
    reference_path = segmentation_path or modal_paths["flair"]
    reference_image = sitk.ReadImage(str(reference_path))
    reference_volume = sitk.GetArrayFromImage(reference_image).astype(np.float32)
    flair_volume = load_volume(modal_paths["flair"])
    if flair_volume.shape != reference_volume.shape:
        flair_volume = resize_volume_to_shape(flair_volume, reference_volume.shape)
    restored = np.zeros(tuple(reversed(reference_image.GetSize())), dtype=np.uint8)
    restored_crop = restored[3:, :, :]
    restored_crop[crop] = prediction
    restored = restrict_mask_to_scan_foreground(restored, flair_volume)

    output_path = checkpoint_output_path(results_root, checkpoint, subject_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = sitk.GetImageFromArray(restored)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, str(output_path))
    return output_path
