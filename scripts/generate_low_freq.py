#!/usr/bin/env python3

from pathlib import Path
import argparse

import dtcwt
import nibabel as nib
import numpy as np


def process_nii_lowpass(nii_path: Path, out_path: Path) -> None:
    """
    Perform DTCWT transform on the entire NIfTI file, extract only the
    low-frequency component, and save it as a new .nii.gz file.
    """
    nii = nib.load(str(nii_path))
    data = nii.get_fdata()  # Assume data is 3D with shape (height, width, slices)
    print(f"  Original data shape: {data.shape}")

    transform = dtcwt.Transform2d()
    lowpass_slices = []

    # Keep the existing processing behavior. The Python entrypoint does not
    # expose nlevels; it always uses the original fixed value of 1.
    for i in range(data.shape[2]):
        slice_img = data[:, :, i]
        transformed = transform.forward(slice_img, nlevels=1)
        lowpass = transformed.lowpass
        lowpass_norm = (lowpass - lowpass.min()) / (lowpass.max() - lowpass.min()) * 255
        lowpass_slices.append(lowpass_norm.astype(np.uint8))

    lowpass_data = np.stack(lowpass_slices, axis=2)
    print(f"  Low-frequency data shape: {lowpass_data.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(lowpass_data, affine=nii.affine), str(out_path))
    print(f"  Low-frequency result saved to: {out_path}")


def is_input_modality(nii_path: Path) -> bool:
    name = nii_path.name.lower()
    return (
        nii_path.name.endswith(".nii")
        or nii_path.name.endswith(".nii.gz")
    ) and "seg" not in name and "_h" not in name and "_l" not in name


def base_name(nii_path: Path) -> str:
    if nii_path.name.endswith(".nii.gz"):
        return nii_path.name[:-7]
    return nii_path.stem


def iter_input_files(root: Path):
    for nii_path in sorted(root.rglob("*.nii")) + sorted(root.rglob("*.nii.gz")):
        if is_input_modality(nii_path):
            yield nii_path


def iter_subject_files(subject_dir: Path):
    for nii_path in sorted(subject_dir.glob("*.nii")) + sorted(subject_dir.glob("*.nii.gz")):
        if is_input_modality(nii_path):
            yield nii_path


def find_split_dirs(input_root: Path) -> list[Path]:
    """Find train/validation/testing folders below the supplied root."""
    split_names = {"train", "validation", "testing"}
    direct_split_dirs = [
        child
        for child in sorted(input_root.iterdir())
        if child.is_dir() and child.name.lower() in split_names
    ]
    if direct_split_dirs:
        return direct_split_dirs

    return sorted(
        candidate
        for candidate in input_root.rglob("*")
        if candidate.is_dir() and candidate.name.lower() in split_names
    )


def iter_subject_dirs(split_dir: Path):
    for subject_dir in sorted(split_dir.iterdir()):
        if subject_dir.is_dir():
            yield subject_dir


def main(input_path: str, output_dir: str = "") -> None:
    input_root = Path(input_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve() if output_dir else None
    split_dirs = find_split_dirs(input_root)

    if split_dirs:
        for split_dir in split_dirs:
            relative_split_dir = split_dir.relative_to(input_root)
            for subject_dir in iter_subject_dirs(split_dir):
                subject_output = (
                    subject_dir
                    if output_root is None
                    else output_root / relative_split_dir / subject_dir.name
                )
                for nii_path in iter_subject_files(subject_dir):
                    process_nii_lowpass(
                        nii_path,
                        subject_output / f"{base_name(nii_path)}_L.nii.gz",
                    )
        return

    for nii_path in iter_input_files(input_root):
        if output_root is None:
            out_path = nii_path.parent / f"{base_name(nii_path)}_L.nii.gz"
        else:
            relative_parent = nii_path.parent.relative_to(input_root)
            out_path = output_root / relative_parent / f"{base_name(nii_path)}_L.nii.gz"
        process_nii_lowpass(nii_path, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract low-frequency features from NIfTI files using DTCWT"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Root folder containing train, validation, and testing folders.",
    )
    parser.add_argument("--output-dir", default="", help="Optional output root for low-frequency volumes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.path).expanduser().is_dir():
        raise SystemExit(f"Error: input path not found: {args.path}")

    main(args.path, args.output_dir)
