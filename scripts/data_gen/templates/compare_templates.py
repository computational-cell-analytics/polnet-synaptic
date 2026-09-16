"""
Compares physical size/shape between an old (PDB-derived) and new (EMD-derived) template
for the same molecule, since the two can't be compared voxel-by-voxel
    Input:
        - Matching lists of old and new .pns paths (relative to their own root)
    Output:
        - Printed bounding-box size (A) and occupied-voxel fraction for each pair
        - Optional side-by-side central-slice PNG per pair
"""

__author__ = "Sage Martineau"

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from polnet import lio
from polnet.stomo import MmerFile
from polnet.utils.utils import tomo_crop_non_zeros


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_root", type=str, required=True, help="Root path for the old .pns/.mrc pair.")
    parser.add_argument("--new_root", type=str, required=True, help="Root path for the new .pns/.mrc pair.")
    parser.add_argument("--old_pns", type=str, nargs="+", required=True,
                        help="Old .pns paths, relative to --old_root.")
    parser.add_argument("--new_pns", type=str, nargs="+", required=True,
                        help="New .pns paths, relative to --new_root, same order as --old_pns.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="If given, save a side-by-side central-slice PNG per pair here.")
    return parser.parse_args()


def load_template(root, pns_rel_path):
    protein = MmerFile(root + "/" + pns_rel_path)
    mrc_path = root + "/" + protein.get_mmer_svol()
    tomo = lio.load_mrc(mrc_path)
    v_size = lio.read_mrc_v_size(mrc_path)[0]
    return protein, tomo, v_size


def report(name, protein, tomo, v_size):
    mask = tomo >= protein.get_iso()
    occupied_frac = mask.mean()
    cropped = tomo_crop_non_zeros(mask.astype(np.float32))
    box_size_a = np.asarray(cropped.shape) * v_size
    print(f"{name}: box size (A) = {tuple(box_size_a.round(1))}, occupied fraction = {occupied_frac:.4f}")


def save_slice_comparison(out_path, old_tomo, new_tomo):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(old_tomo[old_tomo.shape[0] // 2], cmap="gray")
    axes[0].set_title("old (PDB)")
    axes[1].imshow(new_tomo[new_tomo.shape[0] // 2], cmap="gray")
    axes[1].set_title("new (EMD)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    if len(args.old_pns) != len(args.new_pns):
        raise ValueError("--old_pns and --new_pns must have the same length")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    for old_rel, new_rel in zip(args.old_pns, args.new_pns):
        old_protein, old_tomo, old_v_size = load_template(args.old_root, old_rel)
        new_protein, new_tomo, new_v_size = load_template(args.new_root, new_rel)

        print(f"--- {old_protein.get_mmer_id()} vs {new_protein.get_mmer_id()} ---")
        report("old", old_protein, old_tomo, old_v_size)
        report("new", new_protein, new_tomo, new_v_size)

        if args.out_dir:
            out_path = os.path.join(args.out_dir, f"{new_protein.get_mmer_id()}.png")
            save_slice_comparison(out_path, old_tomo, new_tomo)
            print("Wrote", out_path)


if __name__ == "__main__":
    main()
