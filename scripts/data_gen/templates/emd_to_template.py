"""
Builds a polnet template (.mrc + .pns) from an EMDB density map
    Input:
        - EMD accession number
        - Placement parameters (MMER_ISO, PMER_L, PMER_L_MAX, PMER_OCC, PMER_OVER_TOL)
    Output:
        - The resampled, low-pass filtered, normalized density map (.mrc)
        - The matching .pns file for use in --proteins_list
"""

__author__ = "Sage Martineau"

import os
import gzip
import shutil
import tempfile
import argparse

import numpy as np

import wget
import scipy.ndimage
from skimage.filters import threshold_otsu

from polnet import lio
from polnet.utils.utils import bin_volume, lowpass_filter

SIGMA_FACTOR = 0.187  # molmap-style resolution-to-sigma constant, matches gui/core/pdbtomrc.py
CLIP_PERCENTILE_HIGH = 99.9  # EM map background sits near raw 0; normalize() clips negatives
BG_MARGIN_VOXELS = 5  # dilation past the structure mask, zero solvent outside of the dilated mask

REPO_ROOT = os.path.dirname(os.path.realpath(__file__)) + "/../../.."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emd_id", type=str, required=True, help="EMD accession number, e.g. 10001.")
    parser.add_argument("--label", type=str, default=None, help="Free-text description for the .pns comment.")
    parser.add_argument("--out_root", type=str, default=os.path.realpath(REPO_ROOT + "/data/czii"),
                        help="Dataset root to write output files.")
    parser.add_argument("--voxel_size", type=float, default=10.0, help="Target voxel size in Angstrom.")
    parser.add_argument("--lowpass", type=float, default=30.0,
                        help="Low-pass target resolution in Angstrom, to match a tomogram's effective resolution.")
    parser.add_argument("--pad_voxels", type=int, default=10,
                        help="Zero-padding added on every side after resampling.")
    parser.add_argument("--mmer_iso", type=float, default=None, 
                        help="MMER_ISO isosurface threshold. By default, compute using Otsu's method.")
    parser.add_argument("--pmer_l", type=float, default=1.2, help="PMER_L polymer length parameter.")
    parser.add_argument("--pmer_l_max", type=float, default=1.0, help="PMER_L_MAX max polymer length parameter.")
    parser.add_argument("--pmer_occ", type=float, required=True, help="PMER_OCC target occupancy.")
    parser.add_argument("--pmer_over_tol", type=float, default=0.001, help="PMER_OVER_TOL overlap tolerance.")
    return parser.parse_args()


def download_map(emd_id, out_root):
    url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd_id}/map/emd_{emd_id}.map.gz"
    dest_dir = os.path.join(out_root, "templates", "emds")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"emd_{emd_id}.map.gz")
    if os.path.exists(dest_path):
        print("Already downloaded:", dest_path)
        return dest_path
    print("Downloading", url, "to", dest_path)
    wget.download(url, dest_path)
    print()
    return dest_path


def decompress_map(gz_path):
    fd, map_path = tempfile.mkstemp(suffix=".mrc")
    os.close(fd)
    with gzip.open(gz_path, "rb") as f_in, open(map_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return map_path


def pad_margin(tomo, pad_voxels):
    return np.pad(tomo, pad_voxels, mode="constant", constant_values=0)


def normalize(tomo):
    tomo = np.clip(tomo, 0, None)
    hi = np.percentile(tomo, CLIP_PERCENTILE_HIGH)
    tomo = np.clip(tomo, 0, hi)
    return tomo / hi


def build_template(map_path, voxel_size, lowpass, pad_voxels):
    tomo = lio.load_mrc(map_path)
    native_v_size = lio.read_mrc_v_size(map_path)[0]
    tomo, out_v_size = bin_volume(tomo, native_v_size, voxel_size / native_v_size)
    tomo = pad_margin(tomo, pad_voxels)
    tomo = lowpass_filter(tomo, out_v_size, lowpass, sigma_factor=SIGMA_FACTOR)
    tomo = normalize(tomo)
    return tomo, out_v_size


def compute_iso(tomo, override):
    return override if override is not None else float(threshold_otsu(tomo))


def zero_solvent(tomo, iso):
    labeled, n = scipy.ndimage.label(tomo >= iso)
    if n == 0:
        return tomo
    sizes = scipy.ndimage.sum(tomo >= iso, labeled, range(1, n + 1))
    largest_label = np.argmax(sizes) + 1
    structure = scipy.ndimage.binary_dilation(labeled == largest_label, iterations=BG_MARGIN_VOXELS)
    tomo = tomo.copy()
    tomo[~structure] = 0
    return tomo


def write_pns(pns_path, mmer_id, mmer_svol, mmer_iso, label, args):
    id_line = f"MMER_ID = {mmer_id}"
    if label:
        id_line += f"  # {label}"
    with open(pns_path, "w") as f:
        f.write(id_line + "\n")
        f.write(f"MMER_SVOL = {mmer_svol}\n")
        f.write(f"MMER_ISO = {mmer_iso}\n")
        f.write(f"PMER_L = {args.pmer_l}\n")
        f.write(f"PMER_L_MAX = {args.pmer_l_max}\n")
        f.write(f"PMER_OCC = {args.pmer_occ}\n")
        f.write(f"PMER_OVER_TOL = {args.pmer_over_tol}\n")


def generate_template(args):
    n = int(args.voxel_size)
    mrc_dir = os.path.join(args.out_root, "templates", f"mrcs_{n}A")
    pns_dir = os.path.join(args.out_root, f"in_{n}A")
    os.makedirs(mrc_dir, exist_ok=True)
    os.makedirs(pns_dir, exist_ok=True)

    gz_path = download_map(args.emd_id, args.out_root)
    map_path = decompress_map(gz_path)
    try:
        tomo, out_v_size = build_template(map_path, args.voxel_size, args.lowpass, args.pad_voxels)
    finally:
        os.remove(map_path)

    mmer_iso = compute_iso(tomo, args.mmer_iso)
    tomo = zero_solvent(tomo, mmer_iso)

    mmer_id = f"emd_{args.emd_id}"
    mrc_path = os.path.join(mrc_dir, f"emd_{args.emd_id}.mrc")
    lio.write_mrc(tomo, mrc_path, v_size=out_v_size)

    pns_path = os.path.join(pns_dir, f"emd_{args.emd_id}_{n}A.pns")
    mmer_svol = f"/templates/mrcs_{n}A/emd_{args.emd_id}.mrc"
    write_pns(pns_path, mmer_id, mmer_svol, mmer_iso, args.label, args)

    print("Wrote", mrc_path)
    print("Wrote", pns_path)


def main():
    args = parse_args()
    generate_template(args)


if __name__ == "__main__":
    main()
