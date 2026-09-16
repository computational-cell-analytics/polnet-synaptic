# EMD template pipeline — status

Branch: `emd-templates`. Builds polnet templates (`.mrc` + `.pns`) from real EMDB density maps,
as an alternative to the existing PDB-atomic-model pipeline (`gui/core/pdbtomrc.py`).

## Files

- `emd_to_template.py` — generates one template from one EMD accession number.
- `compare_templates.py` — diagnostic: compares an old (PDB) and new (EMD) template's physical
  box size and occupied-voxel fraction, since they can't be compared voxel-by-voxel.
- `polnet/utils/utils.py: bin_volume()` — voxel-size resampling, ported from
  `filament-analysis-toolbox/src/utils/volume.py` (fixed a read-only-array bug in the source).
- `tests/test_emd_to_template.py` — 12 unit tests, no network calls, all passing.

## Pipeline (`emd_to_template.py`)

1. Download `emd_{id}.map.gz` from the EBI FTP, cached under `{out_root}/templates/emds/` (skips
   re-download if already present).
2. Decompress, load, resample to `--voxel_size` (default 10 A) with `bin_volume()`.
3. `pad_margin()` — zero-pad by `--pad_voxels` (default 10) so the structure never touches the box
   edge. EMDB depositors box maps at whatever size they choose; some are tight.
4. `low_pass()` — Gaussian blur, `sigma = 0.187 * (lowpass / voxel_size)`, same molmap-style
   formula as the PDB pipeline. `--lowpass` defaults to 30 A, matching the PDB pipeline's real
   default (`gui/core/widgets_utilities.py`: `apix=10, res=30`) — this degrades the input to a
   *simulated tomogram's* effective resolving power, independent of the input's native resolution.
5. `normalize()` — clip negative values to 0 (EM map background sits near raw 0; a real map can go
   negative from sharpening, unlike the PDB-Gaussian densities which are never negative), scale by
   the 99.9th percentile of the remaining positive signal.
6. `compute_iso()` — `MMER_ISO` computed per-map with Otsu's method (`skimage.filters.
   threshold_otsu`), not a fixed value. The old pipeline never used a fixed threshold either — a
   human picked it per structure in an interactive VTK viewer (`gui/core/vtk_utilities.py:
   select_isosurface`); `0.1` only worked as a fixed default because the synthetic densities are
   consistent by construction. Real EMDB maps vary too much in noise/SNR for one fixed value
   (confirmed empirically: a fixed threshold gave a wildly oversized albumin template before this
   fix). `--mmer_iso` is still available as a manual override.
7. `zero_solvent()` — hard-zero everything outside the largest connected component (dilated by
   `BACKGROUND_MARGIN_VOXELS = 5`) at the computed iso. `normalize()` alone only clips negative
   values, so small positive solvent noise survives and is visible in the final map; this also
   matters at simulation time since `all_features_default.py` uses the raw density values (not
   just the `MMER_ISO` mask) when compositing the tomogram.
8. Write `.mrc` and `.pns` under `{out_root}/templates/mrcs_{n}A/` and `{out_root}/in_{n}A/`.

No cropping to a tight bounding box — `all_features_default.py` already calls `vol_cube()` at load
time, which centers and pads any shape into a cube.

## Data state

All 7 CZII proteins generated under `polnet-synaptic/data/czii/`, using the EMDB accessions from
Table 1 of the phantom-dataset paper (*A realistic phantom dataset for benchmarking cryo-ET data
annotation*, Nature Methods 2025, PMC12446061) — these are the real picking-template maps, not a
web-search-hallucinated set of IDs that surfaced early on (ignore any `EMD-736xx`-looking numbers).

| protein | old PDB id | new EMD id |
|---|---|---|
| beta-galactosidase | `6drv` | `0153` |
| VLP (PP7 bacteriophage) | `1dwn-cargo` | `41917` |
| ribosome (80S, human) | `6qzp` | `3883` |
| thyroglobulin (bovine) | `7n4y` | `24181` |
| apoferritin (equine) | `8cpv` | `41923` |
| albumin (HSA) | `8vaf` | `43090` |
| beta-amylase (sweet potato) | `1fa2-tetramer` | `30405` |

`data/czii/` also has copies of all 7 old PDB-derived `.pns`/`.mrc` pairs (byte-identical to
`data/default/`, kept under their original filenames — no collision with the `emd_*` names), so
both sets are available from one root.

Verified with `compare_templates.py` and a visual render (`workspaces/render_betagal.py` on the
shared filesystem, not in the repo) — all 7 pairs land in the same order of magnitude on box size
and occupied fraction, no outliers, clean solvent background.

## Downstream: czii_dataset_3

Per the user's decision, only VLP has been switched over in the actual dataset — every other
protein stays on its PDB-derived template for now:

- `data/simulation/czii_dataset_3/configs/czii_c0.toml` and `czii_c1.toml`: added
  `root_path = ".../polnet-synaptic/data/czii"`, changed only the VLP `proteins_list` entry to
  `in_10A/emd_41917_10A.pns` (same list index, so `pmer_occ_list` alignment is untouched).
- `faket-polnet/faket_polnet/utils/label_transform.py`: added
  `"emd_41917_10A": "virus-like-particle"` to the hardcoded filename-stem-to-label `mapping` dict
  (kept the old `"1dwn-cargo_10A"` entry too, for reprocessing older runs).

## Open items / not done

- The other 6 proteins have real EMD-derived templates sitting ready in `data/czii/` but are not
  wired into `czii_dataset_3` — that was an explicit choice, not an oversight.
- `all_features_default.py` itself is unchanged; the new templates work through the existing
  `--proteins_list`/`MmerFile` mechanism with no simulator changes needed.
- Diagnostic/investigation scripts from this session (`diameter_compare.py`, `render_betagal.py`,
  `investigate_betagal_*.py`, `generate_emd_templates.sh`) live under
  `/mnt/vast-nhr/projects/nim00020/sage/workspaces/`, not in the repo — scratch, not committed.
