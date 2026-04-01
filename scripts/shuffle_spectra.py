#!/usr/bin/env python3
"""
Shuffle DESI spectra randomly across spec-*.fits files.

After shuffling every output file contains the same number of spectra as
the corresponding input file, but drawn at random from the full dataset.
All metadata (catalog rows, targetid, z, desi_target, ...) is shuffled
consistently so that cat[i] always matches flux[i] and flux_ivar[i].

Memory strategy
---------------
* All catalog data (a few hundred MB) is loaded into RAM once and shuffled
  with a simple index permutation.
* flux/ivar (~50 GB total for 842k spectra) are never all in RAM.  They are
  scattered via numpy memmaps, processed in batches of output files so that
  at most --batch-size file-worth of memmaps are live at once.
* Staging memmaps live in a temporary directory and are deleted after the
  output FITS files are written (unless --keep-staging is given).

Disk: approximately 2× the source data while the script is running (memmaps
+ output FITS).  Memmaps are cleaned up at the end by default.

I/O: each source file is read once per batch, so total source reads =
n_batches × n_files.  With --batch-size 200 and 1685 files this is ~9 passes
(~450 GB of I/O).  Adjust --batch-size upwards to reduce passes at the cost
of more simultaneous open memmap files.

Usage
-----
# defaults (reads spectra/, writes spectra_shuffled/)
python scripts/shuffle_spectra.py

# custom paths / seed
python scripts/shuffle_spectra.py \\
    --spectra-dir /data/spectra/ \\
    --out-dir /data/spectra_shuffled/ \\
    --seed 7

# fewer passes (more memmaps open simultaneously)
python scripts/shuffle_spectra.py --batch-size 500
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import sys
import tempfile
import time

import fitsio
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Shuffle DESI spectra across FITS files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--spectra-dir', default='spectra/',
                   help='Source directory containing spec-*.fits files')
    p.add_argument('--out-dir', default='spectra_shuffled/',
                   help='Output directory for shuffled files')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--batch-size', type=int, default=200,
                   help='Output files processed per pass through source files. '
                        'Lower = fewer simultaneous memmaps; more passes.')
    p.add_argument('--staging-dir', default=None,
                   help='Where to put temporary memmaps.  '
                        'Default: a temp subdir of --out-dir, auto-deleted.')
    p.add_argument('--keep-staging', action='store_true',
                   help='Keep staging memmaps after writing (useful for debugging).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scan_files(spectra_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(spectra_dir, 'spec-*.fits')))
    if not paths:
        sys.exit(f'ERROR: no spec-*.fits files found in {spectra_dir!r}')
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Scan source files: collect sizes, catalog rows, and wave grid.
    # ------------------------------------------------------------------
    paths = scan_files(args.spectra_dir)
    n_files = len(paths)
    print(f'Found {n_files} source files in {args.spectra_dir!r}')

    print('Scanning file sizes and loading all catalog rows...')
    sizes: list[int] = []
    all_cats: list[np.ndarray] = []
    wave: np.ndarray | None = None

    for i, path in enumerate(paths):
        with fitsio.FITS(path) as f:
            sizes.append(f['flux'].read_header()['NAXIS2'])
            all_cats.append(f['cat'].read())
            if wave is None:
                wave = f['wave'].read()
        if (i + 1) % 200 == 0 or (i + 1) == n_files:
            print(f'  {i + 1}/{n_files} files scanned')

    assert wave is not None
    sizes_arr = np.array(sizes, dtype=np.int64)
    cum = np.concatenate([[0], np.cumsum(sizes_arr)])  # (n_files+1,)
    N = int(cum[-1])
    Nf = int(wave.shape[0])

    all_cat = np.concatenate(all_cats)   # structured array, shape (N,)
    del all_cats
    print(f'  Total spectra: {N:,}   wavelength points: {Nf}   '
          f'catalog: {all_cat.nbytes / 1e6:.0f} MB')

    # ------------------------------------------------------------------
    # 2. Build a global random permutation.
    #
    #    perm[dst_global] = src_global
    #      — position dst_global in the output gets data from src_global.
    #
    #    inv_perm[src_global] = dst_global
    #      — used during the scatter phase to route each source row.
    # ------------------------------------------------------------------
    print('Building permutation...')
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)                          # (N,) int64

    inv_perm = np.empty(N, dtype=np.int64)
    inv_perm[perm] = np.arange(N, dtype=np.int64)

    # For every src_global index, pre-compute its output (dst_file, dst_row).
    dst_global_of = inv_perm                           # (N,) — indexed by src_global
    dst_file_of = np.searchsorted(
        cum[1:], dst_global_of, side='right'
    ).astype(np.int32)                                 # (N,)
    dst_row_of = (dst_global_of - cum[dst_file_of]).astype(np.int32)  # (N,)
    del inv_perm, dst_global_of

    # Shuffle catalog: output position dst_global gets all_cat[perm[dst_global]].
    print('Shuffling catalog...')
    shuffled_cat = all_cat[perm]    # (N,) structured array
    del all_cat, perm
    print('  Done.')

    # ------------------------------------------------------------------
    # 3. Staging directory for temporary memmaps.
    # ------------------------------------------------------------------
    staging_dir = args.staging_dir
    staging_is_temp = staging_dir is None
    if staging_is_temp:
        staging_dir = tempfile.mkdtemp(dir=args.out_dir, prefix='_shuffle_staging_')
    else:
        os.makedirs(staging_dir, exist_ok=True)
    print(f'Staging directory: {staging_dir}')

    # ------------------------------------------------------------------
    # 4. Scatter + gather in batches of output files.
    #
    #    For each batch:
    #      a) Allocate memmaps for the batch's output files.
    #      b) Scan all source files; for rows destined for this batch,
    #         write them directly to the correct memmap position.
    #      c) Write each output FITS file from its memmap + catalog slice.
    #      d) Release memmaps.
    # ------------------------------------------------------------------
    n_batches = math.ceil(n_files / args.batch_size)
    print(f'\nScattering in {n_batches} batch(es) of ≤{args.batch_size} output files '
          f'({n_batches} pass(es) through source files).')

    for batch_idx in range(n_batches):
        b_start = batch_idx * args.batch_size
        b_end   = min(b_start + args.batch_size, n_files)
        t_batch = time.time()
        print(f'\n--- Batch {batch_idx + 1}/{n_batches}  '
              f'(output files {b_start}–{b_end - 1}) ---')

        # Allocate memmaps.
        flux_mms: dict[int, np.memmap] = {}
        ivar_mms: dict[int, np.memmap] = {}
        for j in range(b_start, b_end):
            nj = int(sizes_arr[j])
            flux_mms[j] = np.memmap(
                os.path.join(staging_dir, f'flux_{j:05d}.mm'),
                dtype='float32', mode='w+', shape=(nj, Nf),
            )
            ivar_mms[j] = np.memmap(
                os.path.join(staging_dir, f'ivar_{j:05d}.mm'),
                dtype='float32', mode='w+', shape=(nj, Nf),
            )

        # Scatter: scan every source file.
        n_written = 0
        for i, src_path in enumerate(paths):
            src_start = int(cum[i])
            src_end   = int(cum[i + 1])
            n_src     = src_end - src_start
            src_range = np.arange(src_start, src_end)  # src_global indices for this file

            dfs = dst_file_of[src_range]                # (n_src,) which output file
            in_batch = (dfs >= b_start) & (dfs < b_end)
            if not in_batch.any():
                continue

            with fitsio.FITS(src_path) as f:
                flux_src = f['flux'].read().astype(np.float32)      # (n_src, Nf)
                ivar_src = f['flux_ivar'].read().astype(np.float32) # (n_src, Nf)

            drs = dst_row_of[src_range]   # (n_src,) which row within output file

            # Group by destination file for contiguous writes.
            for df in np.unique(dfs[in_batch]):
                sel = in_batch & (dfs == df)
                flux_mms[df][drs[sel]] = flux_src[sel]
                ivar_mms[df][drs[sel]] = ivar_src[sel]
                n_written += int(sel.sum())

            if (i + 1) % 200 == 0 or (i + 1) == n_files:
                print(f'  scatter: {i + 1}/{n_files} source files  '
                      f'({n_written:,} rows routed)')

        # Gather: write output FITS.
        print('  Writing FITS...')
        for j in range(b_start, b_end):
            flux_mms[j].flush()
            ivar_mms[j].flush()

            cat_j = shuffled_cat[int(cum[j]):int(cum[j + 1])]
            out_path = os.path.join(args.out_dir, os.path.basename(paths[j]))

            with fitsio.FITS(out_path, 'rw', clobber=True) as fout:
                fout.write(wave,           extname='wave')
                fout.write(flux_mms[j][:], extname='flux')
                fout.write(ivar_mms[j][:], extname='flux_ivar')
                fout.write(cat_j,          extname='cat')

        del flux_mms, ivar_mms   # close + unmap

        elapsed = time.time() - t_batch
        print(f'  Batch done in {elapsed:.0f}s  '
              f'({b_end - b_start} files written)')

    # ------------------------------------------------------------------
    # 5. Cleanup.
    # ------------------------------------------------------------------
    if staging_is_temp and not args.keep_staging:
        print(f'\nRemoving staging directory ...')
        shutil.rmtree(staging_dir)

    total = time.time() - t_start
    print(f'\nDone.  {n_files} shuffled files written to {args.out_dir!r}  '
          f'(total {total:.0f}s)')


if __name__ == '__main__':
    main()
