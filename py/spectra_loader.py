"""
Loader for DESI spectra stored as FITS files.

Each file contains:
  wave     (7781,)    – shared wavelength grid [Å], float64 → float32
  flux     (N, 7781)  – spectral flux, float64 → float32
  flux_ivar(N, 7781)  – inverse variance of flux, float64 → float32
  cat               – binary table with per-spectrum metadata

Per-spectrum fields returned:
  z            – redshift (float32)
  zerr         – redshift uncertainty (float32), joined from external catalog
  desi_target  – target selection bitmask (int64)

zerr is joined from an external catalog keyed on targetid.  Pass the path
via ``zerr_catalog``; if omitted, zerr is not returned.

Usage example::

    from spectra_loader import SpectraDataset
    import torch

    ds = SpectraDataset(
        "spectra/",
        zerr_catalog="metadata/desi-galaxy-cat-zerr.fits",
        n_spectra=1000,
        desi_target_mask=1,
    )

    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=4)
    wave = ds.wave  # (7781,) float32 tensor, same for every spectrum

    for batch in loader:
        flux   = batch["flux"]         # (B, 7781) float32
        ivar   = batch["ivar"]         # (B, 7781) float32
        z      = batch["z"]            # (B,)      float32
        zerr   = batch["zerr"]         # (B,)      float32
        target = batch["desi_target"]  # (B,)      int64
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import fitsio
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectraDataset(Dataset):
    """PyTorch Dataset for DESI FITS spectra.

    Parameters
    ----------
    path:
        Directory containing ``spec-*.fits`` files, or an explicit list of
        file paths.
    zerr_catalog:
        Path to a FITS file containing ``targetid`` and ``zerr`` columns.
        When provided, zerr is joined to each spectrum on targetid and
        returned in every item.  Spectra whose targetid is absent from the
        catalog are dropped.
    n_spectra:
        If given, cap the dataset at this many spectra (taken from the
        beginning of the optionally filtered index).
    zmin:
        Drop spectra with redshift below this value.
    zmax:
        Drop spectra with redshift above this value.
    desi_target_mask:
        Integer bitmask.  Only spectra where ``desi_target & mask != 0`` are
        kept.  Pass ``None`` (default) to keep everything.
    """

    def __init__(
        self,
        path: str | Sequence[str],
        *,
        zerr_catalog: Optional[str] = None,
        n_spectra: Optional[int] = None,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        desi_target_mask: Optional[int] = None,
    ) -> None:
        files = _resolve_files(path)
        if not files:
            raise FileNotFoundError(f"No spec-*.fits files found in {path!r}")

        # Optionally load zerr lookup: targetid -> zerr
        zerr_map: Optional[dict[int, np.float32]] = None
        if zerr_catalog is not None:
            zerr_map = _load_zerr_map(zerr_catalog)

        # Read catalog from every file upfront (catalog rows are small).
        # index entries: (file_path, local_row_index, zerr_or_nan)
        index: list[tuple[str, int, np.float32]] = []
        for fpath in files:
            with fitsio.FITS(fpath) as f:
                cat = f["cat"].read(columns=["targetid", "z", "desi_target"])
            n = len(cat)
            keep = np.ones(n, dtype=bool)
            if desi_target_mask is not None:
                keep &= (cat["desi_target"] & desi_target_mask) != 0
            if zmin is not None:
                keep &= cat["z"] >= zmin
            if zmax is not None:
                keep &= cat["z"] <= zmax
            if zerr_map is not None:
                keep &= np.array(
                    [int(tid) in zerr_map for tid in cat["targetid"]], dtype=bool
                )
            for r in np.where(keep)[0]:
                tid = int(cat["targetid"][r])
                zerr = zerr_map[tid] if zerr_map is not None else np.float32(np.nan)
                index.append((fpath, int(r), zerr))

        if n_spectra is not None:
            index = index[:n_spectra]

        self._index = index
        self._has_zerr = zerr_map is not None

        # Load the wavelength grid from the first file (identical across files).
        with fitsio.FITS(files[0]) as f:
            wave = f["wave"].read().astype(np.float32)
        self._wave = torch.from_numpy(wave)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def wave(self) -> torch.Tensor:
        """Shared wavelength array, shape (7781,), float32."""
        return self._wave

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fpath, row, zerr = self._index[idx]
        with fitsio.FITS(fpath) as f:
            flux = f["flux"][row : row + 1, :][0].astype(np.float32)
            ivar = f["flux_ivar"][row : row + 1, :][0].astype(np.float32)
            cat_row = f["cat"][row : row + 1]

        z = np.float32(cat_row["z"][0])
        desi_target = int(cat_row["desi_target"][0])

        item: dict[str, torch.Tensor] = {
            "flux": torch.from_numpy(flux),
            "ivar": torch.from_numpy(ivar),
            "z": torch.tensor(z, dtype=torch.float32),
            "desi_target": torch.tensor(desi_target, dtype=torch.int64),
        }
        if self._has_zerr:
            item["zerr"] = torch.tensor(zerr, dtype=torch.float32)
        return item


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_zerr_map(catalog_path: str) -> dict[int, np.float32]:
    """Return {targetid: zerr} from a FITS catalog."""
    with fitsio.FITS(catalog_path) as f:
        cat = f[1].read(columns=["targetid", "zerr"])
    return {int(tid): np.float32(ze) for tid, ze in zip(cat["targetid"], cat["zerr"])}


def _resolve_files(path: str | Sequence[str]) -> list[str]:
    if isinstance(path, str):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "spec-*.fits")))
        else:
            files = sorted(glob.glob(path))
    else:
        files = sorted(path)
    return files
