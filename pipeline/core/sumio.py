"""
Readers for the HYPOINVERSE H71 summary (.sum) and HypoDD (.reloc / event.dat)
catalogs. The `.sum` format is byte-identical to the Ulsan pipeline's, so this
mirrors that project's `catalog_summary` loader: read the comma-separated H71
summary, strip the space-padded headers, and build a UTC `time` column.
"""
from __future__ import annotations

import pandas as pd
from obspy import UTCDateTime


# ------------------------------------------------------------------ .sum (HYPOINVERSE)
def read_sum(path) -> pd.DataFrame:
    """Parse a HYPOINVERSE H71 `.sum` into a tidy DataFrame.

    Columns: id, time (UTCDateTime), lat, lon, depth, num, gap, rms, erh, erz, qual.
    The raw header is comma-separated and space-padded, e.g.
        `   DATE     TIME, SEC ,   LAT  ,   LON   , DEPTH,...,QASR,    ID-NUM,...`
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    datecol = next(c for c in df.columns if c.startswith("DATE"))  # "DATE     TIME"

    def _mk_time(row):
        # datecol looks like "2021/07/20 16:14" (minute resolution) + SEC seconds
        d = str(row[datecol]).strip().replace("/", "-").replace(" ", "T")
        return UTCDateTime(f"{d}:00") + float(row["SEC"])

    out = pd.DataFrame()
    out["id"] = df["ID-NUM"].astype(int)
    out["time"] = df.apply(_mk_time, axis=1)
    out["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
    out["lon"] = pd.to_numeric(df["LON"], errors="coerce")
    out["depth"] = pd.to_numeric(df["DEPTH"], errors="coerce")
    out["num"] = pd.to_numeric(df["NUM"], errors="coerce")
    out["gap"] = pd.to_numeric(df["GAP"], errors="coerce")
    out["rms"] = pd.to_numeric(df["RMS"], errors="coerce")
    out["erh"] = pd.to_numeric(df["ERH"], errors="coerce")
    out["erz"] = pd.to_numeric(df["ERZ"], errors="coerce")
    out["qual"] = df["QASR"].astype(str).str.strip()
    return out


# ------------------------------------------------------------------ .reloc (HypoDD)
RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]


def read_reloc(path) -> pd.DataFrame:
    """Parse a HypoDD `hypoDD.reloc` (whitespace-separated, 24 columns).

    Returns an empty frame for an empty/missing reloc (e.g. a HypoDD run that
    aborted on MAXDATA0) so callers can report rather than crash."""
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=RELOC_COLS)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=RELOC_COLS + ["time"])
    if df.empty:
        df["time"] = pd.Series(dtype=object)
        return df

    def _mk_time(r):
        return (UTCDateTime(int(r.yr), int(r.mo), int(r.dy),
                            int(r.hr), int(r.mi), 0) + float(r.sc))

    df["time"] = df.apply(_mk_time, axis=1)
    return df


def read_event_dat(path) -> pd.DataFrame:
    """Parse a HypoDD `event.dat` / `event.sel` (whitespace-separated).

    Columns: DATE TIME LAT LON DEPTH MAG EH EZ RMS ID
    """
    cols = ["date", "time", "lat", "lon", "depth", "mag", "eh", "ez", "rms", "id"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    df["id"] = df["id"].astype(int)
    return df
