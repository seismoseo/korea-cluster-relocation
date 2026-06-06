# Python-only relocation backend (HypoSVI + relocDD-py)

Locate and relocate **without a Fortran toolchain**. The default chain
(`hyp1.40` + `ph2dt` + `hypoDD`) stays the supported reference; this is an opt-in,
pure-Python alternative. With the bundled (kim1983 / kim2011) EikoNet weights, the
whole thing is a one-line shortcut:

```bash
./pocketquake.sh catalog.csv myslug --python
# = --loc-backend hyposvi --reloc-backend relocdd_py
```

It is **reproducible for any user and any cluster**: pretrained EikoNet weights ship
via a GitHub release, and every per-cluster detail (the SVGD search box) is derived
automatically from that cluster's `region_bounds`.

---

## 1. One-time setup

```bash
# Clone the three tools next to PocketQuake (not on PyPI)
git clone https://github.com/katie-biegel/relocDD-py.git
git clone https://github.com/Ulvetanna/HypoSVI.git
git clone https://github.com/Ulvetanna/EikoNet.git        # HypoSVI's travel-time engine

# Point PocketQuake at them (.env, gitignored)
echo "RELOCDD_PY_DIR=$PWD/relocDD-py" >> .env
echo "HYPOSVI_DIR=$PWD/HypoSVI"       >> .env
echo "EIKONET_DIR=$PWD/EikoNet"       >> .env
```

The extra deps these clones need — PyTorch, pyproj, **seaborn, scikit-learn, scikit-fmm**
(HypoSVI/EikoNet import them at module load) — are all in `environment.yml`. If you set up
the env before v1.8.0, add the last three: `pip install seaborn scikit-learn scikit-fmm`.

## 2. Get the pretrained EikoNet weights (one command)

The weights are ~90 MB each (gitignored), so they ship as GitHub release assets.
Fetch the bundled kim1983 + kim2011 models:

```bash
python -m pipeline.core.fetch_eikonet            # all bundled models (kim1983, kim2011)
# or just one:  python -m pipeline.core.fetch_eikonet --velmodel kim1983
```

They land under `pipeline/velocity_models/eikonet_<vm>/` and the backend
**auto-discovers** them — no `.env` editing needed. (Set `HYPOSVI_EIKONET_P/S` only
to override with your own.)

## 3. Run

```bash
./pocketquake.sh catalog.csv myslug --python
```

Outputs are byte-compatible with the Fortran path (`1.HypoInv/<vm>/<Region>.sum`,
`2.HypoDD/.../hypoDD.reloc`), so the results notebook is identical.

### Compare against the Fortran pipeline

```bash
./pocketquake.sh catalog.csv myslug --compare
```

Runs the default (Fortran) pipeline, then re-runs the Python backend on the **same
picks**, and builds an executed `pipeline/notebooks/04_compare_<slug>.ipynb` comparing
both the **absolute** locations (HYPOINVERSE vs HypoSVI) and the **final** relocation
(ff vs pp), with overlaid maps, depth sections, and per-event delta tables.

---

## Train your own velocity model (optional)

To use a 1-D model that isn't bundled, write a 3-column CSV — one row per layer top,
`depth_km, vp_kms, vs_kms` — and train it in one command (GPU auto-detected):

```bash
cat > my1d.csv <<EOF
0,5.5,3.2
10,6.0,3.5
30,7.8,4.5
EOF
python -m pipeline.core.eikonet_train --velmodel my1d --vel-csv my1d.csv
#   --device auto (GPU if available) | --epochs 200 | --samples 150000
#   --lon 125,131 --lat 33,40 --depth 0,50   # box MUST cover your cluster
```

This writes `pipeline/velocity_models/eikonet_my1d/` + `eikonet_meta.json` (it encodes
the layers as a HYPOINVERSE-matching step model and extends the bottom layer to the box
floor). To use it, set `HYPOSVI_EIKONET_P/S` to the `my1d` checkpoints (or
`cfg.hyposvi_velmodel="my1d"`). One model serves any number of clusters inside its box —
you do **not** retrain per cluster. On GPU a P+S pair trains in a few minutes.

> The recipe, velocity CSV, and `eikonet_meta.json` are tracked; `*.pt` weights are
> gitignored and regenerable, so a fresh clone reproduces identical models.

---

## How it works (and why each piece is there)

| stage | Fortran default | Python backend |
|---|---|---|
| absolute location | `hyp1.40` | **HypoSVI** + EikoNet (`hyposvi_backend.py`) |
| phase→diff times | `ncsn2pha` + `ph2dt` | **relocDD-py** ph2dt, fed a generated `phase.dat` |
| relative relocation | `hypoDD` | **relocDD-py** hypoDD (`relocdd_py_backend.py`) |
| solver | SVD, auto→LSQR+adaptive damping above MAXDATA0 | **same** — SVD ≤10000 diff-times, else LSQR with the condition-number (CND→40–80) damping search ported verbatim |
| 95% uncertainty | bootstrap (resample dt, re-invert ×1000) | **same** — `relocdd_py_backend.bootstrap_relocation` mirrors `hypodd.bootstrap_relocation` |

- The HypoSVI adapter reads picks straight from the SAC `a`/`t0` headers (same
  arrivals the Fortran path uses), locates, and writes a HYPOINVERSE-format `.sum`.
- The relocDD-py adapter generates `phase.dat` from that `.sum` + the picks and runs
  relocDD-py's own ph2dt — so a HypoSVI `.sum` actually drives the relocation, and
  the chain is fully Fortran-free.

## Validation (chungju, 4-event sequence)

What double-difference relocation actually determines is the **relative** structure
(the absolute centroid is not constrained by the data and reflects only the absolute
locator). On the **translation-removed** comparison the two fully-independent pipelines
agree to the metre:

| comparison | median dHoriz | dDepth |
|---|---|---|
| HYPOINVERSE vs HypoSVI — absolute (raw locator difference) | ~180 m | ~70 m |
| ff vs pp — **relative** (translation-removed, final dt.cc) | **1 m** | **3 m** |

The absolute offset is the expected HypoSVI-vs-HYPOINVERSE difference; the relative
structures converge once the cross-correlations drive the relocation.

**Relocator equivalence** — feeding relocDD-py and Fortran `hypoDD` the *identical*
`dt.ct`/`dt.cc`/`event.dat` (same control file), the relocations agree to **1.3 m
horizontal / 2.5 m depth relative**. relocDD-py is a faithful port of `hypoDD`'s
inversion; the adapter only hardens its clone against real-data implementation bugs
(see *For developers* below).

**Bootstrap uncertainty** — the headline 95% error bars come from bootstrapping the
differential times and re-inverting (same procedure as the Fortran workflow; HypoDD's
formal LSQR errors underestimate). Fortran vs relocDD-py bootstrap agree on chungju
(~6/18 m vs ~7/22 m horizontal/vertical 95%).

## For developers — implementation notes

You don't need these to *run* the backend (everything below is automatic); they matter only
if you modify it. Full rationale lives in the code comments of
`pipeline/core/{hyposvi_backend,relocdd_py_backend}.py`.

- **EikoNet uses `+units=km`** (lon/lat→UTM is *not* divided by 1000, while depth/velocity are
  km) — a metres projection makes travel times ~1000× too large and HypoSVI returns NaN.
- **SVGD particles seed inside the cluster** (`region_bounds` + margin), not EikoNet's
  all-Korea domain, or they don't converge. Automatic, per-cluster.
- **relocDD-py is hardened on the clone** by `_ensure_relocdd_patches()` (no upstream edits) so
  it reproduces Fortran `hypoDD` on real data — chiefly an `int8` event-pair counter that
  overflows at 127 (→ `int32`), plus divide-by-zero guards and a few real-data edge cases. The
  adapter renders `ISTART=2` (equivalent to Fortran `ISTART=1` for *relative* locations) and
  `ISOLV=1` (SVD), auto-switching to LSQR + adaptive damping above `MAXDATA0` (10000 diff-times).
- **GPU**: location auto-detects CUDA. EikoNet *training* leaks the autograd graph per epoch on
  **CPU only** — cap samples or just use `--device auto`.

## Bundled models

| model | P val-loss | S val-loss | box | layers |
|---|---|---|---|---|
| kim1983 | ~0.003 | ~0.004 | all-Korea (lon 125–131, lat 33–40, dep 0–50) | 3 |
| kim2011 | ~0.005 | ~0.005 | all-Korea | 4 |

Trained on GPU at 150k samples × 200 epochs (`--device auto`). (An earlier CPU run
plateaued S at ~0.10; more epochs on GPU fixed it.)

The EikoNet encodes a **1-D** layered model — HypoSVI gains probabilistic uncertainties
but not 3-D velocity structure (that would need a 3-D field at EikoNet training time).
