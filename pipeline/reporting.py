"""Auto-generated beamer PDF summary for a finished cluster run.

    from pipeline import reporting
    reporting.make_run_summary("tongyeong")            # -> runs/tongyeong/summary/tongyeong_summary.pdf

    # or on the command line
    python -m pipeline.cli.make_summary --cluster tongyeong

Renders the headline result figures (relocated epicenters, depth cross-sections,
focal mechanisms, cumulative seismicity) plus a stats header into ONE uniform
beamer deck and compiles it with **tectonic** (XeTeX, self-contained). With
``--animate`` it also embeds the time-lapse GIF two ways: a static last frame
(plays everywhere) and a real ``\\animategraphics`` animation (plays in Acrobat /
Okular / pdfpc — a normal/browser PDF viewer shows the first frame).

Every figure is rendered defensively: a stage that errors is logged and its slide
is dropped, so a partial run still yields a deck. The compiler is resolved from
``$TECTONIC_BIN`` -> ``tectonic`` on PATH -> the tex-env fallback.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pipeline import config, viz  # noqa: E402
from pipeline.core import sumio  # noqa: E402

_TECTONIC = (
    os.environ.get("TECTONIC_BIN")
    or shutil.which("tectonic")
    or "/home/msseo/miniforge3/envs/tex/bin/tectonic"
)


def _pq_version():
    """Best-effort PocketQuake version: installed package -> superproject __init__ -> None."""
    try:
        import pocketquake
        return pocketquake.__version__
    except Exception:  # noqa: BLE001
        pass
    # reporting.py lives at <super>/external/korea-cluster-relocation/pipeline/reporting.py
    here = os.path.dirname(os.path.abspath(__file__))
    # here = <super>/external/korea-cluster-relocation/pipeline -> up 3 to <super>
    init = os.path.normpath(os.path.join(here, "..", "..", "..",
                                         "pocketquake", "__init__.py"))
    try:
        import re
        with open(init) as fh:
            m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', fh.read())
        if m:
            return m.group(1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _tex(s) -> str:
    """Escape a string for LaTeX text."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def _savefig(fn, path, **kw):
    """Call a viz function that returns or draws a matplotlib figure; save it.

    Returns the basename on success, ``None`` (logged) on any failure."""
    try:
        ret = fn(**kw)
        fig = ret if isinstance(ret, Figure) else plt.gcf()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        plt.close("all")
        return os.path.basename(path)
    except Exception as e:  # noqa: BLE001 - a bad figure must not kill the deck
        print(f"  [skip figure] {os.path.basename(path)}: {type(e).__name__}: {e}")
        plt.close("all")
        return None


def _collect_stats(cfg, velmodel, fmvm):
    """Headline numbers for the title/overview slide (each piece guarded)."""
    st = {}
    try:
        s = sumio.read_sum(config.sum_file(cfg, velmodel))
        st["n_located"] = len(s)
        times = sorted(u.datetime for u in s.time)
        st["t0"], st["t1"] = times[0], times[-1]
        st["depth_min"], st["depth_max"] = float(s.depth.min()), float(s.depth.max())
        st["rms_med"] = float(s.rms.median())
        st["gap_med"] = float(s.gap.median())
    except Exception as e:  # noqa: BLE001
        print(f"  [stats] sum: {e}")
    try:
        reloc, _branch = viz._reloc_path(cfg)
        rd = sumio.read_reloc(reloc)
        st["dd_depth_min"], st["dd_depth_max"] = float(rd.depth.min()), float(rd.depth.max())
    except Exception as e:  # noqa: BLE001
        print(f"  [stats] dt.cc depth: {e}")
    try:
        rc = viz.relocation_counts(cfg, velmodel).set_index("stage")["events"]
        for k in rc.index:
            if "dt.cc" in k:
                st["n_dtcc"] = int(rc[k])
            elif "dt.ct" in k:
                st["n_dtct"] = int(rc[k])
    except Exception as e:  # noqa: BLE001
        print(f"  [stats] reloc counts: {e}")
    try:
        mt = viz.mechanism_table(cfg, fmvm)
        st["n_mech"] = len(mt)
        st["mech_q"] = {str(k): int(v) for k, v in mt.quality.value_counts().items()}
    except Exception as e:  # noqa: BLE001
        print(f"  [stats] mechanisms: {e}")
    try:
        mags = viz._event_magnitudes(cfg)
        vals = list(getattr(mags, "values", lambda: mags)())
        vals = [float(m) for m in vals if m is not None and m == m]
        if vals:
            st["mag_min"], st["mag_max"] = min(vals), max(vals)
    except Exception as e:  # noqa: BLE001
        print(f"  [stats] magnitudes: {e}")
    return st


def _stats_lines(cfg, velmodel, fmvm, st):
    """Format the stats block as LaTeX itemize lines."""
    L = []
    L.append(f"Velocity model: \\texttt{{{_tex(velmodel)}}} \\quad FM model: \\texttt{{{_tex(fmvm)}}}")
    if "t0" in st:
        L.append(f"Period: {st['t0']:%Y-%m-%d} to {st['t1']:%Y-%m-%d} (UTC)")
    nl = st.get("n_located")
    nd = st.get("n_dtcc")
    if nl is not None:
        reloc = f", {nd} HypoDD dt.cc relocated" if nd is not None else ""
        L.append(f"Events: {nl} HYPOINVERSE located{reloc}")
    if "depth_min" in st:
        line = f"Depth range: {st['depth_min']:.1f}--{st['depth_max']:.1f} km (HYPOINVERSE)"
        if "dd_depth_min" in st:
            line += f"; {st['dd_depth_min']:.1f}--{st['dd_depth_max']:.1f} km (dt.cc)"
        L.append(line)
    if "mag_min" in st:
        L.append(f"Magnitude range: M{st['mag_min']:.1f}--M{st['mag_max']:.1f}")
    if "rms_med" in st:
        L.append(f"Median RMS: {st['rms_med']:.3f} s \\quad Median gap: {st['gap_med']:.0f}\\textdegree")
    if "n_mech" in st:
        q = st.get("mech_q", {})
        qs = ", ".join(f"{k}:{q[k]}" for k in sorted(q)) if q else ""
        L.append(f"Focal mechanisms: {st['n_mech']}" + (f" (quality {qs})" if qs else ""))
    return L


def _extract_gif_frames(gif_path, out_dir, max_frames=80):
    """Split an animated GIF into frame_<i>.png; return (n_frames, last_frame_basename)."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(gif_path)
    n = getattr(im, "n_frames", 1)
    step = max(1, n // max_frames)
    idx = list(range(0, n, step))
    last = None
    for j, fi in enumerate(idx):
        im.seek(fi)
        frame = im.convert("RGB")
        name = f"frame_{j}.png"
        frame.save(os.path.join(out_dir, name))
        last = name
    return len(idx), last


def _build_tex(cfg, slides, stats_lines, anim, region, cluster, pqversion):
    """Assemble the beamer source. ``slides`` = [(title, img_basename, caption)]."""
    head = [
        r"\documentclass[aspectratio=169]{beamer}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{animate}",
        r"\graphicspath{{figs/}}",
        r"\usetheme{metropolis}",
        r"\setbeamertemplate{frame numbering}[fraction]",
        r"\title{PocketQuake run summary}",
        rf"\subtitle{{{region}}}",
        rf"\date{{Generated \today\ \textbar\ PocketQuake{(' v' + pqversion) if pqversion else ''}}}",
        r"\author{Automated cluster-relocation report}",
        r"\begin{document}",
        r"\maketitle",
    ]
    # overview / stats frame
    body = [r"\begin{frame}{Overview}", r"\begin{itemize}"]
    body += [rf"\item {ln}" for ln in stats_lines]
    body += [r"\end{itemize}", r"\end{frame}"]
    # one frame per figure
    for title, img, cap in slides:
        body += [
            rf"\begin{{frame}}{{{_tex(title)}}}",
            r"\centering",
            rf"\includegraphics[width=\linewidth,height=0.82\textheight,keepaspectratio]{{{img}}}",
        ]
        if cap:
            body.append(rf"\par\smallskip\footnotesize {_tex(cap)}")
        body.append(r"\end{frame}")
    # time-lapse: static frame + real animation
    if anim:
        n_frames, last, fps = anim
        body += [
            r"\begin{frame}{Time-lapse seismicity (static key frame)}",
            r"\centering",
            rf"\includegraphics[width=\linewidth,height=0.80\textheight,keepaspectratio]{{anim/{last}}}",
            r"\par\smallskip\footnotesize Last frame of the cumulative time-lapse "
            r"(plays in every PDF viewer).",
            r"\end{frame}",
            r"\begin{frame}{Time-lapse seismicity (animated)}",
            r"\centering",
            rf"\animategraphics[autoplay,loop,width=\linewidth,height=0.78\textheight,keepaspectratio]{{{fps}}}{{anim/frame_}}{{0}}{{{n_frames - 1}}}",
            r"\par\smallskip\footnotesize Animates in Acrobat / Okular / pdfpc; "
            r"a basic or browser viewer shows the first frame.",
            r"\end{frame}",
        ]
    return "\n".join(head + body + [r"\end{document}"]) + "\n"


def make_run_summary(cluster, velmodel="kim1983", fm_velmodel=None,
                     animate=True, anim_fps=6, pqversion=None):
    """Build runs/<cluster>/summary/<cluster>_summary.pdf. Returns the PDF path."""
    cfg = config.load_cluster(cluster)
    fmvm = fm_velmodel or getattr(cfg, "fm_velmodel", velmodel)
    region = _tex(cfg.region)
    if pqversion is None:
        pqversion = _pq_version()

    summ = os.path.join(cfg.output_root, "summary")
    figdir = os.path.join(summ, "figs")
    os.makedirs(figdir, exist_ok=True)
    print(f"[summary] {cluster}: rendering figures -> {figdir}")

    slides = []
    m = _savefig(viz.map_catalog, os.path.join(figdir, "01_map_reloc.png"),
                 cfg=cfg, velmodel=velmodel, source="reloc")
    if m:
        slides.append(("Relocated epicenters (dt.cc)", m,
                       "HypoDD cross-correlation relocation; color = depth."))
    m = _savefig(viz.depth_sections, os.path.join(figdir, "02_depth.png"),
                 cfg=cfg, velmodel=velmodel, source="reloc")
    if m:
        slides.append(("Depth cross-sections", m,
                       "Lon-depth and lat-depth sections of the dt.cc relocation."))
    m = _savefig(viz.map_mechanisms, os.path.join(figdir, "03_mech.png"),
                 cfg=cfg, velmodel=fmvm)
    if m:
        slides.append(("Focal mechanisms", m,
                       "Located epicenters with high-confidence beachballs (SKHASH)."))
    # the highest-quality mechanism as an actual beachball with first-motion polarities + S/P ratios
    try:
        mt = viz.mechanism_table(cfg, fmvm)
        rank = {q: i for i, q in enumerate("ABCD")}
        best = mt.assign(_q=mt.quality.map(lambda q: rank.get(str(q), 9))).sort_values("_q").iloc[0]
        eid, q = str(best.event_id), str(best.quality)
        m = _savefig(viz.plot_custom_beachball, os.path.join(figdir, "03b_beachball.png"),
                     cfg=cfg, event_id=eid, velmodel=fmvm)
        if m:
            slides.append((f"Focal mechanism --- event {eid} (quality {q})", m,
                           "Beachball with per-station first-motion polarities "
                           "(red up / blue down) and S/P amplitude-ratio markers (SKHASH)."))
    except Exception as e:  # noqa: BLE001
        print(f"  [skip figure] 03b_beachball.png: {type(e).__name__}: {e}")
    m = _savefig(viz.cumulative_events, os.path.join(figdir, "04_cumulative.png"),
                 cfg=cfg, velmodel=velmodel)
    if m:
        slides.append(("Cumulative seismicity", m, "Cumulative located-event count over time."))

    # waveform similarity for the largest dt.cc sub-cluster: full-waveform (P+S+coda) gather + Z CC
    # matrix (chronological + hierarchical) at the nearest station common to the events, 5-20 Hz.
    try:
        from pipeline.analysis import similarity as _simil
        _groups = _simil.cluster_events_by_cid(cfg, min_events=4)
        _single = len(_groups) == 1
        for _cid in list(_groups)[:1]:                     # largest sub-cluster only (bounded deck)
            _sfx = "" if _single else f" --- sub-cluster {_cid}"
            g = _savefig(viz.plot_cluster_similarity_gather,
                         os.path.join(figdir, f"05_simgather_cid{_cid}.png"),
                         cfg=cfg, cid=_cid, show_cid=not _single)
            if g:
                slides.append((f"Waveform similarity{_sfx}", g,
                               "Full-waveform (P+S+coda) gather at the nearest common station, P-aligned "
                               "(red), S pick (blue), past (top) to present (bottom), 5-20 Hz; no stack."))
            cc = _savefig(viz.plot_cluster_cc_matrix,
                          os.path.join(figdir, f"05_ccmat_chrono_cid{_cid}.png"),
                          cfg=cfg, cid=_cid, order="chrono", show_cid=not _single)
            if cc:
                slides.append((f"Waveform CC matrix --- chronological{_sfx}", cc,
                               "Z full-waveform NCC matrix in time order; a bright block is a repeating family."))
            ch = _savefig(viz.plot_cluster_cc_matrix,
                          os.path.join(figdir, f"05_ccmat_cluster_cid{_cid}.png"),
                          cfg=cfg, cid=_cid, order="cluster", show_cid=not _single)
            if ch:
                slides.append((f"Waveform CC matrix --- hierarchical{_sfx}", ch,
                               "Same matrix reordered by hierarchical clustering (dendrogram); repeating "
                               "sub-families gather into bright blocks."))
    except Exception as e:  # noqa: BLE001
        print(f"  [skip figure] waveform similarity: {type(e).__name__}: {e}")

    anim = None
    if animate:
        try:
            gif = os.path.join(summ, f"{cluster}_seismicity.gif")
            print("[summary] rendering time-lapse GIF (animate_seismicity) ...")
            viz.animate_seismicity(cfg, velmodel=fmvm, out_path=gif, fps=anim_fps)
            n_frames, last = _extract_gif_frames(gif, os.path.join(figdir, "anim"))
            if last and n_frames > 1:
                anim = (n_frames, last, anim_fps)
                print(f"[summary] GIF -> {n_frames} frames (last={last})")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip animation] {type(e).__name__}: {e}")

    stats = _collect_stats(cfg, velmodel, fmvm)
    stats_lines = _stats_lines(cfg, velmodel, fmvm, stats)
    tex = _build_tex(cfg, slides, stats_lines, anim, region, cluster, pqversion)

    tex_path = os.path.join(summ, f"{cluster}_summary.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    print(f"[summary] wrote {tex_path} ({len(slides)} figure slides"
          f"{', +animation' if anim else ''}); compiling with tectonic ...")

    r = subprocess.run([_TECTONIC, os.path.basename(tex_path)], cwd=summ,
                       capture_output=True, text=True)
    pdf_path = os.path.join(summ, f"{cluster}_summary.pdf")
    if r.returncode != 0 or not os.path.exists(pdf_path):
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"tectonic failed (rc={r.returncode}) for {tex_path}")
    print(f"[summary] OK -> {pdf_path}")
    return pdf_path
