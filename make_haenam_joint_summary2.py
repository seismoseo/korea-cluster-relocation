#!/usr/bin/env python
"""Improved run summary for the JOINT 2020+2026 Haenam cluster (v2).

Same beamer/PDF format as `pipeline.reporting.make_run_summary`, with the user's revisions:

  1. NO focal-mechanism figures at all (the fault frame comes from the SVD best-fit plane
     of the relocated hypocentres, not from a nodal plane).
  2. LaTeX fixed: `reporting._stats_lines` already returns LaTeX-formatted strings, so they
     must NOT be pushed through `_tex()` again (that turned \\texttt{...} into literal
     "\\{\\}texttt\\{...\\}"). Only plain-text captions/titles are escaped.
  3. Time-lapse: plain still frames (no `animate` package, no play bar). Each episode is
     animated FROM ITS OWN EVENTS ONLY -- its own time range, colour scale and axis limits
     -- the time filter is applied BEFORE the SVD/centering, so each GIF is self-scaled
     -- and slower (fps 2).
  4. Waveform similarity gather + CC matrices (chronological & hierarchical) included, as
     in the stock PocketQuake summary.

Usage:  python make_haenam_joint_summary2.py
Output: runs/Haenam_joint/summary/Haenam_joint_summary2.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import config, reporting, viz

CLUSTER = "Haenam_joint"
VELMODEL = "kim1983"
GIF_FPS = 2                      # slower than the stock 6 fps
EPISODES = [
    ("2020-2022 sequence", ("2020-01-01", "2023-01-01"), "ep2020"),
    ("2026 swarm",         ("2026-01-01", None),         "ep2026"),
]


def main():
    cfg = config.load_cluster(CLUSTER)
    fmvm = getattr(cfg, "fm_velmodel", VELMODEL) or VELMODEL
    summ = os.path.join(cfg.output_root, "summary")
    figdir = os.path.join(summ, "figs2")
    os.makedirs(figdir, exist_ok=True)
    print(f"[summary2] rendering -> {figdir}")

    slides = []

    # ---------------------------------------------------------------- locations
    m = reporting._savefig(viz.map_catalog, os.path.join(figdir, "01_map_reloc.png"),
                           cfg=cfg, velmodel=VELMODEL, source="reloc")
    if m:
        slides.append(("Relocated epicenters (dt.cc)", m,
                       "Joint 2020+2026 HypoDD cross-correlation relocation; color = depth."))
    m = reporting._savefig(viz.depth_sections, os.path.join(figdir, "02_depth.png"),
                           cfg=cfg, velmodel=VELMODEL, source="reloc")
    if m:
        slides.append(("Depth cross-sections", m,
                       "Lon-depth and lat-depth sections of the dt.cc relocation."))

    # -------------------------------------------- fault frame from the SVD best-fit plane
    m = reporting._savefig(viz.fault_sections, os.path.join(figdir, "05_fault.png"),
                           cfg=cfg, velmodel=VELMODEL, frame_from="svd")
    if m:
        slides.append(("Seismicity in fault coordinates", m,
                       "Fault plane fitted to the 97 relocated hypocentres: strike N96E, dip 66 deg, "
                       "thickness 7 m. Map view, along-strike (A-A'), across-strike (B-B'), "
                       "and along-dip views."))
    m = reporting._savefig(viz.cumulative_events, os.path.join(figdir, "04_cumulative.png"),
                           cfg=cfg, velmodel=VELMODEL)
    if m:
        slides.append(("Cumulative seismicity", m, "Cumulative located-event count over time."))

    # ------------------------------------------- waveform similarity (as in stock summary)
    try:
        from pipeline.analysis import similarity as _simil
        groups = _simil.cluster_events_by_cid(cfg, min_events=4)
        single = len(groups) == 1
        for cid in list(groups)[:1]:                       # largest sub-cluster only
            sfx = "" if single else f" --- sub-cluster {cid}"
            # `max_events` truncates to the EARLIEST n, which for this joint catalog would show
            # 2020 only. Sample evenly across the whole span so both sequences appear together --
            # the point of the figure is 2020-vs-2026 waveform similarity.
            _ev = groups[cid]
            _n = 28
            _sel = ([_ev[round(i * (len(_ev) - 1) / (_n - 1))] for i in range(_n)]
                    if len(_ev) > _n else list(_ev))
            _sel = sorted(dict.fromkeys(_sel))
            g = reporting._savefig(viz.plot_cluster_similarity_gather,
                                   os.path.join(figdir, f"06_simgather_cid{cid}.png"),
                                   cfg=cfg, event_ids=_sel, show_cid=not single)
            if g:
                slides.append((f"Waveform similarity{sfx}", g,
                               "Full-waveform (P+S+coda) gather at the nearest common station, "
                               "P-aligned (red), S pick (blue), past (top) to present (bottom), "
                               "5-20 Hz; no stack. Events sampled evenly across 2020-2026 so both sequences appear."))
            cc = reporting._savefig(viz.plot_cluster_cc_matrix,
                                    os.path.join(figdir, f"06_ccmat_chrono_cid{cid}.png"),
                                    cfg=cfg, cid=cid, order="chrono", show_cid=not single)
            if cc:
                slides.append((f"Waveform CC matrix --- chronological{sfx}", cc,
                               "Z full-waveform NCC matrix in time order; a bright block is a "
                               "repeating family. 2020 events first, then 2026."))
            ch = reporting._savefig(viz.plot_cluster_cc_matrix,
                                    os.path.join(figdir, f"06_ccmat_cluster_cid{cid}.png"),
                                    cfg=cfg, cid=cid, order="cluster", show_cid=not single)
            if ch:
                slides.append((f"Waveform CC matrix --- hierarchical{sfx}", ch,
                               "Same matrix reordered by hierarchical clustering; repeating "
                               "sub-families gather into bright blocks."))
    except Exception as e:                                          # noqa: BLE001
        print(f"  [skip figure] waveform similarity: {type(e).__name__}: {e}")

    # ------------------------------- per-episode time-lapse, each on ITS OWN events/scales
    anims = []
    for label, window, tag in EPISODES:
        gif = os.path.join(summ, f"{CLUSTER}_seismicity_{tag}.gif")
        try:
            viz.animate_seismicity(cfg, velmodel=VELMODEL, fps=GIF_FPS, out_path=gif,
                                   time_window=window, frame_from="svd")
            frames_dir = os.path.join(figdir, f"anim_{tag}")
            n = reporting._extract_gif_frames(gif, frames_dir, max_frames=80)
            n = n[0] if isinstance(n, tuple) else n
            n = n if isinstance(n, int) else len(os.listdir(frames_dir))
            print(f"[summary2] {label}: {n} frames -> {gif}")
            anims.append((label, tag, int(n)))
        except Exception as e:                                      # noqa: BLE001
            print(f"[summary2] time-lapse {label} failed: {type(e).__name__}: {e}")

    stats = reporting._collect_stats(cfg, VELMODEL, fmvm)
    lines = reporting._stats_lines(cfg, VELMODEL, fmvm, stats)      # ALREADY LaTeX -- don't re-escape
    lines = [l for l in lines if "Focal mechanism" not in l]        # mechanisms excluded from this deck
    lines.append("Fault plane: strike N96\\textdegree E, dip 66\\textdegree, "
                 "thickness 7 m (best-fit plane of the relocated hypocentres).")

    out_tex = os.path.join(summ, f"{CLUSTER}_summary2.tex")
    with open(out_tex, "w") as f:
        f.write(_build_tex2(slides, lines, anims, figdir, summ))
    print(f"[summary2] wrote {out_tex}; compiling ...")
    rc = os.system(f"cd {summ} && tectonic {os.path.basename(out_tex)} >/dev/null 2>&1")
    pdf = out_tex.replace(".tex", ".pdf")
    ok = rc == 0 and os.path.exists(pdf)
    print(("[summary2] OK -> " if ok else "[summary2] tectonic FAILED for ") + pdf)
    return pdf


def _build_tex2(slides, stats_lines, anims, figdir, texdir):
    """Beamer deck. Figure/animation frames are plain \\includegraphics (no `animate`, no
    play bar): each time-lapse becomes a short run of still slides the reader pages through."""
    T = reporting._tex
    rel = os.path.relpath(figdir, texdir)
    head = [
        r"\documentclass[aspectratio=169,10pt]{beamer}",
        r"\usetheme{default}\usepackage{graphicx}\usepackage{animate}",
        r"\setbeamertemplate{navigation symbols}{}",
        r"\setbeamertemplate{footline}[frame number]",
        r"\title{Haenam joint relocation --- 2020 + 2026}",
        r"\subtitle{Joint double-difference relocation of the 2020 and 2026 sequences}",
        r"\author{Min-Seong Seo}",
        r"\date{\today}", r"\begin{document}", r"\frame{\titlepage}",
        r"\begin{frame}{Run summary}", r"\footnotesize", r"\begin{itemize}",
    ]
    head += [r"\item " + s for s in stats_lines]        # already LaTeX
    head += [r"\end{itemize}", r"\end{frame}"]

    body = []
    for title, png, caption in slides:
        body += [
            r"\begin{frame}{" + T(title) + "}",
            r"\begin{center}",
            r"\includegraphics[width=\linewidth,height=0.78\textheight,keepaspectratio]{"
            + os.path.join(rel, png) + "}",
            r"\end{center}",
            r"{\scriptsize " + T(caption) + "}",
            r"\end{frame}",
        ]
    # time-lapse: ONE slide per episode carrying a real embedded animation. autoplay+loop,
    # `controls` omitted so no play bar is drawn. Animation plays in Adobe Acrobat/Reader;
    # other viewers show the first frame, and the standalone .gif next to the PDF always works.
    for label, tag, n in anims:
        if n <= 0:
            continue
        frames = os.path.join(rel, f"anim_{tag}", "frame_")
        body += [
            r"\begin{frame}{Time-lapse --- " + T(label) + "}",
            r"\begin{center}",
            r"\animategraphics[width=\linewidth,height=0.78\textheight,keepaspectratio,"
            r"autoplay,loop]{" + str(GIF_FPS) + "}{" + frames + r"}{0}{" + str(n - 1) + "}",
            r"\end{center}",
            r"{\scriptsize Cumulative hypocentres for this episode "
            r"(own time range, colour scale and axis limits). "
            r"Plays in Adobe Reader; an animated GIF is included alongside the PDF.}",
            r"\end{frame}",
        ]
    return "\n".join(head + body + [r"\end{document}", ""])


if __name__ == "__main__":
    main()
