"""Cluster config for the JOINT 2020+2026 Haenam sequences.

Purpose: relocate both sequences in ONE relative frame so the 2020-vs-2026 offset is
physically meaningful (separate runs leave the inter-cluster offset unconstrained).
97 events: 77 from the 2020 sequence (2020-04 -> 2022-06, STP-fetched SAC incl. KG stations)
+ 20 from the 2026-08 sequence (NECIS event archive). wf_source="mixed" dispatches each
event to its own layout. All available stations are used (KS + KG masters).
"""
import os

from pipeline import config
from pipeline.clusters._base import mixed_cluster

CONFIG = mixed_cluster(
    name="Haenam_joint",
    region="haenam_joint",
    src_root=os.path.join(config.PROJECT_ROOT, "haenam_joint_cluster"),
    epicenter=(34.66, 126.401),
    region_bounds=(34.46, 34.87, 126.20, 126.61),
    station_masters=("KS_station.csv", "KG_station.csv"),
    dtct_isolv=1,
)

# Match 2026_Haenam: the template default of 10 workers left ~50 of this box's 64 cores
# idle through the 1000-replica bootstrap. NOTE: do not hand-edit this file while a
# --mainshock run is patching it -- the patcher inserts by regex and the two collide.
from dataclasses import replace as _replace
CONFIG = _replace(CONFIG, num_cores=24)

# Mainshock cross-correlation treatment for ALL THREE large events across both
# sequences. _window() matches with a set intersection (`if s & set(key)`), so one key
# covers every pair any of them takes part in -- crucially including the 2020 M3.1 vs
# 2026 M3.1 pair itself, which is the measurement that tests whether the two sequences
# ruptured the same patch.
#
# A wider band and shorter window suit their larger, longer sources; the other 145
# events keep the cluster default (0.5 s pre/post, 5-20 Hz).
CONFIG = _replace(CONFIG, xcorr_pair_overrides={
    frozenset({"20200503130714",      # M3.1  2020-05-03 22:07:14 KST
               "20260821192129",      # M2.7  2026-08-22 04:21:29 KST
               "20260822214301"}):    # M3.1  2026-08-23 06:43:01 KST
        dict(pre=0.05, post=0.05, bandpass=(1, 40)),
})
