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
