"""Jangsung cluster — 4 events (2023-2026), KS network, KMA SAC archive."""
import os

from pipeline import config
from pipeline.clusters._base import kma_cluster

CONFIG = kma_cluster(
    name="jangsung",
    region="Jangsung",
    src_root=os.path.join(config.PROJECT_ROOT, "Jangsung_cluster"),
    epicenter=(35.46, 126.81),          # NB0 evla/evlo
    region_bounds=(34.7, 36.2, 126.0, 127.6),
)
