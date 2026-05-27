"""Kimcheon cluster — 26 events (2020-2025), KS network, KMA SAC archive."""
import os

from pipeline import config
from pipeline.clusters._base import kma_cluster

CONFIG = kma_cluster(
    name="kimcheon",
    region="Kimcheon",
    src_root=os.path.join(config.PROJECT_ROOT, "Kimcheon_cluster"),
    epicenter=(36.01, 128.01),          # NB0 evla/evlo
    region_bounds=(35.2, 36.8, 127.2, 128.8),
    dtct_isolv=2,   # LSQR: 26 co-located events -> large dt.ct overflows SVD's MAXDATA0
)
