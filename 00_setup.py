# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 00 — Environment Setup
# MAGIC
# MAGIC Run this notebook **once per cluster** (or add it as an init script) to install
# MAGIC the required Python packages. After it completes, attach the remaining notebooks
# MAGIC to the same cluster.

# COMMAND ----------

# Install required packages (not pre-installed on Databricks Runtime)
%pip install tqdm seaborn fastparquet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Data Paths
# MAGIC
# MAGIC Set the environment variable `DMAN_BASE_DIR` to the DBFS (or Unity Catalog
# MAGIC Volume) path that contains the project's `data/` folder.
# MAGIC
# MAGIC **Example layout on DBFS:**
# MAGIC ```
# MAGIC /dbfs/mnt/<container>/
# MAGIC     data/
# MAGIC         taxi_time/
# MAGIC             median_taxi_matrix.pkl
# MAGIC         converted/
# MAGIC             input_DMAN_1.csv
# MAGIC             ...
# MAGIC ```
# MAGIC
# MAGIC Update the cell below to match your mount point / volume path.

# COMMAND ----------

import os
import sys
from pathlib import Path

# ── Path configuration ────────────────────────────────────────────────────────
# Set this to the DBFS (or Volume) root that contains the `data/` folder.
# If you cloned the repo into Databricks Repos, the default (repo root) already
# contains `data/` so you may leave DMAN_BASE_DIR unset.
DMAN_BASE_DIR = ""   # e.g. "/dbfs/mnt/mycontainer"  — leave empty to use repo root

if DMAN_BASE_DIR:
    os.environ["DMAN_BASE_DIR"] = DMAN_BASE_DIR
    print(f"DMAN_BASE_DIR set to: {DMAN_BASE_DIR}")
else:
    print("DMAN_BASE_DIR not set — using repo root (data/ must live next to the notebooks)")

# ── sys.path — add the databricks/ folder so all library modules are importable ─
# When notebooks live inside the databricks/ subfolder of a Databricks Repo,
# this ensures `import data_loader` etc. resolves correctly.
_NOTEBOOK_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

print("sys.path configured — library modules are importable.")
print("Setup complete. You can now run the other notebooks.")
