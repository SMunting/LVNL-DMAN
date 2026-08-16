# DMAN Flight Scheduler — Databricks

This folder contains the Databricks-ready version of the DMAN flight scheduling codebase.
All scheduling **logic is identical** to the original; only the entry-point boilerplate has been adapted for Databricks.

---

## Folder structure

```
databricks/
├── 00_setup.py                    ← Databricks notebook: install packages, configure paths
├── 01_main_scheduler.py           ← Databricks notebook: slot-based scheduler  (≈ main.py)
├── 02_main_pairwise_scheduler.py  ← Databricks notebook: pairwise scheduler    (≈ main_pairwise.py)
├── 03_analysis.py                 ← Databricks notebook: ATD compliance analyser (≈ analysis.py)
│
├── config.py                      ┐
├── global_vars.py                 │
├── data_loader.py  ← one change*  │ Library modules — unchanged copies of the
├── utils.py                       │ original source files. Import them normally.
├── flight_keys.py                 │
├── flight_metrics.py              │
├── flight_prioritization.py       │
├── flight_scheduler.py            │
├── slot_manager.py                │
├── ctot_analyzer.py               │
├── kpis.py                        │
├── optimized_scheduler.py         │
├── pairwise_scheduler.py          │
├── snapshot_generator.py          │
├── snapshot_generator_bins.py     │
├── snapshot_metrics.py            │
├── snapshot_explorer.py           │
├── schedule_diff.py               ┘
│
├── pairwise/
│   ├── __init__.py
│   ├── aircraft.py
│   ├── sequencer.py
│   └── spacing.py
│
└── data/
    ├── __init__.py
    ├── taxi_time/
    │   └── median_taxi_matrix.pkl  ← required data file
    └── runway/
        ├── __init__.py
        └── runway_mri.py
```

\* `data_loader.py`: the single-line change adds support for the `DMAN_BASE_DIR` environment variable so the taxi-time pickle can live anywhere on DBFS (see below).

---

## Quick start (Databricks Repos)

### 1. Import this repo

In Databricks:

1. Go to **Repos → Add Repo**
2. Paste the Git URL of your repository
3. Click **Create Repo**

### 2. Upload data files

The notebooks need the CSV input files and the taxi-time pickle.
Upload them to DBFS (or a Unity Catalog Volume) and note the mount path.

```
/dbfs/mnt/<your-container>/
    data/
        taxi_time/
            median_taxi_matrix.pkl
        converted/
            input_DMAN_1.csv
            input_DMAN_2026-07-13.csv
            ...
```

> If you place `data/` directly next to the notebooks inside the repo (i.e. the repo already contains `data/taxi_time/median_taxi_matrix.pkl`), no extra configuration is required.

### 3. Run `00_setup.py`

Attach the notebook to a cluster and **Run All**.

- Installs `tqdm`, `seaborn`, `fastparquet` (not pre-installed on Databricks Runtime).
- Sets `DMAN_BASE_DIR` if your data files live outside the repo root.

Open `00_setup.py` and edit the `DMAN_BASE_DIR` variable:

```python
DMAN_BASE_DIR = "/dbfs/mnt/mycontainer"   # path containing the data/ folder
```

Leave it empty (`""`) if `data/` lives in the repo root.

### 4. Run the scheduler notebooks

| Notebook | Purpose |
|----------|---------|
| `01_main_scheduler.py` | Slot-based scheduling (original `main.py`) |
| `02_main_pairwise_scheduler.py` | Continuous pairwise scheduling (original `main_pairwise.py`) |
| `03_analysis.py` | ATD compliance analysis (original `analysis.py`) |

Each notebook exposes **Databricks Widgets** at the top — fill them in and click **Run All**.

---

## Key parameter differences vs. CLI

| CLI flag (`main.py`) | Notebook widget |
|---------------------|-----------------|
| `--csv input_DMAN_1` | Widget `csv` = `input_DMAN_1` |
| `--slot-duration 600` | Widget `slot_duration` = `600` |
| `--slot-capacity 6` | Widget `slot_capacity` = `6` |
| `--ctot-min-margin 5` | Widget `ctot_min_margin` = `5` |
| `--ctot-max-margin 10` | Widget `ctot_max_margin` = `10` |
| `--runway 36C` | Widget `runway` = `36C` |
| `--output result` | Widget `output` = `result` |
| `--build-snapshots 2026-07-13` | Widget `build_snapshots` = `2026-07-13` |
| `--mri-day 2026-07-13` | Widget `mri_day` = `2026-07-13` |
| `-v` | Widget `verbose` = `1` |

---

## `DMAN_BASE_DIR` — configuring data paths

`data_loader.py` resolves the taxi-time matrix as:

```
$DMAN_BASE_DIR / data / taxi_time / median_taxi_matrix.pkl
```

If `DMAN_BASE_DIR` is not set, it falls back to the directory containing `data_loader.py` — which is the repo root when using Databricks Repos. This means **no environment variable is needed** if the `data/` folder is committed to the repo.

Set it in `00_setup.py` if data files live on a DBFS mount outside the repo:

```python
os.environ["DMAN_BASE_DIR"] = "/dbfs/mnt/mycontainer"
```

---

## Output files

By default, results are printed to the notebook output.  
To save to a file, fill in the `output` widget.

- In slot-based notebook: provide a file **stem** (e.g. `result_2026-07-13`). The notebook saves to `/dbfs/tmp/<stem>.csv`.
- In pairwise notebook: provide a **full DBFS path** (e.g. `/dbfs/tmp/pairwise_result.csv`).
- Snapshots are written to the `snapshots_out` widget path (default `/dbfs/tmp/snapshots`).
