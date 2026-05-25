"""Central configuration: paths, label maps, and modeling constants.

All paths are resolved relative to the repository root so the project is
portable across machines (no absolute paths anywhere in the code).
"""

from __future__ import annotations

from pathlib import Path

# Repository root is wherever this file lives.
ROOT = Path(__file__).resolve().parent

# Data directories.
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw file names (exactly as provided by the City of Toronto).
DRIVING_EVENTS_FILE = RAW_DIR / "Driving Events.xlsx"
FAULTS_FILE = RAW_DIR / "Engine Device Fault Alerts.xlsx"
JOBS_FILE = RAW_DIR / "Job History.xlsx"

# Figures output directory.
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Modeling window
# ---------------------------------------------------------------------------
# Telematics signals (faults, drives) span all of calendar year 2025.
# Job history extends from May 2024 to Feb 2026, but our usable prediction
# window is 2025 because that is where all three signals are simultaneously
# available.
MODEL_START_DATE = "2025-01-01"
MODEL_END_DATE = "2025-12-31"
PREDICTION_HORIZON_DAYS = 7

# ---------------------------------------------------------------------------
# JOB_REASON taxonomy
# ---------------------------------------------------------------------------
# Defines the prediction label. A vehicle on day t is positive if any job
# with a JOB_REASON in UNSCHEDULED_REASONS has JOB_OPEN_DT in (t, t+7d].
#
# The mapping below is documented in the README so the assessor can review
# and challenge the assumption. Sensitivity to this choice should be
# discussed (e.g. excluding warranty work `W` or recall campaigns `R`).
SCHEDULED_REASONS = {
    "P",  # PREVENTIVE MAINT
    "C",  # CAPITAL IMPROVEMENT
    "J",  # PREP. FOR DELIVERY
    "K",  # PREP. FOR DISPOSAL
    "S",  # SEASONAL CONVERSION
    "R",  # RECALL/CAMPAIGN (planned even though externally driven)
}

UNSCHEDULED_REASONS = {
    "D",  # DRIVERS REPORT
    "A",  # DEFECTIVE/ADDITIONAL
    "B",  # SERV.CALL/BREAKDOWN
    "F",  # NEGLIGENCE
    "G",  # NO START WITH FLEET
    "M",  # RPTED ACCIDENT/DMG
    "L",  # UNRPTED ACCIDENT/DMG
    "O",  # COMEBACK
    "Q",  # VANDALISM
    "W",  # WARRANTY (unplanned from the fleet's point of view)
}

# Reasons we exclude from both labels (administrative / parts logistics /
# external work that is neither a clear breakdown nor scheduled maintenance).
EXCLUDED_REASONS = {
    "H",  # MODIFICATION
    "U",  # VERIFY VENDOR WORK
    "E",  # PARTS SUPPLY
    "T",  # EXTERNAL REPAIRS
    "Z",  # NON MAINTENANCE
}

# ---------------------------------------------------------------------------
# Geography (Toronto City Hall) for weather queries.
# ---------------------------------------------------------------------------
TORONTO_LAT = 43.6532
TORONTO_LON = -79.3832

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
N_CV_FOLDS = 5
RANDOM_SEED = 42
