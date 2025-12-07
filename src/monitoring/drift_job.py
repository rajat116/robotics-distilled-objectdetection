import os
from datetime import datetime

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def make_dummy_reference(n: int = 500) -> pd.DataFrame:
    """Synthetic 'reference' data – e.g. from training distribution."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "mean_brightness": rng.normal(loc=100, scale=20, size=n),
            "num_detections": rng.poisson(lam=2.0, size=n),
            "avg_confidence": rng.normal(loc=0.7, scale=0.1, size=n),
        }
    )


def make_dummy_current(n: int = 500, drift: bool = True) -> pd.DataFrame:
    """Synthetic 'current' data – slightly shifted if drift=True."""
    rng = np.random.default_rng(123)
    loc_brightness = 110 if drift else 100
    loc_conf = 0.6 if drift else 0.7
    return pd.DataFrame(
        {
            "mean_brightness": rng.normal(loc=loc_brightness, scale=20, size=n),
            "num_detections": rng.poisson(lam=2.0, size=n),
            "avg_confidence": rng.normal(loc=loc_conf, scale=0.1, size=n),
        }
    )


def run_drift_job() -> str:
    """Run a simple data drift report and save HTML.

    Returns the path of the generated report.
    """
    # In a real setup you'd load:
    # - reference stats from training
    # - current stats from recent predictions
    reference = make_dummy_reference()
    current = make_dummy_current(drift=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report_dir = os.environ.get("DRIFT_REPORT_DIR", "reports")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(report_dir, f"drift_report_{timestamp}.html")
    report.save_html(out_path)

    print(f"[drift_job] Drift report written to: {out_path}")
    return out_path


if __name__ == "__main__":
    run_drift_job()
