"""用原始窗口复验专项 SDK，比较不可变实验分数与事件判定。"""

import json
from pathlib import Path

import numpy as np

from phm_core.actuator import load_records
from phm_core.artifacts import run_path
from phm_core.conditioning_inference import ConditioningScorer
from phm_core.profiles import load_profile

ROOT = Path(__file__).resolve().parents[2]


def run():
    path = run_path(ROOT / "outputs", "conditioning-study")
    scorer = ConditioningScorer(path / "bundle")
    x, _labels, _digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    split = json.loads((path / "valve-split.json").read_text())
    ids = np.array(split["test"])
    with np.load(path / "scores.npz") as archive:
        actual = scorer.valve(x[ids])
        expected = archive["valve_timing_rf_42"][ids]
        np.testing.assert_allclose(actual["score"], expected, atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(
            actual["anomalous"], expected > actual["threshold"]
        )
        report = {
            "run_id": path.name,
            "valve_test_windows": len(ids),
            "valve_max_abs_error": float(np.abs(actual["score"] - expected).max()),
            "actuator": {},
        }
        x, _y, _records, metadata = load_records(ROOT / "data/raw/cranfield-actuator")
        full = x.reshape(-1, 2000, 3)
        for protocol in scorer.report["actuator"]:
            ids = scorer.report["actuator"][protocol]["partitions"]["test"]
            contexts = [f"{metadata[i]['profile']}/{metadata[i]['load']}" for i in ids]
            for conditioned in (False, True):
                arm = "conditioned_rf" if conditioned else "raw_rf"
                result = scorer.actuator(full[ids], contexts, protocol, conditioned)
                expected = archive[f"actuator_{protocol}_{arm}_42"][ids]
                np.testing.assert_allclose(
                    result["scores"], expected, atol=1e-5, rtol=1e-4
                )
                np.testing.assert_array_equal(result["predicted"], expected.argmax(1))
                report["actuator"][f"{protocol}/{arm}"] = {
                    "records": len(ids),
                    "max_abs_error": float(np.abs(result["scores"] - expected).max()),
                }
        try:
            scorer.actuator(full[:1], ["unknown/new"])
        except ValueError:
            report["unknown_context_rejected"] = True
        else:
            raise AssertionError("未知工况未拒绝")
    output = ROOT / "work/validation" / f"conditioning-{path.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
