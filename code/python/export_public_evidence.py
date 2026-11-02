"""从冻结运行导出公开指标；不复制波形、逐样本真值或模型。"""

import argparse
import json
from pathlib import Path

from public_data import sha256

FIELDS = {
    "care": ["run_id", "protocol", "cases", "source_manifest_sha256", "source_hash"],
    "electrode": [
        "run_id",
        "protocol",
        "results",
        "paired_gains",
        "source_manifest_sha256",
        "source_hash",
        "limitations",
    ],
    "rejection": [
        "run_id",
        "protocol",
        "cases",
        "source_hash",
        "motor_manifest_sha256",
        "actuator_inputs",
    ],
    "sensitivity": [
        "run_id",
        "protocol",
        "comopi",
        "zema",
        "zema_input_sha256",
        "source_hash",
        "helper_sha256",
        "causality",
    ],
}


def export(name, directory, destination):
    source = directory / "metrics.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    result = {key: report[key] for key in FIELDS[name]}
    result["metrics_sha256"] = sha256(source)
    if name == "care":
        result["data_attribution"] = {
            "authors": ["Christian Gück", "Cyriana Roelofs"],
            "source": "https://zenodo.org/records/15846963",
            "license": "CC-BY-SA-4.0",
            "changes": "12-case subset; independent feature extraction, training and custom event evaluation",
        }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / (name + ".json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in FIELDS:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks",
    )
    args = parser.parse_args()
    for name in FIELDS:
        export(name, getattr(args, name), args.destination)
