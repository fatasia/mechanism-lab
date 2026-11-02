"""寿命与视觉的模型包、独立 SDK、HTTP 与错误输入验收。"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from phm_core.artifacts import run_path
from phm_core.domain_inference import DomainScorer
from phm_demo.api import app

ROOT = Path(__file__).resolve().parents[2]


def run():
    client = TestClient(app)
    result = {}
    for kind, family in (("rul-study", "rul"), ("vision-study", "vision")):
        path = run_path(ROOT / "outputs", kind)
        scorer = DomainScorer(path)
        with np.load(path / "parity.npz") as reference:
            raw = reference["raw"]
            scores = []
            for batch in np.array_split(
                raw, int(np.ceil(len(raw) / scorer.manifest["maxBatch"]))
            ):
                values = scorer.score(batch)
                scores.append(
                    values["remaining_cycles" if family == "rul" else "score"]
                )
            expected = (
                np.clip(reference["reference"], 0, 125)
                if family == "rul"
                else reference["score"]
            )
            np.testing.assert_allclose(
                np.concatenate(scores), expected, atol=1e-3, rtol=1e-4
            )
        body = {
            "run_id": path.name,
            "values": raw[:1].tolist(),
            "layout": scorer.manifest["input"]["layout"],
            "channels": scorer.manifest["input"].get("channels"),
        }
        response = client.post(f"/api/domain/{family}", json=body)
        assert response.status_code == 200 and response.json()["status"] == "scored"
        body["values"] = [[[]]]
        assert client.post(f"/api/domain/{family}", json=body).status_code == 422
        assert client.get(f"/api/benchmarks/{kind}").status_code == 200
        assert (
            client.get(
                f"/api/benchmarks/{kind}/download?run_id={path.name}"
            ).status_code
            == 200
        )
        if family == "vision":
            assert (
                client.get(
                    f"/api/benchmarks/vision/image/0?run_id={path.name}&layer=raw"
                ).status_code
                == 200
            )
            assert (
                client.get(
                    f"/api/benchmarks/vision/image/-1?run_id={path.name}"
                ).status_code
                == 404
            )
        result[family] = {
            "run_id": path.name,
            "reference_samples": len(raw),
            "sdk_and_api": "passed",
        }
    output = (
        ROOT
        / "work/validation"
        / (datetime.now(UTC).strftime("domains-%Y%m%dT%H%M%S") + ".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"report": str(output), "results": result}), flush=True)


if __name__ == "__main__":
    run()
