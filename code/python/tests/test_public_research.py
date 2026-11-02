import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]


def test_public_export_omits_raw_predictions(tmp_path, monkeypatch):
    exporter = module("export_public_evidence", monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    payload = {key: {} for key in exporter.FIELDS["electrode"]}
    payload["predictions"] = [{"actual": 123, "sample": "private-row"}]
    (run / "metrics.json").write_text(json.dumps(payload))
    exporter.export("electrode", run, tmp_path / "public")
    published = json.loads((tmp_path / "public/electrode.json").read_text())
    assert "predictions" not in published
    assert len(published["metrics_sha256"]) == 64


def module(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_range_zip_seeks_and_member_crc(tmp_path, monkeypatch):
    data = module("public_data", monkeypatch)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("example.csv", "id,value\n1,2\n")
    raw = stream.getvalue()
    monkeypatch.setattr(
        data,
        "curl_bytes",
        lambda url, byte_range: raw[byte_range[0] : byte_range[1] + 1],
    )
    remote = data.RemoteZipFile("https://example.test/file.zip", len(raw), tmp_path)
    with zipfile.ZipFile(remote) as archive:
        assert archive.read("example.csv") == b"id,value\n1,2\n"
    remote.seek(-3, 2)
    assert remote.read() == raw[-3:]
    with pytest.raises(ValueError):
        remote.seek(-1)


def test_rejecting_all_faults_does_not_improve_full_denominator(monkeypatch):
    study = module("study_condition_rejection", monkeypatch)
    result = study.rejection_metrics([0, 1, 1], [0, 1, 1], [True, False, False])
    assert result["accepted_fault_recall"] is None
    assert (
        result["all_fault_detected_lower_bound"] == 0 and result["rejected_faults"] == 2
    )


def test_heldout_labels_cannot_change_rejection_threshold(monkeypatch):
    study = module("study_condition_rejection", monkeypatch)
    monkeypatch.setitem(study.PROTOCOL, "trees", 3)
    monkeypatch.setitem(study.PROTOCOL, "seeds", [42])
    x = np.arange(24, dtype=float).reshape(12, 2)
    context = np.array(["source"] * 8 + ["target"] * 4)
    y = np.array([0, 1] * 6)
    first = study.compare(
        x, y, context, np.arange(6), np.arange(6, 8), np.arange(8, 12)
    )
    y[8:] = 1 - y[8:]
    second = study.compare(
        x, y, context, np.arange(6), np.arange(6, 8), np.arange(8, 12)
    )
    assert [r.get("threshold") for r in first["results"]] == [
        r.get("threshold") for r in second["results"]
    ]


def test_causal_smoothing_is_prefix_invariant(monkeypatch):
    study = module("study_public_sensitivity", monkeypatch)
    x = np.r_[np.zeros(20), np.ones(20) * 100]
    np.testing.assert_array_equal(
        study.trailing_median(x, 5)[:20], study.trailing_median(x[:20], 5)
    )


def test_electrode_encoder_cannot_use_target_or_density(monkeypatch):
    study = module("study_electrode", monkeypatch)
    row = {
        "metadata": {
            "Web_speed": 1,
            "Roll_Gap": 2,
            "Coat_Weight": 3,
            "Calendering_State": "before-calendering",
            "Thickness": 50,
            "Density": 4,
        },
        "fft_frequency": [1, 3, 6, 11],
        "fft_magnitude": [0.1, 0.3, 0.5, 0.2],
    }
    first = study.encode(row)
    row["metadata"].update(Thickness=999, Density=999, Sample_ID="another")
    assert study.encode(row) == first


def test_care_loader_excludes_status_and_obeys_partitions(tmp_path, monkeypatch):
    study = module("study_care", monkeypatch)
    path = tmp_path / "example.csv"
    path.write_text(
        "time_stamp;asset_id;id;train_test;status_type_id;sensor_0_avg;sensor_0_max\n2025-01-01 00:00:00;1;0;train;0;2;999\n2025-01-01 00:10:00;1;1;prediction;4;3;999\n"
    )
    times, x, train, normal, _ = study.load(path)
    assert x.tolist() == [[2], [3]] and train.tolist() == [True, False]
    assert normal.tolist() == [True, False] and times[1] > times[0]
