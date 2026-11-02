"""VisA candle 图像异常和低分辨率定位验证；不是原版 PatchCore 复现。"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.inference import cpu_session
from phm_core.monitoring import calibrated_threshold
from phm_core.vision import PatchEncoder, PatchMemoryScore, kcenter_memory

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/visa"


def image_tensor(path, size=224):
    with Image.open(path) as image:
        return (
            np.asarray(
                image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                dtype="float32",
            ).transpose(2, 0, 1)
            / 255
        )


def run(profile="high-resolution"):
    torch.set_num_threads(4)
    size, dimensions, per_image, bank_size = (
        (448, 128, 32, 1024) if profile == "high-resolution" else (224, 64, 16, 256)
    )
    grid = size // 16
    with (RAW / "1cls.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["object"] == "candle"]
    official_train = [r for r in rows if r["split"] == "train"]
    official_test = [r for r in rows if r["split"] == "test"]
    ids = np.random.default_rng(42).permutation(len(official_train))
    split = {
        "train": [official_train[i] for i in ids[200:]],
        "validation": [official_train[i] for i in ids[:100]],
        "calibration": [official_train[i] for i in ids[100:200]],
        "test": official_test,
    }
    if any(r["label"] != "normal" for r in official_train):
        raise ValueError("一类建模不接受异常训练图片")
    run_id, out = new_run(ROOT / "outputs", "vision-study")
    protocol = {
        "profile": profile,
        "analysis_status": "resolution/memory expansion added after inspecting initial224 test; exploratory, not new blind data",
        "data_inventory_sha256": hashlib.sha256(
            (RAW / "candle-source.json").read_bytes()
        ).hexdigest(),
        "split_sha256": hashlib.sha256((RAW / "1cls.csv").read_bytes()).hexdigest(),
        "pretrained_source": json.loads(
            (ROOT / "data/raw/resnet18-weights/source.json").read_text()
        ),
        "dataset": "VisA candle, official 1cls split",
        "source": "https://github.com/amazon-science/spot-diff",
        "license": "CC BY 4.0",
        "split": {k: [r["image"] for r in v] for k, v in split.items()},
        "method": "PatchCore-style bounded patch memory, not original implementation or paper reproduction",
        "encoder": f"official ImageNet ResNet18 frozen layer2/layer3, pooled to{grid}x{grid}, fixed Gaussian projection384to{dimensions}",
        "preprocess": f"full image bilinear resize{size}x{size}; RGB[0,1]; ImageNet normalization embedded; no central crop",
        "memory": f"{per_image} random patches per training image, seed42; greedy kcenter{bank_size} in projected space",
        "threshold": "healthy calibration100 at alpha.01, no anomalous validation or test tuning",
        "scope": "single object class, not CCD/defect or machine-failure certification",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    source = capture_source(
        ROOT / "code/python", out / "train-source.zip", "study_vision.py"
    )
    encoder = PatchEncoder(dimensions).eval()
    encoder.load_pretrained(ROOT / "data/raw/resnet18-weights/resnet18-f37072fd.pth")
    features = {}
    with torch.inference_mode():
        for name, records in split.items():
            features[name] = np.stack(
                [
                    encoder(
                        torch.from_numpy(image_tensor(RAW / r["image"], size)[None])
                    )[0].numpy()
                    for r in records
                ]
            )
            print(f"encoded {name}: {len(records)}", flush=True)
    rng = np.random.default_rng(42)
    patches = np.concatenate(
        [
            f.reshape(-1, dimensions)[rng.choice(grid * grid, per_image, replace=False)]
            for f in features["train"]
        ]
    )
    memory = kcenter_memory(patches, bank_size)
    model = PatchMemoryScore(encoder, memory).eval()

    def distances(values):
        flat = torch.from_numpy(values)
        bank = torch.from_numpy(memory)
        with torch.inference_mode():
            d = (
                (
                    flat.square().sum(-1, keepdim=True)
                    + bank.square().sum(-1)
                    - 2 * (flat @ bank.T)
                )
                .clamp_min(0)
                .amin(-1)
                .sqrt()
            )
        return d.numpy()

    maps = {
        k: np.concatenate(
            [
                distances(batch)
                for batch in np.array_split(v, max(1, int(np.ceil(len(v) / 8))))
            ]
        )
        for k, v in features.items()
        if k != "train"
    }
    scores = {k: v.max((1, 2)) for k, v in maps.items()}
    threshold = calibrated_threshold(scores["calibration"])
    truth = np.array([r["label"] != "normal" for r in split["test"]])
    predicted = scores["test"] > threshold
    # 像素标签仅参与评估，按 nearest 缩到输出网格，结果不冒充原分辨率分割。
    masks = []
    for r in split["test"]:
        if r["label"] == "normal":
            masks.append(np.zeros((grid, grid), dtype=bool))
        else:
            with Image.open(RAW / r["mask"]) as image:
                masks.append(
                    np.asarray(
                        image.convert("L").resize(
                            (grid, grid), Image.Resampling.NEAREST
                        )
                    )
                    > 0
                )
    metrics = {
        "image_auroc": float(roc_auc_score(truth, scores["test"])),
        "image_ap": float(average_precision_score(truth, scores["test"])),
        "false_positives": int((predicted & ~truth).sum()),
        "normal_images": int((~truth).sum()),
        "false_negatives": int((~predicted & truth).sum()),
        "anomalous_images": int(truth.sum()),
        "pixel_auroc": float(
            roc_auc_score(np.array(masks).ravel(), maps["test"].ravel())
        ),
    }
    # 全局表征对照：相同深层特征，仅比较整图均值，不使用局部记忆。
    global_train = features["train"].mean((1, 2))
    center, scale = global_train.mean(0), np.maximum(global_train.std(0), 1e-4)
    baseline = {
        k: np.abs((v.mean((1, 2)) - center) / scale).mean(1)
        for k, v in features.items()
        if k != "train"
    }
    base_threshold = calibrated_threshold(baseline["calibration"])
    base_pred = baseline["test"] > base_threshold
    baseline_metrics = {
        "image_auroc": float(roc_auc_score(truth, baseline["test"])),
        "image_ap": float(average_precision_score(truth, baseline["test"])),
        "false_positives": int((base_pred & ~truth).sum()),
        "false_negatives": int((~base_pred & truth).sum()),
    }
    operating_points = []
    for alpha in (0.01, 0.05, 0.1):
        cut = calibrated_threshold(scores["calibration"], alpha)
        flag = scores["test"] > cut
        operating_points.append(
            {
                "calibration_alpha": alpha,
                "threshold": cut,
                "false_positives": int((flag & ~truth).sum()),
                "false_negatives": int((~flag & truth).sum()),
            }
        )
    np.savez_compressed(
        out / "scores.npz",
        calibration=scores["calibration"],
        test=scores["test"],
        truth=truth,
        memory=memory,
    )
    raw = np.stack([image_tensor(RAW / r["image"], size) for r in split["test"][:16]])
    torch.onnx.export(
        model,
        torch.from_numpy(raw[:1]),
        str(out / "model.onnx"),
        input_names=["image"],
        output_names=["score", "heatmap"],
        dynamic_axes={
            "image": {0: "batch"},
            "score": {0: "batch"},
            "heatmap": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    session = cpu_session(out / "model.onnx")
    observed = [session.run(None, {"image": x}) for x in np.array_split(raw, 4)]
    with torch.inference_mode():
        expected = model(torch.from_numpy(raw))
    errors = {}
    for i, name in enumerate(("score", "heatmap")):
        actual = np.concatenate([v[i] for v in observed])
        np.testing.assert_allclose(actual, expected[i].numpy(), atol=1e-3, rtol=1e-4)
        errors[name] = float(np.max(np.abs(actual - expected[i].numpy())))
    times = []
    for i in range(25):
        start = time.perf_counter()
        session.run(None, {"image": raw[:1]})
        if i >= 5:
            times.append((time.perf_counter() - start) * 1000)
    (out / "previews").mkdir()
    records = []
    color_scale = float(np.quantile(maps["calibration"], 0.99))
    for i, r in enumerate(split["test"]):
        with Image.open(RAW / r["image"]) as image:
            image.convert("RGB").resize((336, 336)).save(out / "previews" / f"{i}.jpg")
        level = np.clip(maps["test"][i] / color_scale, 0, 1)
        heat = np.stack([level, 1 - np.abs(level * 2 - 1), 1 - level], -1)
        Image.fromarray((heat * 255).astype("uint8")).resize(
            (336, 336), Image.Resampling.BILINEAR
        ).save(out / "previews" / f"{i}-heat.png")
        records.append(
            {
                "id": i,
                "file": r["image"],
                "truth": bool(truth[i]),
                "flagged": bool(predicted[i]),
                "score": float(scores["test"][i]),
            }
        )
    manifest = {
        "schemaVersion": 1,
        "task": "visual-anomaly",
        "version": run_id,
        "file": "model.onnx",
        "sha256": hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest(),
        "input": {
            "name": "image",
            "shape": [None, 3, size, size],
            "layout": "BCHW",
            "range": [0, 1],
            "color": "RGB",
            "resize": "bilinear, full image",
            "normalizationEmbedded": True,
        },
        "outputs": {"score": 0, "heatmap": 1},
        "threshold": threshold,
        "maxBatch": 4,
        "scope": protocol["scope"],
        "parity": errors,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(
        out / "parity.npz",
        raw=raw,
        score=expected[0].numpy(),
        heatmap=expected[1].numpy(),
    )
    report = {
        "run_id": run_id,
        "source_hash": source,
        "protocol": protocol,
        "counts": {k: len(v) for k, v in split.items()},
        "output_grid": grid,
        "operating_points": operating_points,
        "metrics": metrics,
        "baseline": baseline_metrics,
        "threshold": threshold,
        "latency_ms": {
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
        },
        "records": records,
        "manifest": manifest,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    with ZipFile(out / "model-bundle.zip", "w", ZIP_DEFLATED) as archive:
        for name in ("model.onnx", "manifest.json", "protocol.json"):
            archive.write(out / name, name)
        archive.write(
            ROOT / "code/licenses/torchvision-BSD-3-Clause.txt",
            "licenses/torchvision-BSD-3-Clause.txt",
        )
        archive.write(ROOT / "code/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")
    publish_run(ROOT / "outputs", "vision-study", run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "metrics": metrics,
                "baseline": baseline_metrics,
                "parity": errors,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=["initial", "high-resolution"], default="high-resolution"
    )
    run(parser.parse_args().profile)
