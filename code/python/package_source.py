"""白名单打包源码；不包含数据、模型、私有文档或本机依赖。"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "secrets",
}
EXTENSIONS = {
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".js",
    ".json",
    ".css",
    ".html",
    ".ps1",
    ".md",
    ".txt",
    ".svg",
}


def package(output: Path | None = None) -> Path:
    output = output or ROOT / "outputs/mechanism-lab-source.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = {}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "code").rglob("*")):
            if not path.is_file() or path.suffix not in EXTENSIONS:
                continue
            if any(
                part in EXCLUDED or part.endswith(".egg-info") for part in path.parts
            ):
                continue
            relative = path.relative_to(ROOT).as_posix()
            inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, relative)
        for name in (".gitignore", "README.md", "LICENSE"):
            path = ROOT / name
            if not path.is_file():
                continue
            inventory[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, name)
        archive.writestr("source-manifest.json", json.dumps(inventory, indent=2))
    print(
        json.dumps(
            {
                "source_files": len(inventory),
                "archive": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package(args.output.resolve() if args.output else None)
