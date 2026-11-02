import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location(
    "package_source", Path(__file__).resolve().parents[1] / "package_source.py"
)
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def test_source_archive_excludes_private_material_and_tracks_entrypoint_hashes(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    files = {
        "code/python/example.py": "print('example')",
        "code/.git/config.json": "repository internals",
        "code/node_modules/dependency/index.js": "dependency",
        "code/python/.venv/installed.py": "installed package",
        "code/python/.env.local": "configuration fixture",
        "references/vendor.pdf": "private document fixture",
        "data/sample.json": "raw data fixture",
        "work/process.md": "process fixture",
        ".gitignore": "/data/\n/work/\n/references/\n",
        "README.md": "public entrypoint",
    }
    for name, content in files.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(packager, "ROOT", workspace)
    destination = packager.package(tmp_path / "release/source.zip")
    with ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            "code/python/example.py",
            ".gitignore",
            "README.md",
            "source-manifest.json",
        }
        manifest = json.loads(archive.read("source-manifest.json"))
        assert set(manifest) == {"code/python/example.py", ".gitignore", "README.md"}
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
