"""白名单公开项目资料的可追溯检索；只返回原文片段，不生成故障结论。"""

import re
from pathlib import Path

DOCUMENTS = ("OPERATIONS.md", "RESULTS.md", "PUBLIC_BENCHMARKS.md", "PUBLIC_PILOT.md")


def terms(text):
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_-]{2,}", lowered))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", lowered):
        words.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return words


def search(directory: Path, query: str, limit=5):
    if not 1 <= limit <= 10 or not 1 <= len(query) <= 200:
        raise ValueError("查询须为1至200字符，最多10项")
    wanted = terms(query)
    if not wanted:
        return []
    results = []
    for name in DOCUMENTS:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for start in range(0, len(lines), 8):
            snippet = "\n".join(lines[start : start + 12])
            overlap = wanted & terms(snippet)
            if len(overlap) < min(2, len(wanted)):
                continue
            results.append(
                {
                    "document": name,
                    "line_start": start + 1,
                    "line_end": min(start + 12, len(lines)),
                    "matched_terms": sorted(overlap),
                    "text": snippet,
                    "source_url": f"https://github.com/fatasia/mechanism-lab/blob/main/code/{name}#L{start + 1}",
                }
            )
    return sorted(
        results,
        key=lambda r: (-len(r["matched_terms"]), r["document"], r["line_start"]),
    )[:limit]
