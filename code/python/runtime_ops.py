"""显式运行备份与基础磁盘监控；不安装计划任务，不向网络发送数据。"""

import argparse
import math
import time
from pathlib import Path

from phm_runtime.backups import create_backup
from phm_runtime.observability import json_event, storage_snapshot
from phm_runtime.store import StateStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument(
        "--interval", type=float, default=0, help="0执行一次；正数为循环间隔秒数"
    )
    parser.add_argument("--keep", type=int, default=96)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 0 or args.keep < 1:
        parser.error("interval不能为负，keep必须为正")
    if not args.database.is_file():
        parser.error("运行数据库不存在；不创建空库冒充备份")
    store = StateStore(args.database)
    while True:
        try:
            result = create_backup(store, args.backup_dir, args.keep)
            print(
                json_event(
                    "backup_complete", **result, storage=storage_snapshot(store.path)
                ),
                flush=True,
            )
        except Exception as error:
            print(
                json_event("backup_failed", error_type=type(error).__name__), flush=True
            )
            if not args.interval:
                raise SystemExit(1) from None
        if not args.interval:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
