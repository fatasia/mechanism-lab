"""小窗口推理服务入口；导入NumPy前固定BLAS线程，避免进程提交内存膨胀。"""

import argparse
import os


def configure_threads(threads):
    if threads < 1:
        raise ValueError("BLAS线程数必须为正")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--blas-threads", type=int, default=1)
    args = parser.parse_args()
    configure_threads(args.blas_threads)
    import uvicorn

    # 自带访问日志包含原始查询串；改用应用层固定字段JSON日志。
    uvicorn.run("phm_demo.api:app", host="127.0.0.1", port=args.port, access_log=False)
