"""导出公开试点指标与抽样协议；原始时序、模型、密钥和运行库不发布。"""

import argparse
import json
from pathlib import Path

from public_data import sha256


def run(args):
    args.destination.mkdir(parents=True,exist_ok=True)
    for name,path in [("care-holdout",args.care),("electrode-uncertainty",args.electrode),("inferred-condition",args.condition)]:
        source=path/"metrics.json"
        report=json.loads(source.read_text(encoding="utf-8"))
        report["metrics_sha256"]=sha256(source)
        if name=="care-holdout":
            report["license"]="CC-BY-SA-4.0"
            report["attribution"]="Christian Gück, Cyriana Roelofs; https://zenodo.org/records/15846963; subset selection, feature processing and custom evaluation by Mechanism Lab"
            report["editorial_correction"]="Original limitations text used a 12-asset placeholder; actual frozen set contains 9 assets. No metric changed."
            report["limitations"]=report["limitations"].replace("12 asset",f"{len(report['cases'])} asset")
        (args.destination/(name+".json")).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("care","electrode","condition"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--destination",type=Path,default=Path(__file__).resolve().parents[1]/"benchmarks")
    run(parser.parse_args())
