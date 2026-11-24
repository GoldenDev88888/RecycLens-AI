"""Validate and safely re-split a YOLO dataset without modifying originals."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecovision.dataset import audit, split_dataset

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',default='data');p.add_argument('--output',default='data_balanced');p.add_argument('--seed',type=int,default=42);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:
        _,report=audit(a.source);print(json.dumps(report,indent=2));return int(bool(report['errors']))
    try: print(json.dumps(split_dataset(a.source,a.output,a.seed),indent=2));return 0
    except ValueError as exc: p.error(str(exc))
if __name__=='__main__': raise SystemExit(main())
