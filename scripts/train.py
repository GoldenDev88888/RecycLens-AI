"""Configurable YOLO training, evaluation and optional export."""
import argparse, json, tempfile
import yaml
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data.yaml');p.add_argument('--weights',default='yolov8n.pt');p.add_argument('--epochs',type=int,default=50);p.add_argument('--imgsz',type=int,default=640);p.add_argument('--batch',type=int,default=16);p.add_argument('--device',default=None);p.add_argument('--seed',type=int,default=42);p.add_argument('--workers',type=int,default=2);p.add_argument('--project',default='runs/detect');p.add_argument('--name',default='ecovision');p.add_argument('--evaluate-only',action='store_true');p.add_argument('--export-onnx',action='store_true');a=p.parse_args()
    if not Path(a.data).is_file():p.error('Dataset YAML not found.')
    if min(a.epochs,a.imgsz,a.batch)<1 or a.workers<0:p.error('Invalid training settings.')
    from ultralytics import YOLO
    config_path=Path(a.data).resolve(); config=yaml.safe_load(config_path.read_text(encoding='utf-8')); root=Path(config.get('path','.')); config['path']=str((config_path.parent/root).resolve());
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'dataset.yaml'; path.write_text(yaml.safe_dump(config),encoding='utf-8'); run(a,str(path),YOLO)

def run(a,data,YOLO):
    model=YOLO(a.weights);opts={'device':a.device} if a.device else {}
    if not a.evaluate_only:
        model.train(data=data,epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,seed=a.seed,workers=a.workers,project=a.project,name=a.name,**opts)
        best=Path(model.trainer.best)
        if not best.is_file():raise RuntimeError('Training produced no best checkpoint.')
        model=YOLO(str(best))
    result=model.val(data=data,split='test',imgsz=a.imgsz,**opts)
    print(json.dumps({'test_metrics':{k:float(v) for k,v in result.results_dict.items()}},indent=2))
    if a.export_onnx: print('ONNX export:',model.export(format='onnx',imgsz=a.imgsz,**opts))
if __name__=='__main__':main()
