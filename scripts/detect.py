"""Stream predictions from a file/folder with the existing YOLO workflow."""
import argparse
from collections import Counter
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',default='models/best.pt');parser.add_argument('--source',default='data/images/test');parser.add_argument('--conf',type=float,default=.25);parser.add_argument('--iou',type=float,default=.45);parser.add_argument('--imgsz',type=int,default=640);parser.add_argument('--device');parser.add_argument('--save-dir',default='results/sample_predictions');a=parser.parse_args()
    if not 0<a.conf<=1 or not 0<a.iou<=1:parser.error('Thresholds must be in (0, 1].')
    if not Path(a.weights).is_file():parser.error('Local model weights not found.')
    if not Path(a.source).exists():parser.error('Local image, folder or video not found.')
    from ultralytics import YOLO
    model=YOLO(a.weights);opts={'device':a.device} if a.device else {};counts=Counter();processed=0;output=None
    for result in model.predict(source=a.source,conf=a.conf,iou=a.iou,imgsz=a.imgsz,save=True,project=a.save_dir,name='predictions',exist_ok=False,stream=True,**opts):
        processed+=1;output=result.save_dir
        if result.boxes is not None:counts.update(result.names[int(i)] for i in result.boxes.cls.tolist())
    print(f'Processed {processed} image(s)/frame(s). Output: {output}')
    print('Detections (frame-level for videos):',dict(counts))
if __name__=='__main__':main()
