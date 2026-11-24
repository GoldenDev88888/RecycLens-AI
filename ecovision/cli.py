import argparse, json, os
from pathlib import Path
from .detection import ROOT, YoloDetector, decode_image, annotate, encode_jpeg
from .history import History
from .video import process_video

def main():
    parser=argparse.ArgumentParser(description='EcoVision image and video inspection')
    sub=parser.add_subparsers(dest='command',required=True)
    for kind in ('image','video'):
        p=sub.add_parser(kind); p.add_argument('source'); p.add_argument('--output',required=True)
        p.add_argument('--weights',default=str(ROOT/'models/best.pt')); p.add_argument('--device',default=None)
        p.add_argument('--conf',type=float,default=.25); p.add_argument('--iou',type=float,default=.45)
        p.add_argument('--imgsz',type=int,default=640); p.add_argument('--db',default=str(ROOT/'runtime/inspections.sqlite3'))
        if kind=='video':
            p.add_argument('--camera',action='store_true',help='Interpret source as a local camera index')
            p.add_argument('--track',action='store_true'); p.add_argument('--max-frames',type=int,default=1800); p.add_argument('--preview',action='store_true')
    p=sub.add_parser('serve'); p.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    if args.command=='serve':
        import uvicorn
        uvicorn.run('ecovision.api:app',host='127.0.0.1',port=args.port); return
    detector=YoloDetector(args.weights,args.device)
    opts=dict(conf=args.conf,iou=args.iou,imgsz=args.imgsz)
    if args.command=='image':
        source=Path(args.source); output=Path(args.output)
        if output.exists(): parser.error('Output already exists')
        if output.suffix.lower() not in ('.jpg','.jpeg'): parser.error('Output must be JPEG')
        with source.open('rb') as f: rgb=decode_image(f.read(12*1024*1024+1))
        pred=detector.predict(rgb,**opts); result=pred.to_dict()
        output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(encode_jpeg(annotate(rgb,pred)))
    else:
        if Path(args.output).suffix.lower()!='.mp4': parser.error('Output must be MP4')
        result=process_video(int(args.source) if args.camera else args.source,args.output,detector,
            **opts,track=args.track,max_frames=args.max_frames,preview=args.preview)
    result['id']=History(args.db).add(args.command,result)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
