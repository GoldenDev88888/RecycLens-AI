"""Bounded video or webcam processing; counts are frame-level unless track IDs exist."""
from collections import Counter
from pathlib import Path
from time import perf_counter
import cv2
from .detection import annotate, InputError

def process_video(source, output, detector, *, conf=.25,iou=.45,imgsz=640,track=False,max_frames=1800,preview=False):
    if max_frames < 1:
        raise InputError('max_frames must be positive')
    output=Path(output).resolve()
    if not isinstance(source,int) and Path(source).resolve()==output:
        raise InputError('Output cannot overwrite source video.')
    if output.exists():
        raise InputError('Output exists; choose a new filename.')
    output.parent.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(source)
    writer=None; frames=0; counts=Counter(); unique={}; lat=0; start=perf_counter(); stopped='end_of_stream'
    try:
        if not cap.isOpened():
            raise InputError('Unable to open video or camera.')
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not 0<fps<=240: fps=25.0
        while frames<max_frames:
            ok,bgr=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            pred=detector.predict(rgb,conf=conf,iou=iou,imgsz=imgsz,track=track)
            lat+=pred.latency_ms; counts.update(pred.counts)
            for d in pred.detections:
                if d.track_id is not None: unique.setdefault(d.track_id,Counter())[d.label]+=1
            rendered=cv2.cvtColor(annotate(rgb,pred),cv2.COLOR_RGB2BGR)
            if writer is None:
                h,w=rendered.shape[:2]
                writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                if not writer.isOpened(): raise RuntimeError('Video encoder unavailable.')
            writer.write(rendered); frames+=1
            if preview:
                cv2.imshow('EcoVision AI (Q to stop)',rendered)
                if cv2.waitKey(1)&255==ord('q'):
                    stopped='user_stop'; break
        else: stopped='frame_limit'
        if frames==0: raise InputError('No readable video frames.')
    except BaseException:
        if writer is not None: writer.release(); writer=None
        output.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        if writer is not None: writer.release()
        if preview: cv2.destroyAllWindows()
    elapsed=perf_counter()-start
    return {'frames':frames,'counts':dict(counts),'total':sum(counts.values()),'count_semantics':'frame detections, not unique physical objects',
            'tracked_objects':dict(Counter(v.most_common(1)[0][0] for v in unique.values())) if track else None,
            'latency_ms':round(lat/frames,2),'processing_fps':round(frames/max(elapsed,1e-9),2),'stop_reason':stopped,'output':str(output)}
