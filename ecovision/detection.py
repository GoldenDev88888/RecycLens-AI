"""Validated image input, pluggable inference, and consistent output rendering."""
from collections import Counter
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
import math
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 16_000_000
COLORS = {'CARDBOARD': (181, 121, 69), 'METAL': (74, 105, 166), 'PAPER': (43, 148, 118), 'PLASTIC': (199, 98, 68)}

class InputError(ValueError):
    pass

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None

@dataclass
class Prediction:
    detections: list[Detection]
    latency_ms: float
    width: int
    height: int
    @property
    def counts(self):
        return dict(sorted(Counter(d.label for d in self.detections).items()))
    def to_dict(self):
        return {'detections': [asdict(d) for d in self.detections], 'counts': self.counts,
                'total': len(self.detections), 'latency_ms': round(self.latency_ms, 2),
                'width': self.width, 'height': self.height}

def decode_image(raw: bytes):
    if not raw or len(raw) > MAX_BYTES:
        raise InputError('Image must be between 1 byte and 12 MiB.')
    try:
        with Image.open(BytesIO(raw)) as im:
            if im.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise InputError('Use a JPEG, PNG or WebP image.')
            if im.width * im.height > MAX_PIXELS:
                raise InputError('Image exceeds the 16 megapixel limit.')
            return np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InputError('Invalid or oversized image.') from exc

def validate_settings(conf, iou, imgsz):
    if not math.isfinite(conf) or not 0.01 <= conf <= 1:
        raise InputError('Confidence must be between 0.01 and 1.')
    if not math.isfinite(iou) or not 0.01 <= iou <= 1:
        raise InputError('IoU must be between 0.01 and 1.')
    if imgsz not in (320, 416, 640, 960):
        raise InputError('Image size must be 320, 416, 640 or 960.')

class YoloDetector:
    def __init__(self, weights=ROOT / 'models/best.pt', device=None):
        self.weights = Path(weights).resolve()
        self.device = device
        self._model = None
        self._lock = Lock()
    def predict(self, rgb, conf=.25, iou=.45, imgsz=640, track=False):
        validate_settings(conf, iou, imgsz)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise InputError('Expected an RGB uint8 image.')
        with self._lock:
            if self._model is None:
                if not self.weights.is_file():
                    raise RuntimeError(f'Model weights not found: {self.weights}')
                from ultralytics import YOLO
                self._model = YOLO(str(self.weights))
            options = dict(source=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=conf, iou=iou,
                           imgsz=imgsz, verbose=False, save=False)
            if self.device is not None:
                options['device'] = self.device
            start = perf_counter()
            if track:
                result = self._model.track(**options, persist=True, tracker='bytetrack.yaml')[0]
            else:
                result = self._model.predict(**options)[0]
            boxes = result.boxes
            detections = []
            h, w = rgb.shape[:2]
            if boxes is not None:
                ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                for xy, score, cls, tid in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), ids):
                    x1,y1,x2,y2 = xy
                    detections.append(Detection(str(result.names[int(cls)]).upper(), float(score),
                        (max(0,min(w,x1)),max(0,min(h,y1)),max(0,min(w,x2)),max(0,min(h,y2))),
                        int(tid) if tid is not None else None))
            return Prediction(detections, (perf_counter()-start)*1000, w, h)

def annotate(rgb, prediction):
    out = rgb.copy()
    h,w = out.shape[:2]
    for d in prediction.detections:
        x1,y1,x2,y2 = [int(v) for v in d.xyxy]
        x1,x2 = max(0,min(w-1,x1)),max(0,min(w-1,x2))
        y1,y2 = max(0,min(h-1,y1)),max(0,min(h-1,y2))
        color = COLORS.get(d.label, (70, 130, 180))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        caption = f'{d.label} {d.confidence:.0%}' + (f' #{d.track_id}' if d.track_id is not None else '')
        (tw,th),base = cv2.getTextSize(caption,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
        top = max(0,y1-th-base-6)
        cv2.rectangle(out,(x1,top),(min(w-1,x1+tw+6),min(h-1,top+th+base+6)),color,-1)
        cv2.putText(out,caption,(x1+3,min(h-1,top+th+2)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return out

def encode_jpeg(rgb):
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer,format='JPEG',quality=88)
    return buffer.getvalue()
