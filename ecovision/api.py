"""Local-first inspection API and dashboard. No public authentication included."""
import base64, logging, os
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .detection import ROOT, MAX_BYTES, InputError, YoloDetector, decode_image, annotate, encode_jpeg
from .history import History

log = logging.getLogger(__name__)

def create_app(detector=None, history=None):
    app = FastAPI(title='EcoVision AI', version='1.0.0')
    detector = detector if detector is not None else YoloDetector(os.getenv('ECOVISION_WEIGHTS', str(ROOT/'models/best.pt')), os.getenv('ECOVISION_DEVICE'))
    history = history if history is not None else History(os.getenv('ECOVISION_DB', str(ROOT/'runtime/inspections.sqlite3')))
    static = Path(__file__).parent/'static'
    app.mount('/static', StaticFiles(directory=static), name='static')
    @app.get('/', include_in_schema=False)
    def dashboard():
        return FileResponse(static/'index.html')
    @app.get('/api/health')
    def health():
        return {'status':'ok','model_loaded':getattr(detector,'_model',None) is not None,'note':'Service health is not model readiness.'}
    @app.post('/api/predict')
    async def predict(file: UploadFile=File(...), conf: float=Form(.25), iou: float=Form(.45), imgsz: int=Form(640)):
        try:
            raw = await file.read(MAX_BYTES+1)
            def run():
                rgb=decode_image(raw)
                pred=detector.predict(rgb,conf=conf,iou=iou,imgsz=imgsz)
                result=pred.to_dict()
                image=encode_jpeg(annotate(rgb,pred))
                result['id']=history.add('image',result)
                result['annotated_image']='data:image/jpeg;base64,'+base64.b64encode(image).decode('ascii')
                return result
            return await run_in_threadpool(run)
        except InputError as exc:
            raise HTTPException(400,str(exc)) from exc
        except Exception as exc:
            log.exception('Inference failed')
            raise HTTPException(503,'Inference unavailable. Check local model and server logs.') from exc
        finally:
            await file.close()
    @app.get('/api/history')
    def recent(limit: int=100):
        return history.recent(limit)
    @app.get('/api/summary')
    def summary():
        return history.summary()
    @app.get('/api/history.csv')
    def export():
        return Response(history.export(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=inspections.csv'})
    return app

app=create_app()
