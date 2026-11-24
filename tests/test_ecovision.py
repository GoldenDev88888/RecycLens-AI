import asyncio, io
from pathlib import Path
import cv2, httpx, numpy as np, pytest
from PIL import Image
from ecovision.detection import Detection, Prediction, decode_image, annotate, InputError, validate_settings, YoloDetector
from ecovision.history import History
from ecovision.api import create_app
from ecovision.video import process_video
from ecovision.dataset import audit, split_dataset

class FixtureDetector:
    def predict(self,rgb,**kw):
        validate_settings(kw.get('conf',.25),kw.get('iou',.45),kw.get('imgsz',640))
        h,w=rgb.shape[:2]
        return Prediction([Detection('PLASTIC',.9,(5,5,25,25),1 if kw.get('track') else None)],4.0,w,h)

def img_bytes():
    f=io.BytesIO();Image.new('RGB',(64,48),(160,140,120)).save(f,format='PNG');return f.getvalue()

def test_decode():
    assert decode_image(img_bytes()).shape==(48,64,3)
@pytest.mark.parametrize('raw',[b'',b'not an image',b'a'*(12*1024*1024+1)])
def test_invalid_image(raw):
    with pytest.raises(InputError):decode_image(raw)
@pytest.mark.parametrize('args',[(0,.45,640),(.25,float('nan'),640),(.25,.45,12)])
def test_settings(args):
    with pytest.raises(InputError):validate_settings(*args)
def test_render():
    rgb=decode_image(img_bytes());p=FixtureDetector().predict(rgb)
    assert p.counts=={'PLASTIC':1}; assert not np.array_equal(annotate(rgb,p),rgb)
def test_empty_prediction():
    assert Prediction([],1,10,10).to_dict()['total']==0

def test_history(tmp_path):
    db=History(tmp_path/'history.db');p=FixtureDetector().predict(decode_image(img_bytes())).to_dict()
    assert db.add('image',p)==1;db.add('image',p)
    assert db.summary()['counts']=={'PLASTIC':2};assert len(db.recent(1))==1
    assert len(db.export().splitlines())==3

def test_api(tmp_path):
    app=create_app(FixtureDetector(),History(tmp_path/'test.db'))
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
            assert (await c.get('/')).status_code==200
            assert (await c.get('/static/app.js')).status_code==200
            assert (await c.get('/api/health')).json()['status']=='ok'
            r=await c.post('/api/predict',files={'file':('image.png',img_bytes(),'image/png')})
            assert r.status_code==200,r.text;assert r.json()['total']==1;assert r.json()['annotated_image'].startswith('data:image/jpeg;base64,')
            assert (await c.get('/api/summary')).json()['inspections']==1
            assert (await c.get('/api/history')).json()[0]['counts']=={'PLASTIC':1}
            assert 'counts_json' in (await c.get('/api/history.csv')).text
            assert (await c.post('/api/predict',files={'file':('bad.png',b'bad')})).status_code==400
            assert (await c.post('/api/predict',files={'file':('a.png',img_bytes())},data={'conf':0})).status_code==400
    asyncio.run(check())

def test_api_model_failure(tmp_path):
    app=create_app(YoloDetector(tmp_path/'missing.pt'),History(tmp_path/'test.db'))
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
            r=await c.post('/api/predict',files={'file':('a.png',img_bytes())});assert r.status_code==503
            assert (await c.get('/api/summary')).json()['inspections']==0
    asyncio.run(check())

def test_video(tmp_path):
    source=tmp_path/'source.mp4';out=tmp_path/'out.mp4';w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,48))
    assert w.isOpened()
    for i in range(5):w.write(np.full((48,64,3),i*30,dtype=np.uint8))
    w.release();r=process_video(str(source),out,FixtureDetector(),track=True,max_frames=3)
    assert r['frames']==3;assert r['total']==3;assert r['tracked_objects']=={'PLASTIC':1};assert r['stop_reason']=='frame_limit'
    assert out.stat().st_size>0
    c=cv2.VideoCapture(str(out));assert int(c.get(cv2.CAP_PROP_FRAME_COUNT))==3;c.release()
    with pytest.raises(InputError):process_video(str(source),source,FixtureDetector())

def make_dataset(root,n=60):
    for s in ('train','val','test'):
        for d in ('images','labels'):(root/d/s).mkdir(parents=True)
    for i in range(n):
        s=('train','val','test')[i%3];name=f'img{i//3}'
        Image.new('RGB',(16,16),(i,0,0)).save(root/'images'/s/(name+'.png'))
        (root/'labels'/s/(name+'.txt')).write_text(f'{i%2} .5 .5 .5 .5\n')

def test_dataset_split(tmp_path):
    source=tmp_path/'data';target=tmp_path/'new';make_dataset(source)
    _,r=audit(source);assert not r['errors'];assert r['images']==60
    manifest=split_dataset(source,target);assert sum(manifest['split_counts'].values())==60
    _,r=audit(target);assert not r['errors'];assert r['images']==60
    assert audit(source)[1]['images']==60
    with pytest.raises(ValueError):split_dataset(source,target)

def test_dataset_invalid(tmp_path):
    root=tmp_path/'data';make_dataset(root,6)
    (root/'labels/train/img0.txt').write_text('9 .5 .5 .5 .5')
    assert audit(root)[1]['errors']

def test_duplicate_detection(tmp_path):
    root=tmp_path/'data';make_dataset(root,6)
    (root/'images/val/img0.png').write_bytes((root/'images/train/img0.png').read_bytes())
    assert any('Duplicate' in e for e in audit(root)[1]['errors'])
