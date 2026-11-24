"""Optional Streamlit frontend using the shared detection service."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from ecovision.detection import YoloDetector, decode_image, annotate, encode_jpeg
from ecovision.history import History
st.set_page_config(page_title='EcoVision AI',layout='wide')
st.title('EcoVision AI | Waste inspection')
@st.cache_resource
def detector():return YoloDetector()
@st.cache_resource
def history():return History(Path(__file__).resolve().parents[1]/'runtime/inspections.sqlite3')
conf=st.sidebar.slider('Confidence',.01,1.,.25,.01);iou=st.sidebar.slider('IoU',.01,1.,.45,.01)
file=st.file_uploader('Upload an image',type=['jpg','jpeg','png','webp'])
if file and st.button('Run inspection'):
    try:
        rgb=decode_image(file.getvalue());pred=detector().predict(rgb,conf=conf,iou=iou);rendered=annotate(rgb,pred);history().add('image',pred.to_dict());st.session_state['result']=(rgb,rendered,pred.to_dict())
    except Exception as exc:st.error(str(exc))
if 'result' in st.session_state:
    rgb,rendered,result=st.session_state['result'];a,b=st.columns(2);a.image(rgb,caption='Original');b.image(rendered,caption='Detected materials');st.write(result['counts']);st.metric('Inference latency (ms)',result['latency_ms']);st.download_button('Download image',encode_jpeg(rendered),'detection.jpg','image/jpeg')
st.subheader('Recent inspections');st.dataframe(history().recent());st.download_button('Export history',history().export(),'inspections.csv','text/csv')
