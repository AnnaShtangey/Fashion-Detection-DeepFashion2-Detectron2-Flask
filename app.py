from flask import Flask,render_template,url_for,request, send_from_directory, redirect, flash
from werkzeug.utils import secure_filename
import detectron2
import numpy as np
import shutil
import os
import cv2
import random
from matplotlib import colors, pyplot as plt
import json
import pycocotools
import requests
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer 
from detectron2.engine import DefaultPredictor


#UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = 'some_secret'


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, "static")

 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
      


           
@app.route('/', methods=['GET', 'POST'])
def upload_file():
		
    if request.method == 'POST':
		
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], filename),os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg') )
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Загружено. Нажмите Detect и подождите.')
            return redirect(url_for('upload_file'))
    
    
    
    return render_template('index.html')
    
    



app.route('/<filename>')
def send_image(filename):
    return send_from_directory("images", filename )
	


@app.route('/predict', methods= ['POST'])
def predict():
  
      
  flag=0
  for i in range(len(DatasetCatalog.list())):
	  if DatasetCatalog.list()[i]=="deep_fashion_test":
		  flag=1	  
  if flag==0:
	  register_coco_instances("deep_fashion_test", {}, os.path.join(APP_ROOT, 'data/deepfashion2/annotation_test.json'), os.path.join(APP_ROOT, "static"))
      
  
  deep_fashion_test_metadata = MetadataCatalog.get("deep_fashion_test")

  
  deep_fashion_test_dict=DatasetCatalog.get("deep_fashion_test")


  
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
  cfg.DATASETS.TEST = ("deep_fashion_test",)
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13  
  
  cfg.MODEL.DEVICE = "cpu"
  model= build_model(cfg)
  
  cfg.MODEL.WEIGHTS = (os.path.join(APP_ROOT,'data/deepfashion2/model_final_clear.pth'))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model

  predictor = DefaultPredictor(cfg)

  
  im= cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg"))
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1], deep_fashion_test_metadata, scale=0.5)
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2.imwrite((os.path.join(app.config['UPLOAD_FOLDER'], "prediction.jpg")),v.get_image()[:, :, ::-1])
  flash('Успешно! Нажмите Show для просмотра. Нажмите Clear перед следующим использованием детектора.')
  return render_template('index.html',prediction = v.get_image()[:, :, ::-1])


@app.route('/about')
def about():
	return render_template('about.html')
	
@app.route('/show')
def show():
	return render_template('show.html')
	

	

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


if __name__ == '__main__':
  app.run(debug=True)

