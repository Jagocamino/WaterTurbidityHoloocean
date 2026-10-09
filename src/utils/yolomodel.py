import cv2
from PIL import Image
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks # gain access to utility funcitons for models
import sys
import torch
import torchvision.ops as ops
from utils.nms import use_nms
from . import custom_modules



def register_custom_modules():              #   module names must be registered in order to work
    sys.modules["custom_modules"]=custom_modules
    custom_classes = ["SimAM","LEF","FEM","C2fLEFEM","AMC2fLEFEM","AMRFB","SimSPPF"]
    for singular_class in custom_classes:
        if hasattr(custom_modules, singular_class):
            added_class = getattr(custom_modules, singular_class)
            setattr(tasks, singular_class, added_class)
            tasks.__dict__[singular_class] = added_class    #to be indexed

class YoloModel:

    conf = 0.01     # minimum precision degree of the object 
    imgsz = 320     # image size
    iou = 0.1       # overlapping boxes forgiveness %

    def init_yolo_model(self):

        
        register_custom_modules()

        # ----- Load a YOLO model -----

        #self.model = YOLO("./rete_custom_trained/model_5NLE/my_model.pt")
        self.model = YOLO("./rete_custom_trained/model_26/my_model.pt")
        #self.model = YOLO("./rete_custom_trained/model_2/my_model.pt")
        #self.model = YOLO("./rete_custom_trained/model_shapes/my_model.pt")

        self.class_names = self.model.names

    def detect(self, image):
        results  = self.model.predict(source=image, save=True, imgsz=self.imgsz, conf=self.conf)
        for result in results:
            bbox_list = result.boxes.xyxy.tolist()          # bounding boxes all objects, you can also get xywh with boxes.xywh
            clss_list = result.boxes.cls.int().tolist()     # class index all objects
            conf_list = result.boxes.conf.tolist()          # confidence list all objects
            for box, cls, conf in zip(bbox_list, clss_list, conf_list):  # Iterate over each bbox, cls and conf, zip puts the corr elem in same row
                print(f"Bounding box: {box}, Class index: {cls}, Class name: {self.class_names[cls]}, Confidence: {conf}")
                # ... any downstream task.
    
    def detect_nosave(self, image): # returns the image with bounding boxes and bounding boxes themself
       
        results  = self.model.predict(source=image, save=False, imgsz=self.imgsz, conf=self.conf, verbose=False)

        results = use_nms(results, self.conf, self.iou) # remove comment to apply nms filter

        for result in results:
            bbox_list = result.boxes.xyxy.tolist()          # bounding boxes all objects, you can also get xywh with boxes.xywh
            clss_list = result.boxes.cls.int().tolist()     # class index all objects
            conf_list = result.boxes.conf.tolist()          # confidence list all objects
        img_w_bound_box = result.plot()
        return img_w_bound_box, results
            

