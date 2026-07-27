import cv2
from PIL import Image
from ultralytics import YOLO

class YoloModel:

    def init_yolo_model(self):
        # Load a pretrained YOLO model (recommended for training)
        # self.model = YOLO("yolo26n.pt")
        self.model = YOLO("./rete_custom_trained/model_2/my_model.pt")
        self.class_names = self.model.names

    def detect(self, image):
        results  = self.model.predict(source=image, save=True, imgsz=320)
        help(self.model.predict)
        print(results)
        for result in results:
            bbox_list = result.boxes.xyxy.tolist()          # bounding boxes all objects, you can also get xywh with boxes.xywh
            clss_list = result.boxes.cls.int().tolist()     # class index all objects
            conf_list = result.boxes.conf.tolist()          # confidence list all objects
            for box, cls, conf in zip(bbox_list, clss_list, conf_list):  # Iterate over each bbox, cls and conf
                print(f"Bounding box: {box}, Class index: {cls}, Class name: {self.class_names[cls]}, Confidence: {conf}")
                # ... any downstream task.
    
    def detect_nosave(self, image): # returns the image with bounding boxes and bounding boxes themself
       # conf= rappresenta il grado minimo di precisione dell'oggetto detectato, iou=0.9 per ridurre gli elementi overlappati
        results  = self.model.predict(source=image, save=False, imgsz=320, conf=0.005, iou=0.4, verbose=False)
        for result in results:
            bbox_list = result.boxes.xyxy.tolist()          # bounding boxes all objects, you can also get xywh with boxes.xywh
            clss_list = result.boxes.cls.int().tolist()     # class index all objects
            conf_list = result.boxes.conf.tolist()          # confidence list all objects
        img_w_bound_box = result.plot()
        return img_w_bound_box, results
            

