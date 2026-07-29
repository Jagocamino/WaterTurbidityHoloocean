import holoocean
import cv2
import numpy as np
import time

from controllers.keyboard_controller import KeyboardController
from lib.scenario_builder import ScenarioConfig
from lib.worlds import World
from lib.rover import Rover
from utils.sonar_viz import PolarSonarVisualizer
from utils.camera_viz import show_camera
from yolomodel import YoloModel
from telemetry.parsing import parse_pose
from telemetry.estimation import (
    parse_velocity,
    estimate_motion_state,
    parse_depth,
    estimate_depth_from_seabed,
)
from telemetry.hud import draw_telemetry_hud

SENSOR_MAP = {
    "Pose": "PoseSensor",
    "Velocity": "VelocitySensor",
    "IMU": "IMUSensor",
    "DVL": "DVLSensor",
    "RangeFinder": "RangeFinderSensor",
    "Collision": "CollisionSensor",
    "Depth": "DepthSensor",
}

rov0 = Rover.BlueROV2(
    name="rov0",
    location=[0, -8, -3],
    rotation=[0, 0, 90], # yaw = 90 top-down view
    control_scheme=0,
)

yolo = YoloModel()
yolo.init_yolo_model()
inputUnavailable = False
predictUnavailable = False
delayTime = time.time()
timeCameraYOLO = time.time()
screenSessione = 0
boolFlashlights = False
switchProp = True
secchi_distance = None
delayUpdateYOLOcam = 0.2
delayMissinput = 0.8

for sensor in rov0.sensors:
    if sensor.get("sensor_name") == "FrontCamera":
        FOV = sensor["configuration"]["FOV"] 
        break

def computeDistance(oldBoundingBox, window_sz, FOV):
    real_sz = 0.3
    pixl_sz = oldBoundingBox[2]-oldBoundingBox[0]   
    focal_length = window_sz[1]/(2*np.tan(np.radians(FOV)/2))
    distance = (real_sz * focal_length)/pixl_sz
    return distance

def distanceDetector(results, image, FOV):
    global secchi_distance
    innerBox = 0.4 # quanta percentuale dello schermo nella parte centrale si prende il box adibito al riconoscimento del disco  
    window_sz = np.array(image.shape)
    # window_sz[1] 0=480 (height) 1=640 (width)
    box_dim = min(window_sz[0],window_sz[1])
    shift = box_dim*innerBox/2
    # i box vengono misurati con coordinate top-left,bottom-right  
    x1 = window_sz[1]/2-shift
    y1 = window_sz[0]/2-shift
    x2 = window_sz[1]/2+shift
    y2 = window_sz[0]/2+shift
    for result in results:
        for bounding_box in result.boxes.xyxy.tolist():
            xb1 = bounding_box[0]
            yb1 = bounding_box[1]
            xb2 = bounding_box[2]
            yb2 = bounding_box[3] 
            if xb1>x1 and yb1>y1 and xb2<x2 and yb2<y2:# mi salvo il primo bounding box che rispetta la condizione 
                distanceDetector.oldBoundingBox = bounding_box
                return 
            # calcolo distanza della oldBoundingBox 
    if hasattr(distanceDetector,"oldBoundingBox"):
        if distanceDetector.oldBoundingBox is not None:
            secchi_distance = computeDistance(distanceDetector.oldBoundingBox, window_sz, FOV)
            print("Secchi disk distance: ",secchi_distance)
            distanceDetector.oldBoundingBox = None

def camera_YOLO(state, predictUnavailable, timeCameraYOLO):
    if predictUnavailable == False:
        predictUnavailable = True
        timeCameraYOLO = time.time()
        img6 = state["FrontCamera"]
        img6 = state["DownCamera"]
        if img6 is not None:
            img6 = np.asarray(img6)
            if img6.ndim == 3 and img6.shape[2] >= 3:
                img6 = img6[:, :, :3]
                # Convert float images to uint8 if needed, the "normal" OpenCV image format
            if img6.dtype != np.uint8:
                img6 = np.clip(img6 * 255.0, 0, 255).astype(np.uint8)
        result_w_bound_box, results = yolo.detect_nosave(img6)
        turbidity = distanceDetector(results, result_w_bound_box, FOV)
       # distanceDetector(result_w_bound_box)
        cv2.imshow("Accuracy runtime", result_w_bound_box)
    return predictUnavailable, timeCameraYOLO

scenario = (
    ScenarioConfig("BlueROV_CustomOctree")
    .set_world(World.Dam) #Rooms PierHarbor Dam 
    .add_agent(rov0)
)

sonar_viz = PolarSonarVisualizer(
    azimuth_deg=90,
    range_min=1,
    range_max=30,
    plot_hz=5,
    ema_alpha=0.1
)

controller = KeyboardController()

with holoocean.make(
    scenario_cfg=scenario.to_dict(),
    show_viewport=False,
    ticks_per_sec=30,
    frames_per_sec=True
) as env:
        
   # env.spawn_prop(prop_type="sphere",location=[0,0,-3],rotation=[0.0,0.0,90.0],
   #                scale=[0.3,0.01,0.3],sim_physics=False,material="cobblestone",tag=str(delayTime))
   # env.spawn_prop(prop_type="cylinder",location=[1,0,-3],rotation=[0.0,0.0,0.0],
   #                scale=[0.3,0.3,0.01],sim_physics=False,material="cobblestone",tag=str(delayTime))

   # environment manager
    env.water_fog(fogDensity=9.8, fogDepth=1.0, color_R=0.4, color_G=0.6, color_B=1.0)
   # env.water_fog(fogDensity=0.0, fogDepth=0.0, color_R=0.0, color_G=0.0,color_B=0.0) #clear water condition
   # env.change_weather(0) #0 - sunny, 1 - cloudy, and 2 - rainy
    env.set_rain_parameters(0,400,-1000, 2000)  # Custom rain behavior   
    env.air_fog(fogDensity=3,fogDepth=5.0,color_R=0.5,color_G=0.5,color_B=0.6) # o direttamente env.air_fog(2.2) per il val della denistà
   # env.turn_on_flashlight("flashlight1",100000,80)
    env.change_time_of_day(14)
    
    last = {}

    while True:
        cmd = controller.get_command()
        if cmd is None:
            break

        state = env.step(cmd)

        for k, v in SENSOR_MAP.items():
            if v in state:
                last[k] = state[v]

        telemetry = {
            "pose": parse_pose(last.get("Pose")),
            "velocity": parse_velocity(last.get("Velocity")),
            "altitude": parse_depth(last.get("Depth")),
            "under_range": estimate_depth_from_seabed(last.get("RangeFinder")),
            "motion": estimate_motion_state(last.get("IMU")),
            "collision": last.get("Collision"),
            "secchi_depth": secchi_distance
        }

        show_camera(state, "FrontCamera", "Front Camera")  
        show_camera(state, "DownCamera", "Down Camera") # Top-to-bottom view, uso questo

        if inputUnavailable == True: # delay, no more than 1 shot per second
            if delayTime + delayMissinput < time.time():
                inputUnavailable = False
        
        if predictUnavailable == True: # delay, per la finestra YOLO
            if timeCameraYOLO + delayUpdateYOLOcam < time.time():
                predictUnavailable = False

        if controller.print_image_key_l():
            if inputUnavailable == False:
                inputUnavailable = True                 
                delayTime = time.time()
                img6 = state["DownCamera"]
                if img6 is not None:
                    img6 = np.asarray(img6)
                    if img6.ndim == 3 and img6.shape[2] >= 3:
                        img6 = img6[:, :, :3]
                        # Convert float images to uint8 if needed, the "normal" OpenCV image format
                    if img6.dtype != np.uint8:
                        img6 = np.clip(img6 * 255.0, 0, 255).astype(np.uint8)
                    percorso = f'/home/jago.camoni.STUDENTI/Documenti/HoloOceanLibrary/src/runs/camerascreens/foto_{int(time.time()*1000)}.png'
                    cv2.imwrite(percorso,img6)
                    screenSessione = screenSessione + 1
                    print("Took photo ",screenSessione)


        if controller.spawn_prop_key_o():
            if inputUnavailable == False:
                inputUnavailable = True
                delayTime = time.time()
                rovPos = parse_pose(last.get("Pose"))['pos'] # coordinate del rover, aggiungere round(rovPos[a],b) per una precisione meno accurata
                rovPos = {
                    "x": rovPos[0],
                    "y": rovPos[1],
                    "z": rovPos[2]
                }
                if switchProp == True:    
                    env.spawn_prop(prop_type="cylinder", #box sphere cylinder cone 
                                    location=[rovPos["x"],rovPos["y"],rovPos["z"]-2],
                                    rotation=[0.0,0.0,0.0],
                                    scale=[0.3,0.3,0.01],sim_physics=False,material="cobblestone",tag=str(delayTime))
                    switchProp = False
                else:
                    switchProp = True
                    env.spawn_prop(prop_type="sphere", #box sphere cylinder cone 
                                    location=[rovPos["x"],rovPos["y"],rovPos["z"]-2],
                                    rotation=[0.0,0.0,90.0],
                                    scale=[0.01,0.3,0.3],sim_physics=False,material="cobblestone",tag=str(delayTime))
                                        

        if controller.spawn_prop_key_p(): # not working 
            if inputUnavailable == False:
                inputUnavailable = True
                delayTime = time.time()
                rovRPY = parse_pose(last.get("Pose"))['rpy'] # coordinate del roll,pitch,yaw, aggiungere round(rovPos[a],b) per una precisione meno accurata
                rovPos = parse_pose(last.get("Pose"))['pos'] # coordinate del rover, aggiungere round(rovPos[a],b) per una precisione meno accurata
                d = 1 # cofattore di distanza dalla camera 
                rovPos = {
                    "x": rovPos[0],
                    "y": rovPos[1],
                    "z": rovPos[2]
                }
                V = {
                    "x": np.cos(np.radians(rovRPY[2]))*np.cos(np.radians(rovRPY[1])),
                    "y": np.sin(np.radians(rovRPY[2]))*np.cos(np.radians(rovRPY[1])),
                    "z": np.sin(np.radians(rovRPY[1]))
                }
                newSecchiPos = {
                    "x": rovPos["x"] + V["x"]*d,
                    "y": rovPos["y"] + V["y"]*d,
                    "z": rovPos["z"] + V["z"]*d
                }
                env.spawn_prop(prop_type="sphere", #box sphere cylinder cone 
                               location=[newSecchiPos["x"],newSecchiPos["y"],newSecchiPos["z"]],
                               rotation=[round(rovRPY[0]),round(rovRPY[1]),round(rovRPY[2])],  # REGISTRA LA ROTAZIONE SBAGLIATA 
                               scale=[0.01,0.3,0.3],sim_physics=False,material="cobblestone",tag=str(delayTime))
        
        predictUnavailable, timeCameraYOLO = camera_YOLO(state, predictUnavailable, timeCameraYOLO)

        if controller.flashlights_on_off_b():
            if inputUnavailable == False:
                inputUnavailable = True
                delayTime = time.time()
                if boolFlashlights == False:
                    boolFlashlights = True
                    env.turn_on_flashlight("flashlight1")
                    env.turn_on_flashlight("flashlight2")
                    env.turn_on_flashlight("flashlight3")
                    env.turn_on_flashlight("flashlight4")
                    print("lights ON")
                else:
                    boolFlashlights = False
                    env.turn_off_flashlight("flashlight1")
                    env.turn_off_flashlight("flashlight2")
                    env.turn_off_flashlight("flashlight3")
                    env.turn_off_flashlight("flashlight4")
                    print("lights OFF")

        draw_telemetry_hud(telemetry)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

#sonar_viz.close()
cv2.destroyAllWindows()
