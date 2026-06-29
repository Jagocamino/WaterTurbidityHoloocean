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


#TODO
rov0 = Rover.BlueROV2(
    name="rov0",
    location=[4, 0, -3],
    rotation=[0, 0, 180],
    control_scheme=0,
)

# ---------- TEST SECTION out While
yolo = YoloModel()
yolo.init_yolo_model()
imgRecord = False
ritardoInput = False
ritardoTime = time.time()
screenSessione = 0

# ---------- TEST SECTION out While end


scenario = (
    ScenarioConfig("BlueROV_CustomOctree")
    .set_world(World.PierHarbor) #Rooms Dam PierHarbor 
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
    
    
   # turbidity manager , gestisco torbidità dentro la simulazione
   # env.water_fog(fogDensity=0.8, fogDepth=3.0, color_R=0.4, color_G=0.6, color_B=1.0) #x opacità acqua 0<fogDensity<10
    env.water_fog(fogDensity=0.0, fogDepth=0.0) #per raccogliere i dati, altrimenti usare il water_fog di sopra 
    env.change_weather(0) #0 - sunny, 1 - cloudy, and 2 - rainy
   #env.set_rain_parameters(0,400,-1000, 2000)  # Custom rain behavior   
   #env.air_fog(0.8,fogDepth=5.0,color_R=0.5,color_G=0.5,color_B=0.6) o direttamente env.air_fog(2.2) per il val della denistà
   #TODO x test luci 
    env.turn_on_flashlight("flashlight1",40000,80) #intensity(0,100000) angle_pitch(-70,70) angle_yaw(-70,70)
    env.turn_on_flashlight("flashlight2",40000,80)
    env.spawn_prop(prop_type="box",location=[0,0,-3],rotation=[0,0,0],scale=[1,1,1],sim_physics=False,material="wood",tag="")
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
        }

        show_camera(state, "FrontCamera", "Front Camera")  
       # show_camera(state, "SonarCamera", "Sonar Camera") #togli commento per camera

        

       # ---------- TEST SECTION in While

        if ritardoInput == True: # così non fa più di uno screen al secondo 
            if ritardoTime + 0.8 < time.time():
                ritardoInput = False

       # screenshot training 
        if controller.print_image_key_l():
            if ritardoInput == False:
                ritardoInput = True
                ritardoTime = time.time()
                img6 = state["FrontCamera"]
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
                    print("Immagine ",screenSessione," scattata")
                   # cv2.imshow("fotografia front", img6)
            # result = yolo.detect(img6)

       # screenshot testing   
        if controller.print_image_key_l():
            if ritardoInput == False:
                ritardoInput = True
                ritardoTime = time.time()
                img6 = state["FrontCamera"]
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
                    print("Immagine ",screenSessione," scattata")
                   # cv2.imshow("fotografia front", img6)
            # result = yolo.detect(img6)
        

       #TODO2 DA FINIRE
        if controller.record_scene_key_b():
            img6 = state["FrontCamera"]
            if img6 is not None:
                img6 = np.asarray(img6)
                if img6.ndim == 3 and img6.shape[2] >= 3:
                    img6 = img6[:, :, :3]
                    # Convert float images to uint8 if needed, the "normal" OpenCV image format
                if img6.dtype != np.uint8:
                    img6 = np.clip(img6 * 255.0, 0, 255).astype(np.uint8)
                cv2.imshow("fotografia front", img6)
            result = yolo.detect(img6)
       
       
       # if controller.record_scene_key_b():
       #     if imgRecord == False:
       #         imgRecord = True
       #         img6 = state["FrontCamera"]
       #         if img6 is not None:
       #             img6 = np.asarray(img6)
       #             if img6.ndim == 3 and img6.shape[2] >= 3:
       #                 img6 = img6[:, :, :3]
       #                 # Convert float images to uint8 if needed, the "normal" OpenCV image format
       #             if img6.dtype != np.uint8:
       #                 img6 = np.clip(img6 * 255.0, 0, 255).astype(np.uint8)
       #             cv2.imshow("fotografia front", img6)
       #         result = yolo.detect(img6)
       #     else:
       #         imgRecord = False
        
        


       # -------- END TEST SECTION in While 


       # if "ImagingSonar" in state:
       #     sonar_viz.submit(state["ImagingSonar"])
       # sonar_viz.update_plot()

        draw_telemetry_hud(telemetry)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

#sonar_viz.close()
cv2.destroyAllWindows()
