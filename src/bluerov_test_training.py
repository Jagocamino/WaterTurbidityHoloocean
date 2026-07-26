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
    location=[0, -8, -3],
    rotation=[0, 0, 90], # modificare yaw per orientamento 
    control_scheme=0,
)

# ---------- TEST SECTION out While

def _fmt_vec3(pos): # appartenente a telemetry/hud per uniformare la posizione tridimensionale del rover, serve per il disco di Secchi 
    return f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"

yolo = YoloModel()
yolo.init_yolo_model()
ritardoInput = False
ritardoInputCameraYOLO = False
ritardoTime = time.time()
ritardoTimeCameraYOLO = time.time()
screenSessione = 0
boolFlashlights = False

def camera_YOLO(state, ritardoInputCameraYOLO, ritardoTimeCameraYOLO):
    if ritardoInputCameraYOLO == False:
        ritardoInputCameraYOLO = True
        ritardoTimeCameraYOLO = time.time()
        img6 = state["FrontCamera"]
        img6 = state["DownCamera"]
        if img6 is not None:
            img6 = np.asarray(img6)
            if img6.ndim == 3 and img6.shape[2] >= 3:
                img6 = img6[:, :, :3]
                # Convert float images to uint8 if needed, the "normal" OpenCV image format
            if img6.dtype != np.uint8:
                img6 = np.clip(img6 * 255.0, 0, 255).astype(np.uint8)
            # cv2.imshow("Accuracy runtime", img6)
        result_w_bound_box = yolo.detect_nosave(img6)
        cv2.imshow("Accuracy runtime", result_w_bound_box)
    return ritardoInputCameraYOLO, ritardoTimeCameraYOLO

# ---------- TEST SECTION out While end
#TODO 
scenario = (
    ScenarioConfig("BlueROV_CustomOctree")
    .set_world(World.PierHarbor) #Rooms PierHarbor Dam 
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
   
    #secchi Disk , da usare
    #env.spawn_prop(prop_type="sphere",location=[8,0,-5],rotation=[0.0,0.0,0.0],
                   #scale=[0.01,0.3,0.3],sim_physics=False,material="cobblestone",tag=str(ritardoTime))
    
#TODO modificare
    #frontal, per compensare al problema
    env.spawn_prop(prop_type="sphere",location=[0,0,-3],rotation=[0.0,0.0,90.0],
                   scale=[0.3,0.01,0.3],sim_physics=False,material="cobblestone",tag=str(ritardoTime))
   # env.spawn_prop(prop_type="cylinder",location=[1,0,-3],rotation=[0.0,0.0,0.0],
    #               scale=[0.3,0.3,0.01],sim_physics=False,material="cobblestone",tag=str(ritardoTime))
    
   # environment manager , gestisco torbidità dentro la simulazione
   # env.water_fog(fogDensity=6.8, fogDepth=1.0, color_R=0.4, color_G=0.6, color_B=1.0) #x opacità acqua 0<fogDensity<10
    env.water_fog(fogDensity=0.0, fogDepth=0.0, color_R=0.0, color_G=0.0,color_B=0.0) #per raccogliere i dati
    env.change_weather(0) #0 - sunny, 1 - cloudy, and 2 - rainy
   # env.set_rain_parameters(0,400,-1000, 2000)  # Custom rain behavior   
    env.air_fog(0.8,fogDepth=5.0,color_R=0.5,color_G=0.5,color_B=0.6) # o direttamente env.air_fog(2.2) per il val della denistà
   # env.turn_on_flashlight("flashlight1",100000,80) #intensity(0,100000) angle_pitch(-70,70) angle_yaw(-70,70)
   # env.turn_off_flashlight("flashlight2")
    env.change_time_of_day(12)
    
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
        show_camera(state, "DownCamera", "Down Camera") # Top-to-bottom view, uso questo

       # ---------- TEST SECTION in While

        if ritardoInput == True: # così non fa più di uno screen al secondo 
            if ritardoTime + 0.8 < time.time():
                ritardoInput = False
        
        if ritardoInputCameraYOLO == True: # per la finestra YOLO
            if ritardoTimeCameraYOLO + 0.8 < time.time():
                ritardoInputCameraYOLO = False

        if controller.print_image_key_l():
            if ritardoInput == False:
                ritardoInput = True                 
                ritardoTime = time.time()
                img6 = state["DownCamera"]
               # img6 = state["FrontCamera"]
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

        if controller.print_image_key_k(): #salva immagine screenata con bounding box 
            if ritardoInput == False:
                ritardoInput = True
                ritardoTime = time.time()
               # img6 = state["FrontCamera"]
                if img6 is not None:
                    img6 = np.asarray(img6)
                    if img6.ndim == 3 and img6.shape[2] >= 3:
                        img6 = img6[:, :, :3]
                        # Convert float images to uint8 if needed, the "normal" OpenCV image format
                    if img6.dtype != np.uint8:
                        img6 = np.clip(img6 * 255.0, 0, 255).astype(np.uint8)
                    screenSessione = screenSessione + 1
                    print("Immagine ",screenSessione," scattata")
                   # cv2.imshow("Accuracy runtime", img6)
                result_w_bound_box = yolo.detect(img6)
                cv2.imshow("Accuracy runtime", result_w_bound_box)

        if controller.spawn_prop_key_o():
            if ritardoInput == False:
                ritardoInput = True
                ritardoTime = time.time()
                rovPos = parse_pose(last.get("Pose"))['pos'] # coordinate del rover, aggiungere round(rovPos[a],b) per una precisione meno accurata
                rovPos = {
                    "x": rovPos[0],
                    "y": rovPos[1],
                    "z": rovPos[2]
                }
                env.spawn_prop(prop_type="cylinder", #box sphere cylinder cone 
                                location=[rovPos["x"],rovPos["y"],rovPos["z"]-2],
                                rotation=[0.0,0.0,0.0],
                                scale=[0.3,0.3,0.01],sim_physics=False,material="cobblestone",tag=str(ritardoTime))

        if controller.spawn_prop_key_p():
            if ritardoInput == False:
                ritardoInput = True
                ritardoTime = time.time()
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
                print(round(rovRPY[0])," ",round(rovRPY[1])," ",round(rovRPY[2])) # DEBUG rotazione non funziona correttamente per il disco 
                env.spawn_prop(prop_type="sphere", #box sphere cylinder cone 
                               location=[newSecchiPos["x"],newSecchiPos["y"],newSecchiPos["z"]],
                               rotation=[round(rovRPY[0]),round(rovRPY[1]),round(rovRPY[2])],  # REGISTRA LA ROTAZIONE SBAGLIATA 
                               scale=[0.01,0.3,0.3],sim_physics=False,material="cobblestone",tag=str(ritardoTime))
        
        ritardoInputCameraYOLO, ritardoTimeCameraYOLO = camera_YOLO(state, ritardoInputCameraYOLO, ritardoTimeCameraYOLO)

        if controller.flashlights_on_off_b():
            if ritardoInput == False:
                ritardoInput = True
                ritardoTime = time.time()
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



       # -------- END TEST SECTION in While 


       # if "ImagingSonar" in state:
       #     sonar_viz.submit(state["ImagingSonar"])
       # sonar_viz.update_plot()

        draw_telemetry_hud(telemetry)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

#sonar_viz.close()
cv2.destroyAllWindows()
