import time
import mujoco.viewer
import numpy as np
import json
import xml.etree.ElementTree as ET
from scriptSaveLabels import saveLabels
from scipy.spatial.transform import Rotation as R
from scriptGriglia import creazioneGriglia as cartesianGrid
from scriptGriglia import creazioneGrigliaRadiale as radialGrid
from scriptTraiettoria import move, posStep
from scriptYawAndPitch import firstRot
from scriptImageAcquisition import imageAcquisition
from scriptCreaLenzuolo import lenzuolo_maker, connect_maker

# PARAMETRI
# tempi
t1 = 5          # durata movimento da un punto della griglia all'altro
tz = 7          # durata cambio piano
tr = 2          # durata rotazione
tR = 4          # durata riallineamento

# nodi grglia (x+1)
len_G = 3         # lunghezza griglia (righe)
wid_G = 3         # larghezza griglia (colonne)
height_G = 3      # altezza griglia (piani)
dimCell = 0.05        # distanza fra due nodi adiacenti (cm)
offX, offY, offZ = 0, 0.45, 0      # offset per l'allineamento della griglia (cm)

# coordinate cella
i, j, k = 0, 0, 0

# rotazione                     # la rotazione deve essere divisibile per gli incrementi!
pitch_rot, yaw_rot = 10, 6      # il yaw è la rotazione sul piano trasverso (Z) mentre il pitch la rotazione
pitch_step, yaw_step = 5, 3     #  sul piano frontale (Y). Nessuna rotazione sul piano sagittale (X)

view = True
# view = False

depth_images = []
segmentation_images = []
angles = []
poses = []

# CREAZIONE GRIGLIA CARTESIANA
cartesianGrid(len_G, wid_G, height_G, dimCell, offX, offY, offZ)
json_path = (f"/home/pablo/PycharmProjects/learn_mujoco/griglia_{wid_G}x"
             f"{len_G}x{height_G}.json")
with open(json_path, "r") as file:
    griglia = json.load(file)

# CREAZIONE GRIGLIA RADIALE
radialGrid()
json_path = (f"/home/pablo/PycharmProjects/learn_mujoco/grigliaRadiale.json")
with open(json_path, "r") as file:
    grigliaRad = json.load(file)

# CARICAMENTO MODELLO MUJOCO XML
# Parametri XML
mass = 0.1
radius = 0.001
damping = 0.001
poisson = 0
young = 20
thickness = 1e-2
pos = [0, 0, 1.5]
dimension = [19, 19, 1]
spacing = [0.05, 0.05, 1]
posizione_mano = 0.50

xml_file = """
<mujoco model="muovoBandiera2">
    <statistic center=".4 0 .8" extent="1.3"/>
    <option gravity="0 0 -9.81" density="10" solver="CG" tolerance="1e-6"/>	
    <extension>
        <plugin plugin="mujoco.elasticity.shell"/>
    </extension>
    <compiler eulerseq="XYZ"/>

    <visual>
        <global offheight="1024"/>
    </visual>

    <worldbody>
    </worldbody>	

</mujoco>		
"""

root = ET.fromstring(xml_file)

xml_file = lenzuolo_maker(root, mass, radius, damping, poisson, young, thickness, pos, dimension, spacing)
xml_file = connect_maker(root, dimension, spacing, posizione_mano)

xml_str = ET.tostring(xml_file, encoding='unicode', method='xml')
print(xml_str)

m = mujoco.MjModel.from_xml_string(xml_str)
d = mujoco.MjData(m)

timeStep = m.opt.timestep               # mujoco simulation timestep
pos0 = np.array(d.mocap_pos[1])         # posa iniziale mocap body
nextPose = griglia[f"cella_{i}_{j}_{k}"]

or0 = R.from_quat(np.array(d.mocap_quat[1]))
or0 = or0.as_euler('xyz', degrees=True)
t = 0

yaw, pitch, roll = or0[2], or0[1], or0[0]  # orientamento (terna fissa) yaw(z) pitch(y) roll(x)

if view:
    viewer = mujoco.viewer.launch_passive(m, d)
else:
    viewer = None

# Reset state and time
mujoco.mj_resetData(m, d)

# Transitorio spawn bandiera
while t <= 20:
    t += timeStep
    mujoco.mj_step(m, d)

while i <= wid_G and j <= len_G and k <= height_G:

    # MOVIMENTO SU TUTTA LA GRIGLIA

    # Allineamento con la griglia
    if i == 0 and j == 0 and k == 0:
        posStep(m, d, viewer, pos0, nextPose, 25, timeStep)
        i += 1

    depth_images, segmentation_images, angles, poses = imageAcquisition(m, d, yaw, pitch, roll, depth_images,
                                                        segmentation_images, angles, poses)

    # creaRotMat(d, yaw, pitch, roll)
    # ROTAZIONI
    depth_images, segmentation_images, angles, poses = firstRot('YAW', m, d, viewer, yaw_step, yaw_rot, pitch_step, pitch_rot,
                                                        or0, roll, pitch, yaw, grigliaRad, tr, tR, timeStep, depth_images,
                                                        segmentation_images, angles, poses)
    depth_images, segmentation_images, angles, poses = firstRot('YAW', m, d, viewer, -yaw_step, -yaw_rot, pitch_step, pitch_rot,
                                                        or0, roll, pitch, yaw, grigliaRad, tr, tR, timeStep, depth_images,
                                                        segmentation_images, angles, poses)

    # firstRot('PITCH', m, d, viewer, pitch_step, pitch_rot, yaw_step, yaw_rot, or0, roll, pitch, yaw, grigliaRad, tr, tR, timeStep)
    # firstRot('PITCH', m, d, viewer, -pitch_step, -pitch_rot, yaw_step, yaw_rot, or0, roll, pitch, yaw, grigliaRad, tr, tR, timeStep)

    # TRASLAZIONE
    pos0 = nextPose
    nextPose = griglia[f"cella_{i}_{j}_{k}"]
    posStep(m, d, viewer, pos0, nextPose, t1, timeStep)
    # creaRotMat(d, yaw, pitch, roll)
    T = t1

    # Movimento lungo X, Y e Z
    if j % 2 == 0:                                  # riga pari
        if i == wid_G and j != len_G:   # incrementa riga se non sono all'ultima riga
            j += 1
        else:
            if i != wid_G:                    # incrementa colonna se non sono all'ultima colonna
                i += 1
            else:                                   # inc. piano se all'ultima colonna e
                k += 1
                i = 0
                j = 0
                T = tz
    else:                                           # riga dispari
        if i == 0 and j != len_G:             # incrementa riga se non sono all'ultima riga
            j += 1
        else:
            if i != 0:                              # decrementa colonna se non sono alla prima colonna
                i -= 1
            else:                                   # inc. piano se all'ultima colonna e
                k += 1
                i = 0
                j = 0
                T = tz

np.savez('immaginiDepth.npz', depth_images)
np.savez('immaginiSegmentate.npz', segmentation_images)

saveLabels(angles, poses)

