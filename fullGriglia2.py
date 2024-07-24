import time
import mujoco.viewer
import numpy as np
import json
from saveLabels import saveLabels
from scipy.spatial.transform import Rotation as R
from scriptGriglia import creazioneGriglia as cartesianGrid
from scriptGriglia import creazioneGrigliaRadiale as radialGrid
from scriptTraiettoria import move, posStep
from scriptYawAndPitch import firstRot
from scriptImageAcquisition import imageAcquisition

# PARAMETRI
# tempi
t1 = 5          # durata movimento da un punto della griglia all'altro
tz = 7          # durata cambio piano
tr = 2          # durata rotazione
tR = 4          # durata riallineamento

# nodi grglia (x+1)
len_G = 1         # lunghezza griglia (righe)
wid_G = 1         # larghezza griglia (colonne)
height_G = 1      # altezza griglia (piani)
dimCell = 0.05        # distanza fra due nodi adiacenti (cm)
offX, offY, offZ = 0, 0.3, 0      # offset per l'allineamento della griglia (cm)

# coordinate cella
i, j, k = 0, 0, 0

# rotazione
pitch_rot, yaw_rot = 10, 6     # la rotazione deve essere divisibile per gli incrementi!
pitch_step, yaw_step = 5, 3

# view = True
view = False

depth_images = []
segmentation_images = []
A_ws_TCP = []
angles = []


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
xml_path = "/home/pablo/PycharmProjects/learn_mujoco/muoviBandiera.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

timeStep = m.opt.timestep               # mujoco simulation timestep
pos0 = np.array(d.mocap_pos[1])         # posa iniziale mocap body
nextPose = griglia[f"cella_{i}_{j}_{k}"]

# np.array(d.mocap_quat[1]) è l'orientamento in quaternioni come me lo dà mujoco -> [1 0 0 0]. pitch0.as_quat() è
# l'orientamento in quaternioni da scipy ->[1 0 0 0]. pitch0.as_euler() è l'orientamento in angoli di Eulero da
# scipy -> [180 0 0] (nell'ordine roll pitch yaw)
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

    depth_images, segmentation_images, A_ws_TCP, angles = imageAcquisition(m, d, yaw, pitch, roll, depth_images,
                                                                   segmentation_images, A_ws_TCP, angles)

    # creaRotMat(d, yaw, pitch, roll)
    # ROTAZIONI
    depth_images, segmentation_images, A_ws_TCP, angles = firstRot('YAW', m, d, viewer, yaw_step, yaw_rot, pitch_step, pitch_rot,
                                                 or0, roll, pitch, yaw, grigliaRad, tr, tR, timeStep, depth_images,
                                                 segmentation_images, A_ws_TCP, angles)
    depth_images, segmentation_images, A_ws_TCP, angles = firstRot('YAW', m, d, viewer, -yaw_step, -yaw_rot, pitch_step, pitch_rot,
                                                 or0, roll, pitch, yaw, grigliaRad, tr, tR, timeStep, depth_images,
                                                 segmentation_images, A_ws_TCP, angles)

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

saveLabels(A_ws_TCP, angles)

