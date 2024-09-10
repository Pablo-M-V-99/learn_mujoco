import mujoco.viewer
import numpy as np
import json
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from scriptSaveLabels import saveLabels
from scriptRandomNumber import generaNumeroCasuale
from scriptGriglia import creazioneGriglia as cartesianGrid
from scriptGriglia import creazioneGrigliaRadiale as radialGrid
from scriptTraiettoria import posStep
from scriptYawAndPitch import firstRot
from scriptImageAcquisition import imageAcquisition
from scriptCreaLenzuolo import lenzuolo_maker, connect_maker

# Parametri TEMPO
# tempi
t1 = 5          # durata movimento da un punto della griglia all'altro
tz = 7          # durata cambio piano
tr = 2          # durata rotazione
tR = 4          # durata riallineamento

# CARICAMENTO MODELLO MUJOCO XML
# Parametri XML
mass = 0.2
radius = 0.001
damping = 0.001
poisson = 0
young = 1000
thickness = 1e-2
pos = [0, 0, 1.5]
dimension = [9, 19, 1]
spacing = [0.05, 0.05, 0.05]
posizione_manoDx = generaNumeroCasuale(30, 100, True)/100
posizione_manoSx = generaNumeroCasuale(30, 100, True)/100

# Parametri GRIGLIA
len_G = 2           # numero di nodi lungo Y
wid_G = 2           # numero di nodi lungo X
height_G = 2        # numero di nodi lungo Z
dimCell = 0.05      # distanza fra due nodi adiacenti
offX, offY, offZ = 0, 0, 0 - pos[2]      # offset per l'allineamento della griglia

# Parametri ROTAZIONE
pitch_rot, yaw_rot = 20, 36         # la rotazione deve essere divisibile per gli incrementi!
pitch_step, yaw_step = 10, 12       # il yaw è la rotazione sul piano trasverso (Z) mentre il pitch la rotazione
                                    # sul piano frontale (Y). Nessuna rotazione sul piano sagittale (X)

view = True
# view = False

# coordinate cella
i, j, k = 0, 0, 0

# crezione vettori per append
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

xml_file = lenzuolo_maker(root, mass, radius, damping, poisson, young, thickness, pos, dimension, spacing, posizione_manoDx, posizione_manoSx)
xml_file = connect_maker(root, dimension, spacing, posizione_manoDx, posizione_manoSx)

xml_str = ET.tostring(xml_file, encoding='unicode', method='xml')
# print(xml_str)

m = mujoco.MjModel.from_xml_string(xml_str)
d = mujoco.MjData(m)

timeStep = m.opt.timestep               # mujoco simulation timestep
pos0 = np.array(d.mocap_pos[1])         # posa iniziale mocap body
nextPose = griglia[f"cella_{i}_{j}_{k}"]

or0 = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True)
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

while i <= wid_G - 1 and j <= len_G - 1 and k <= height_G - 1:

    # MOVIMENTO SU TUTTA LA GRIGLIA

    # Allineamento con la griglia
    if i == 0 and j == 0 and k == 0:
        posStep(m, d, viewer, pos0, nextPose, 25)
        i += 1

    depth_images, segmentation_images, angles, poses = imageAcquisition(m, d, depth_images, segmentation_images, angles,
                                                                        poses)


    # ROTAZIONI
    depth_images, segmentation_images, angles, poses = firstRot(m, d, viewer, yaw_step, yaw_rot, pitch_step, pitch_rot,
                                                                or0, tr, tR, depth_images, segmentation_images, angles,
                                                                poses)
    depth_images, segmentation_images, angles, poses = firstRot(m, d, viewer, -yaw_step, -yaw_rot, pitch_step, pitch_rot,
                                                                or0, tr, tR, depth_images, segmentation_images, angles,
                                                                poses)

    # TRASLAZIONE
    pos0 = nextPose
    nextPose = griglia[f"cella_{i}_{j}_{k}"]
    posStep(m, d, viewer, pos0, nextPose, t1)
    # T = t1

    # Movimento lungo X, Y e Z
    if j % 2 == 0:                                  # riga pari
        if i == wid_G - 1 and j != len_G - 1:   # incrementa riga se non sono all'ultima riga
            j += 1
        else:
            if i != wid_G - 1:                    # incrementa colonna se non sono all'ultima colonna
                i += 1
            else:                                   # inc. piano se all'ultima colonna e
                k += 1
                i = 0
                j = 0
                # T = tz
    else:                                           # riga dispari
        if i == 0 and j != len_G - 1:             # incrementa riga se non sono all'ultima riga
            j += 1
        else:
            if i != 0:                              # decrementa colonna se non sono alla prima colonna
                i -= 1
            else:                                   # inc. piano se all'ultima colonna e
                k += 1
                i = 0
                j = 0
                # T = tz

np.savez('immaginiDepth.npz', depth_images)
np.savez('immaginiSegmentate.npz', segmentation_images)

saveLabels(angles, poses)

print("Tutto BENE")
