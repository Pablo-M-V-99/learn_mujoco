import mujoco.viewer
import numpy as np
import json
from scriptSaveLabels import saveLabels
from scipy.spatial.transform import Rotation as R
from scriptGriglia import creazioneGriglia as cartesianGrid
# from scriptGriglia import creazioneGrigliaRadiale as radialGrid
from scriptTraiettoria import posStep
from scriptYawAndPitch import firstRot, secondRot
from scriptImageAcquisition import imageAcquisition

# PARAMETRI
# tempi
t1 = 5          # durata movimento da un punto della griglia all'altro
tz = 7          # durata cambio piano
tr = 2          # durata rotazione
tR = 4          # durata riallineamento

# nodi griglia
len_G = 2         # numero di nodi lungo Y
wid_G = 2         # numero di nodi lungo X
height_G = 2      # numero di nodi lungo Z
dimCell = 0.05        # distanza fra due nodi adiacenti (cm)
offX, offY, offZ = 0, 0, 0 - 1.5      # offset per l'allineamento della griglia (cm)

# coordinate cella
i, j, k = 0, 0, 0

# rotazione
pitch_rot, yaw_rot = 10, 12         # la rotazione deve essere divisibile per gli incrementi!
pitch_step, yaw_step = 10, 12       # il yaw è la rotazione sul piano trasverso (Z) mentre il pitch la rotazione
                                    # sul piano frontale (Y). Nessuna rotazione sul piano sagittale (X)

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

# # CREAZIONE GRIGLIA RADIALE
# radialGrid()

# CARICAMENTO MODELLO MUJOCO XML
xml_path = "/home/pablo/PycharmProjects/learn_mujoco/muoviBandiera.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

timeStep = m.opt.timestep               # mujoco simulation timestep
pos0 = np.array(d.mocap_pos[1])         # posa iniziale mocap body
nextPose = griglia[f"cella_{i}_{j}_{k}"]

# np.array(d.mocap_quat[1]) è l'orientamento in quaternioni come me lo dà mujoco -> [1 0 0 0].
# pitch0.as_euler() è l'orientamento in angoli di Eulero da scipy -> [180 0 0] (nell'ordine roll pitch yaw)
# devo riordinare il quaternione
or0 = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True)
or0 = or0.as_euler('xyz', degrees=True)
t = 0

yaw, pitch, roll = or0[2], or0[1], or0[0]  # orientamento iniziale (terna fissa) yaw(z) pitch(y) roll(x)

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

    # Acquisizione posizione neutra (appena arrivato sul nodo)
    depth_images, segmentation_images, angles, poses = imageAcquisition(m, d, depth_images, segmentation_images, angles,
                                                                        poses)

    # Rotazione in PITCH (chiamando qui secondRot si effettua  la rotazione con yaw = 0 (posizione neutra))
    depth_images, segmentation_images, angles, poses = secondRot(m, d, viewer, pitch_step, pitch_rot, or0, tr, tR,
                                                                 depth_images, segmentation_images, angles, poses)
    depth_images, segmentation_images, angles, poses = secondRot(m, d, viewer, -pitch_step, -pitch_rot, or0, tr, tR,
                                                                 depth_images, segmentation_images, angles, poses)

    # ROTAZIONI
    depth_images, segmentation_images, angles, poses = firstRot(m, d, viewer, yaw_step, yaw_rot, pitch_step, pitch_rot,
                                                                or0, tr, tR, depth_images, segmentation_images, angles,
                                                                poses)
    depth_images, segmentation_images, angles, poses = firstRot(m, d, viewer, -yaw_step, -yaw_rot, pitch_step,
                                                                pitch_rot,
                                                                or0, tr, tR, depth_images, segmentation_images, angles,
                                                                poses)

    # TRASLAZIONE
    pos0 = nextPose
    nextPose = griglia[f"cella_{i}_{j}_{k}"]
    posStep(m, d, viewer, pos0, nextPose, t1)

    # Movimento lungo X, Y e Z
    if j % 2 == 0:                                  # riga pari
        if i == wid_G - 1  and j != len_G - 1:   # incrementa riga se non sono all'ultima riga
            j += 1
        else:
            if i != wid_G - 1:                    # incrementa colonna se non sono all'ultima colonna
                i += 1
            else:                                   # inc. piano se all'ultima colonna e
                k += 1
                i = 0
                j = 0
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

np.savez('immaginiDepth.npz', depth_images)
np.savez('immaginiSegmentate.npz', segmentation_images)

saveLabels(angles, poses)

print("Tutto BENE")

