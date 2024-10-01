import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET
import os
import datetime
import random
import json
from math import sin, radians
from scipy.spatial.transform import Rotation as R
from scriptSaveLabels import saveLabels, saveParameters
from scriptGriglia import creazioneGriglia as cartesianGrid
from scriptMovimento import move, moveToNext
from scriptCreaLenzuolo import lenzuolo_maker, connect_maker

def main(massa, smorzamento, mod_Poisson, mod_Young, seed, sampling):

    # Attivare VIEWER
    # view = True
    view = False

    # Attivare PLOT
    # plot = True
    plot = False

    # Attivare il SEED
    # random.seed(seed)

    # Parametri TEMPO
    t1 = 5           # durata movimento da un punto della griglia all'altro
    tz = 10          # durata cambio piano
    tr = 10          # durata rotazione
    T = t1           # T è una variabile di appoggio per alternare tra il tempo di cambio piano e quello di traslazione

    # Parametri XML
    mass = round(random.uniform(massa * 0.9, massa * 1.1), 3)
    radius = 0.001
    damping = round(random.uniform(smorzamento * 0.9 ,smorzamento * 1.1), 4)
    poisson = round(random.uniform(0, mod_Poisson), 4)
    young = round(random.uniform(mod_Young * 0.9, mod_Young * 1.1), 1)
    thickness = round(random.uniform(0.001, 0.004), 3)
    larghezza_ply = round(random.uniform(0.35, 0.80), 2)
    lunghezza_ply = round(random.uniform(0.70, 1.2), 2)
    spacing = [round(random.uniform(0.01, 0.05), 2), round(random.uniform(0.01, 0.05), 2), 0.05]
    pos = [0, 0, 0]
    dimension = [int(larghezza_ply / spacing[0]) + 1, int(lunghezza_ply / spacing[1]) + 1, 1]
    posizione_manoDx = round(random.uniform(0.5, 1), 3)
    posizione_manoSx = round(random.uniform(0.5, 1), 3)

    # Parametri ROTAZIONE
    pitch_rot, yaw_rot = 20, 30         # Il yaw(30°) è la rotazione sul piano trasverso (Z) mentre il pitch(20°) la rotazione
    pitch_step, yaw_step = 5, 5         # sul piano frontale (Y). Nessuna rotazione sul piano sagittale (X)

    # Parametri GRIGLIA
    wid_G = 0.3                         # larghezza griglia (X)
    len_G = 0.3                         # lunghezza grglia (Y)
    height_G = 0.3                      # altezza griglia (Z)
    dimCell = 0.1                            # distanza fra due nodi adiacenti
    X_G = round(wid_G / dimCell) + 1           # numero di nodi lungo X
    Y_G = round(len_G / dimCell) + 1           # numero di nodi lungo Y
    Z_G = round(height_G / dimCell) + 1        # numero di nodi lungo Z
    # offsets per l'allineamento della griglia
    offX = -wid_G / 2
    offY = round(((dimension[0] - 1) * spacing[0] * sin(radians(yaw_rot)) - (dimension[1] - 1) * spacing[1]) / 2, 2)
    offZ = -height_G / 2
    # Calcolo numero di configurazioni
    c_pitch = int(pitch_rot / pitch_step) * 2 + 1  # numero di configurazioni in Pitch per nodo
    c_yaw = int(yaw_rot / yaw_step) * 2 + 1  # numero di configurazioni in Yaw per nodo
    c_nodi = Z_G * X_G * Y_G  # numero nodi
    c_tot = c_yaw * c_pitch * c_nodi  # configurazioni totali
    if c_tot < sampling:
        sampling = c_tot
        # print('ERROR: il numero di pose da campionare è superiore al numero di pose totali. Verranno campionate tutte '
        #         'le pose')

    # CREAZIONE GRIGLIA CARTESIANA
    cartesianGrid(X_G, Y_G, Z_G, dimCell, offX, offY, offZ)
    json_path = (f"griglia_{X_G}x{Y_G}x{Z_G}.json")
    with open(json_path, "r") as file:
        griglia = json.load(file)

    # GENERA LISTA CONFIGURAZIONI
    lista_configurazioni = [ii for ii in range(c_tot)]
    lista_campionata = []
    ii = 0
    while ii < sampling:
        ii += 1
        sample = random.randint(0, c_tot - 1)
        if sample not in lista_campionata:
            lista_campionata.append(sample)
        else:
            ii -= 1
    lista_campionata.sort()

    # coordinate cella
    i, j, k = 0, 0, 0

    # creazione vettori per append
    depth_images = []
    angles = []
    poses = []

    # Crezione XML
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
    xml_file = lenzuolo_maker(root, mass, radius, damping, poisson, young, thickness, pos, dimension, spacing,
                              posizione_manoDx, posizione_manoSx)
    xml_file = connect_maker(root, dimension, spacing, posizione_manoDx, posizione_manoSx)
    xml_str = ET.tostring(xml_file, encoding='unicode', method='xml')
    # print(xml_str)

    # Caricamento dati per MUJOCO
    m = mujoco.MjModel.from_xml_string(xml_str)
    d = mujoco.MjData(m)

    # Dati di MUJOCO necessari per il movimento
    timeStep = m.opt.timestep                   # mujoco simulation timestep
    pos0 = np.array(d.mocap_pos[1])             # posa iniziale mocap body
    nextPose = griglia[f"cella_{i}_{j}_{k}"]    # posa successiva
    or0 = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True)
    or0 = or0.as_euler('xyz', degrees=True)

    # Genera il timestamp corrente
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Crea la cartella con il timestamp
    folder_name = f"datasets/dati_{current_time}"
    os.makedirs(folder_name, exist_ok=True)  # Crea la cartella (exist_ok=True evita errori se esiste già)

    # saveLabels(angles, poses, folder_name)
    saveParameters(mass, radius, damping, poisson, young, thickness, spacing, dimension, posizione_manoDx,
                   posizione_manoSx, folder_name)


    if view:
        viewer = mujoco.viewer.launch_passive(m, d)
    else:
        viewer = None

    # Reset state and time
    mujoco.mj_resetData(m, d)

    # Transitorio spawn bandiera
    t = 0
    while t <= 20:
        t += timeStep
        mujoco.mj_step(m, d)

    # Allineamento con la griglia
    moveToNext(m, d, viewer, pos0, nextPose, 10, 'TRANSLATE')
    pos0 = nextPose

    # MOVIMENTO SU TUTTA LA GRIGLIA
    while i <= X_G - 1 and j <= Y_G - 1 and k <= Z_G - 1 and len(lista_campionata) != 0:

        # Determinzazione del prossimo nodo verso cui traslare
        nextPose = griglia[f"cella_{i}_{j}_{k}"]

        # Movimento (traslazione + rotazione)
        or0, pos0, depth_images, angles, poses = move(m, d, viewer, yaw_step, yaw_rot, pitch_step, pitch_rot, or0, pos0,
                                                      nextPose, tr, T, plot, depth_images, angles, poses,
                                                      lista_configurazioni, lista_campionata)

        T = t1
        # Movimento lungo X, Y e Z (della griglia)
        # i righe, j colonne, k piani
        if j % 2 == 0:                                  # riga pari
            if i == X_G - 1 and j != Y_G - 1:       # incrementa riga se non sono all'ultima riga
                j += 1
            else:
                if i != X_G - 1:                      # incrementa colonna se non sono all'ultima colonna
                    i += 1
                else:                                   # incrementa piano se all'ultima colonna e all'ultima riga
                    k += 1
                    i = 0
                    j = 0
                    T = tz
        else:                                           # riga dispari
            if i == 0 and j != Y_G - 1:               # incrementa riga se non sono all'ultima riga
                j += 1
            else:
                if i != 0:                              # decrementa colonna se non sono alla prima colonna
                    i -= 1
                else:                                   # incrementa piano se alla prima colonna e all'ultima riga
                    k += 1
                    i = 0
                    j = 0
                    T = tz

    np.savez(f"{folder_name}/depth_and_labels.npz", depth_images = depth_images, angles = angles, poses = poses)

    # print("Tutto BENE")

if __name__ == '__main__':
    main(0.2, 0.1, 0.5, 1000, 3, 3000)
