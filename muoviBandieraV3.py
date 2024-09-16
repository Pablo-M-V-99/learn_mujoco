import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET
import os
import datetime
import random
import json
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
    random.seed(seed)

    # Parametri TEMPO
    t1 = 5          # durata movimento da un punto della griglia all'altro
    tz = 7          # durata cambio piano
    tr = 5          # durata rotazione
    T = t1          # T è una variabile di appoggio per alternare tra il tempo di cambio piano e quello di traslazione

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

    # Parametri GRIGLIA
    total_length = 0.3
    dimCell = 0.1                                    # distanza fra due nodi adiacenti
    wid_G = int(total_length//dimCell + 1)           # numero di nodi lungo X
    len_G = int(total_length//dimCell +1)            # numero di nodi lungo Y
    height_G = int(total_length//dimCell + 1)        # numero di nodi lungo Z
    offX, offY, offZ = 0, -(dimension[1] - 1) * spacing[1] / 2, 0              # offset per l'allineamento della griglia

    # Parametri ROTAZIONE
    pitch_rot, yaw_rot = 20, 60         # Il yaw è la rotazione sul piano trasverso (Z) mentre il pitch la rotazione
    pitch_step, yaw_step = 5, 5         # sul piano frontale (Y). Nessuna rotazione sul piano sagittale (X)

    # Calcolo numero di configurazioni
    c_yaw = int(yaw_rot // yaw_step * 2 + 1)            # numero di configurazioni in Yaw per nodo
    c_pitch = int(pitch_rot // pitch_step * 2 + 1)      # numero di configurazioni in Pitch per nodo
    c_nodi = height_G * wid_G * len_G                   # numero nodi
    c_tot = c_yaw * c_pitch * c_nodi                    # configurazioni totali

    # GENERA LISTA CONFIGURAZIONI
    lista_configurazioni = [ii for ii in range(c_tot)]
    lista_campionata = []
    ii = 0
    while ii < sampling:
        ii += 1
        sample = random.randint(0, c_tot-1)
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

    # CREAZIONE GRIGLIA CARTESIANA
    cartesianGrid(len_G, wid_G, height_G, dimCell, offX, offY, offZ)
    json_path = (f"griglia_{wid_G}x{len_G}x{height_G}.json")
    with open(json_path, "r") as file:
        griglia = json.load(file)

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

    # MOVIMENTO SU TUTTA LA GRIGLIA
    while i <= wid_G - 1 and j <= len_G - 1 and k <= height_G - 1 and len(lista_campionata) != 0:

        # Allineamento con la griglia
        if i == 0 and j == 0 and k == 0:
            moveToNext(m, d, viewer, pos0, nextPose, 20, 'TRANSLATE')
            pos0 = nextPose

        # Movimento (traslazione + rotazione)
        or0, pos0, depth_images, angles, poses = move(m, d, viewer, yaw_step, yaw_rot, pitch_step, pitch_rot, or0, pos0,
                                                      nextPose, tr, T, plot, depth_images, angles, poses,
                                                      lista_configurazioni, lista_campionata)

        # Determinzazione del prossimo nodo verso cui traslare
        nextPose = griglia[f"cella_{i}_{j}_{k}"]
        T = t1
        # Movimento lungo X, Y e Z (della griglia)
        # i righe, j colonne, k piani
        if j % 2 == 0:                                  # riga pari
            if i == wid_G - 1 and j != len_G - 1:       # incrementa riga se non sono all'ultima riga
                j += 1
            else:
                if i != wid_G - 1:                      # incrementa colonna se non sono all'ultima colonna
                    i += 1
                else:                                   # incrementa piano se all'ultima colonna e all'ultima riga
                    k += 1
                    i = 0
                    j = 0
                    T = tz
        else:                                           # riga dispari
            if i == 0 and j != len_G - 1:               # incrementa riga se non sono all'ultima riga
                j += 1
            else:
                if i != 0:                              # decrementa colonna se non sono alla prima colonna
                    i -= 1
                else:                                   # incrementa piano se alla prima colonna e all'ultima riga
                    k += 1
                    i = 0
                    j = 0
                    T = tz

    # Genera il timestamp corrente
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Crea la cartella con il timestamp
    folder_name = f"dati_{current_time}"
    os.makedirs(folder_name, exist_ok=True)  # Crea la cartella (exist_ok=True evita errori se esiste già)

    np.savez(f"{folder_name}/immaginiDepth.npz", depth_images = depth_images, angles = angles, poses = poses)

    # saveLabels(angles, poses, folder_name)
    saveParameters(mass, radius, damping, poisson, young, thickness, spacing, dimension, posizione_manoDx,
                   posizione_manoSx, folder_name)

    # print("Tutto BENE")

if __name__ == '__main__':
    main(0.2, 0.1, 0.5, 1000, 2, 10)
