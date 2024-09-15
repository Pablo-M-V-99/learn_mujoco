import xml.etree.ElementTree as ET
from typing import List
import numpy as np


def lenzuolo_maker(root_xml: ET.Element,
                   mass: float,
                   radius: float,
                   damping: float,
                   poisson: float,
                   young: float,
                   thickness: float,
                   pos: List[float],
                   dimension: List[float],
                   spacing: List[float],
                   posizione_manoDx: float,
                   posizione_manoSx: float):

    """
    Aggiunge il worldbody e tutti i body

    :param root_xml:
    :param mass: massa del lenzuolo
    :param radius: raggio della particella
    :param damping: smorzamento
    :param poisson:
    :param young:
    :param thickness: spessore
    :param pos: posizione del centro di massa
    :param dimension: numero di nodi nelle tre dimensioni
    :param spacing: distanza fra due nodi adiacenti nelle tre dimensioni
    :param posizione_manoDx: posizione della mano destra espressa in percentuale (0 è al centro, 1 è al lembo)
    :param posizione_manoSx: posizione della mano sinistra espressa in percentuale (0 è al centro, 1 è al lembo)
    :return:
    """

    worldbody = root_xml.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root_xml, 'worldbody')

    # LIGHT
    light_element = ET.SubElement(worldbody, 'light',
                                  attrib={'diffuse': '0.6 0.6 0.6',
                                          'specular': '0.2 0.2 0.2',
                                          'pos': '0 0 4',
                                          'dir': '0 0 -1'})

    # BODY 0
    origin_element = ET.SubElement(worldbody, 'body',
                                   attrib={'name': 'origin',
                                           'mocap': 'true'})

    # BODY 1
    sferaCentrale_element = ET.SubElement(worldbody, 'body',
                              attrib={'name': 'sferaCentrale',
                                      'mocap': 'true',
                                      'pos': f'0 '
                                             f'{-(dimension[1] - 1) * spacing[1] / 2} '
                                             f'{pos[2]}'})
    # ET.SubElement(sferaCentrale_element, 'site',
    #               attrib={'name': "Z_0",
    #                       'pos': '0 0 0.075',
    #                       'size': '0.003 0.003 0.1',
    #                       'type': 'box',
    #                       'rgba': '0 0 1 1'})
    # ET.SubElement(sferaCentrale_element, 'site',
    #               attrib={'name': "Y_0",
    #                       'pos': '0 0.075 0',
    #                       'size': '0.003 0.1 0.003',
    #                       'type': 'box',
    #                       'rgba': '0 1 0 1'})
    # ET.SubElement(sferaCentrale_element, 'site',
    #               attrib={'name': "X_0",
    #                       'pos': '0.075 0 0',
    #                       'size': '0.1 0.003 0.003',
    #                       'type': 'box',
    #                       'rgba': '1 0 0 1'})
    # CAMERA
    ET.SubElement(sferaCentrale_element, 'camera',
                  attrib={'name': 'azure_kinect',
                          'mode': 'fixed',
                          'resolution': '512 512',
                          'fovy': '120',
                          'pos': '0 0.311 0.708',
                          'euler': '39 0 0'})

    # BODY 2
    manoSx_element = ET.SubElement(worldbody, 'body',
                           attrib={'name': 'manoSx',
                                   'mocap': 'true',
                                   'pos': f'{(dimension[0] - 1) * spacing[0] / 2 * posizione_manoSx } '
                                          f'{pos[1] + (dimension[1] - 1) * spacing[1] / 2 } '
                                          f'{pos[2]}'})
    # BODY 3
    manoDx_element = ET.SubElement(worldbody, 'body',
                           attrib={'name': 'manoDx',
                                   'mocap': 'true',
                                   'pos': f'{-(dimension[0] - 1) * spacing[0] / 2 * posizione_manoDx } '
                                          f'{pos[1] + (dimension[1] -1 ) * spacing[1] / 2 } '
                                          f'{pos[2]}'})

    # BODY 4
    pin_element = ET.SubElement(worldbody, 'body',
                        attrib={'name': 'pin',
                                'pos': f'{pos[0]} {pos[1]} {pos[2]}'})
    flexcomp_element= ET.SubElement(pin_element, 'flexcomp',
                          attrib={'name': 'flag',
                                  'type': 'grid',
                                  'dim': '2',
                                  'count': f'{dimension[0]} {dimension[1]} {dimension[2]}',
                                  'spacing': f'{spacing[0]} {spacing[1]} {spacing[2]}',
                                  'mass': f'{mass}',
                                  'radius': f'{radius}'})
    ET.SubElement(flexcomp_element, 'edge',
                  attrib={'equality': 'true',
                           'damping': f'{damping}'})
    plugin_element = ET.SubElement(flexcomp_element, 'plugin',
                        attrib={'plugin':'mujoco.elasticity.shell'})
    ET.SubElement(plugin_element, 'config',
                  attrib={'key': 'thickness',
                          'value': f'{thickness}'})
    ET.SubElement(plugin_element, 'config',
                  attrib={'key': 'poisson',
                          'value': f'{poisson}'})
    ET.SubElement(plugin_element, 'config',
                  attrib={'key': 'young',
                          'value': f'{young}'})

    # BODY 5
    ternaUomo_element = ET.SubElement(worldbody, 'body',
                                attrib={'name': 'ternaUomo',
                                        'mocap': 'true',
                                        'pos': f'{(dimension[0] - 1) * spacing[0] / 2 * (posizione_manoSx - posizione_manoDx)} '
                                               f'{(dimension[1] - 1) * spacing[1] / 2} '
                                               f'{pos[2]}'})

    return root_xml

def connect_maker(root_xml: ET.Element,
                  dimension: List[float],
                  spacing: List[float],
                  posizione_manoDx: float,
                  posizione_manoSx: float):
    """
    Aggiunge i connect lato robot e lato uomo

    :param root_xml:
    :param dimension: numero di nodi nelle tre dimensioni
    :param spacing: distanza fra due nodi adiacenti nelle tre dimensioni
    :param posizione_manoDx: posizione della mano destra espressa in percentuale (0 è al centro, 1 è al lembo)
    :param posizione_manoSx: posizione della mano sinistra espressa in percentuale (0 è al centro, 1 è al lembo)
    :return:
    """

    equality = ET.SubElement(root_xml, 'equality')

    # robot grasping
    for i in range(dimension[0] * dimension[1]):
        if i == 0:  # first node of the flag
            ii = 0
        else:
            if i % dimension[1] == 0:
                ii = i
        if i == ii:
            ET.SubElement(equality, 'connect',
                          attrib={'body1': f'flag_{ii}',
                                  'body2': 'sferaCentrale',
                                  'anchor': '0 0 0',
                                  'solref': '-1000 -100'})

    # calcolo delle coordinate dei nodi del lenzuolo
    flag_coordinate = []
    flag_0_pos = [-(dimension[0] - 1) * spacing[0] / 2,
                  -(dimension[1] - 1) * spacing[1] / 2]

    for i in range(dimension[0] * dimension[1] * dimension[2]):
        if i == 0:
            flag_coordinate.append(flag_0_pos)
            # j += 1
        else:

            if i % dimension[1] != 0:
                flag_coordinate.append(np.array(flag_coordinate[i - 1]) + np.array([0, spacing[1]]))
            else:
                flag_coordinate.append(np.array(flag_coordinate[i - 1]) + np.array([spacing[0], - (dimension[1] - 1) * spacing[1]]))

    # human grasping
    # dimensione mano (rettangolo AxB)
    A = 0.10     # LARGHEZZA MANO
    B = 0.13    # LUNGHEZZA MANO

    # mano Sx
    centro_mano_Sx = [(dimension[0] - 1) * spacing[0] / 2 * posizione_manoSx,
                      (dimension[1] - 1) * spacing[1] / 2 ]
    lim_mano_Sx = [centro_mano_Sx[0] - A/2, centro_mano_Sx[0] + A/2,    # larghezza mano
                   centro_mano_Sx[1] - B, centro_mano_Sx[1]]            # lunghezza mano

    for i in range(len(flag_coordinate)):
        if lim_mano_Sx[0] - flag_coordinate[i][0] <= 0.001 and lim_mano_Sx[1] - flag_coordinate[i][0] >= -0.001:
            if lim_mano_Sx[2] - flag_coordinate[i][1] <= 0.001 and lim_mano_Sx[3] - flag_coordinate[i][1] >=  -0.001:
                ET.SubElement(equality, 'connect',
                              attrib={'body1': f'flag_{i}',
                                      'body2': 'manoSx',
                                      'anchor': '0 0 0',
                                      'solref': '-1000 -100'})

    # mano Dx
    centro_mano_Dx = [-(dimension[0] - 1) * spacing[0] / 2 * posizione_manoDx,
                      (dimension[1] - 1) * spacing[1] / 2]
    lim_mano_Dx = [centro_mano_Dx[0] - A/2, centro_mano_Dx[0] + A/2,    # larghezza mano
                   centro_mano_Dx[1] - B, centro_mano_Dx[1]]            # lunghezza mano

    for i in range(len(flag_coordinate)):
        if lim_mano_Dx[0] - flag_coordinate[i][0] <= 0.001  and lim_mano_Dx[1] - flag_coordinate[i][0] >= -0.001:
            if lim_mano_Dx[2] - flag_coordinate[i][1] <= 0.001  and  lim_mano_Dx[3] - flag_coordinate[i][1] >=  -0.001:
                ET.SubElement(equality, 'connect',
                              attrib={'body1': f'flag_{i}',
                                      'body2': 'manoDx',
                                      'anchor': '0 0 0',
                                      'solref': '-1000 -100'})

    # # Vincolo solo lembi
    # ET.SubElement(equality, 'connect',
    #               attrib={'body1': f'flag_{dimension[1]-1}',
    #                       'body2': 'manoDx',
    #                       'anchor': '0 0 0'})
    # ET.SubElement(equality, 'connect',
    #               attrib={'body1': f'flag_{dimension[1]*dimension[0]-1}',
    #                       'body2': 'manoSx',
    #                       'anchor': '0 0 0'})

    return root_xml


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
