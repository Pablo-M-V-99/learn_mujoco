import xml.etree.ElementTree as ET
from typing import List


def lenzuolo_maker(root_xml: ET.Element,
                   mass: float,
                   radius: float,
                   damping: float,
                   poisson: float,
                   young: float,
                   thickness: float,
                   pos: List[float],
                   dimension: List[float],
                   spacing: List[float]):

    worldbody = root_xml.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root_xml, 'worldbody')

    origin_element = ET.SubElement(worldbody, 'body',
                                   attrib={'name': 'origin',
                                           'mocap': 'true'})

    sferaCentrale_element = ET.SubElement(worldbody, 'body',
                              attrib={'name': 'sferaCentrale',
                                      'mocap': 'true',
                                      'pos': f'0 {-(dimension[1]-1)*spacing[1]/2} {pos[2]}'})

    ET.SubElement(sferaCentrale_element, 'camera',
                  attrib={'name': 'azure_kinect',
                          'mode': 'fixed',
                          'resolution': '512 512',
                          'fovy': '120',
                          'pos': '0 0.1 0.5',
                          'euler': '45 0 0'})

    manoSx_element = ET.SubElement(worldbody, 'body',
                           attrib={'name': 'manoSx',
                                   'mocap': 'true',
                                   'pos': f'{pos[0] + (dimension[0]-1)*spacing[0]/2 + 0.02} '
                                          f'{pos[1] + (dimension[1]-1)*spacing[1]/2 + 0.02} '
                                          f'{pos[2]}'})

    manoDx_element = ET.SubElement(worldbody, 'body',
                           attrib={'name': 'manoDx',
                                   'mocap': 'true',
                                   'pos': f'{pos[0] - (dimension[0]-1)*spacing[0]/2 + 0.02} '
                                          f'{pos[1] + (dimension[1]-1)*spacing[1]/2 + 0.02} '
                                          f'{pos[2]}'})


    pin_element = ET.SubElement(worldbody, 'body',
                        attrib={'name': 'pin',
                                'pos': f'{pos[0]} {pos[1]} {pos[2]}'})
    flexcomp_element= ET.SubElement(pin_element, 'flexcomp',
                          attrib={'name': 'flag',
                                  'type': 'grid',
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


    return root_xml

def connect_maker(root_xml: ET.Element,
                  dimension: List[float],):

    # if equality is None:
    equality = ET.SubElement(root_xml, 'equality')

    # robot grasping
    for i in range(dimension[0]):
        if i == 0:  # first node of the flag
            ii = 0
        else:
            if i % 2 != 0:  # odd nodes
                ii += (2*dimension[1] - 1)
            else:           # even nodes
                ii += 1
        ET.SubElement(equality, 'connect',
                      attrib={'body1': f'flag_{ii}',
                              'body2': 'sferaCentrale',
                              'anchor': '0 0 0',
                              'solref': '-1000 -100' })

    # human grasping
    ET.SubElement(equality, 'connect',
                  attrib={'body1': f'flag_{dimension[1]-1}',
                          'body2': 'manoDx',
                          'anchor': '0 0 0'})
    ET.SubElement(equality, 'connect',
                  attrib={'body1': f'flag_{dimension[1]*dimension[0]-1}',
                          'body2': 'manoSx',
                          'anchor': '0 0 0'})

    return root_xml
