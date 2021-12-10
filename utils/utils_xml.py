from xml.etree import ElementTree
from xml.dom import minidom

import logging

logger = logging.getLogger(__name__)
log = logger


def xml_prettify(elem):
    ''' Returns a pretty-printed XML string for the Element.
        Inspired from: https://pymotw.com/2/xml/etree/ElementTree/create.html
    '''
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
