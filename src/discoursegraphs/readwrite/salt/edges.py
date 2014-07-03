#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles the parsing of SALT edges.
"""

from discoursegraphs.readwrite.salt.util import get_xsi_type
from discoursegraphs.readwrite.salt.labels import SaltLabel
from discoursegraphs.readwrite.salt.elements import (SaltElement,
                                                     get_element_name,
                                                     get_graph_element_id,
                                                     get_annotations,
                                                     get_layer_ids,
                                                     get_subelements)

class SaltEdge(SaltElement):
    """
    An edge connects a source node with a target node, belongs to a layer
    and has two or more labels attached to it.
    """
    def __init__(self, name, element_id, xsi_type, labels, source, target,
                 layers=None, xml=None):
        """
        Every edge has these attributes (in addition to the attributes
        inherited from the ``SaltElement`` class):

        Attributes
        ----------
        layers : list of int or None
            list of indices of the layers that the edge belongs to,
            or ``None`` if the edge doesn't belong to any layer
        source : int
            the index of the source node connected to the edge
        target : int
            the index of the target node connected to the edge
        """
        super(SaltEdge, self).__init__(name, element_id, xsi_type, labels, xml)
        self.layers = layers
        self.source = source
        self.target = target

    @classmethod
    def from_etree(cls, etree_element):
        """
        creates a ``SaltEdge`` from the etree representation of an <edges>
        element from a SaltXMI file.
        """
        label_elements = get_subelements(etree_element, 'labels')
        labels = [SaltLabel.from_etree(elem) for elem in label_elements]
        return cls(name=get_element_name(etree_element),
                   element_id=get_graph_element_id(etree_element),
                   xsi_type=get_xsi_type(etree_element),
                   labels=labels,
                   source=get_node_id(etree_element, 'source'),
                   target=get_node_id(etree_element, 'target'),
                   layers=get_layer_ids(etree_element),
                   xml=etree_element)

    def __str__(self):
        ret_str = super(SaltEdge, self).__str__() + "\n"
        ret_str += "source node: {0}\n".format(self.source)
        ret_str += "target node: {0}".format(self.target)
        return ret_str

class SpanningRelation(SaltEdge):
    """
    Every SpanningRelation edge inherits all the attributes from `SaltEdge`
    (and `SaltElement`). A ``SpanningRelation`` is an ``Edgde`` that links a
    ``SpanNode`` to a ``TokenNode``.

    A SpanningRelation edge looks like this::

        <edges xsi:type="sDocumentStructure:SSpanningRelation" source="//@nodes.167" target="//@nodes.59" layers="//@layers.1">
            <labels xsi:type="saltCore:SFeature" namespace="salt" name="SNAME" valueString="sSpanRel27"/>
            <labels xsi:type="saltCore:SElementId" namespace="graph" name="id" valueString="edge181"/>
        </edges>

    """
    def __init__(self, name, element_id, xsi_type, labels, source, target,
                 layers=None, xml=None):
        """A ``SpanningRelation`` is created just like an ``SaltEdge``."""
        super(SpanningRelation, self).__init__(name, element_id, xsi_type,
                                               labels, source, target,
                                               layers=None, xml=None)


class TextualRelation(SaltEdge):
    """
    An TextualRelation edge always links a token (source node) to the
    PrimaryTextNode (target node 0). Textual relations don't belong to a layer.
    A TextualRelation contains the onset/offset of a token. This enables us to
    retrieve the text/string of a token from the documents primary text.

    Every TextualRelation has these attributes (in addition to those inherited
    from `SaltEdge` and `SaltElement`):

    Attributes
    ----------
    onset : int
        the string onset of the source node (``TokenNode``)
    offset : int
        the string offset of the source node (``TokenNode``)
    """
    def __init__(self, name, element_id, xsi_type, labels, source, target,
                 onset, offset, layers=None, xml=None):
        super(TextualRelation, self).__init__(name, element_id, xsi_type,
                                              labels, source, target,
                                              layers=None, xml=None)

    @classmethod
    def from_etree(cls, etree_element):
        """
        create a ``TextualRelation`` from an etree element representing
        an <edges> element with xsi:type 'sDocumentStructure:STextualRelation'.
        """
        #~ instance = super(TextualRelation, cls).from_etree(etree_element)
        #~ instance.onset = get_string_onset(etree_element)
        #~ instance.offset = get_string_offset(etree_element)
        #~ return instance
        # doesn't work
        raise NotImplementedError


class DominanceRelation(SaltEdge):
    """
    A `DominanceRelation` edge always links a `StructureNode` (source) to a
    `TokenNode` (target). Every `DominanceRelation` has a `feature` attribute:

    :ivar feature: `dict` of (`str`, `str`) key-value pairs which e.g. describe
    the syntactical constituent a token belongs to, such as {'tiger.func': 'PP'}.

    A DominanceRelation edge looks like this::

        <edges xsi:type="sDocumentStructure:SDominanceRelation" source="//@nodes.251" target="//@nodes.134" layers="//@layers.2">
            <labels xsi:type="saltCore:SFeature" namespace="saltCore" name="STYPE" valueString="edge"/>
            <labels xsi:type="saltCore:SFeature" namespace="salt" name="SNAME" valueString="sDomRel185"/>
            <labels xsi:type="saltCore:SElementId" namespace="graph" name="id" valueString="edge530"/>
            <labels xsi:type="saltCore:SAnnotation" name="tiger.func" valueString="OC"/>
        </edges>

    """
    def __init__(self, element, element_id, doc_id):
        super(DominanceRelation, self).__init__(element, element_id, doc_id)
        self.features = get_annotations(element)


def get_node_id(edge, node_type):
    """
    returns the source or target node id of an edge, depending on the
    node_type given.
    """
    assert node_type in ('source', 'target')
    _, node_id_str = edge.attrib[node_type].split('.') # e.g. //@nodes.251
    return int(node_id_str)

def get_string_onset(edge):
    onset_label = edge.find('labels[@name="SSTART"]')
    onset_str = onset_label.xpath('@valueString')[0]
    return int(onset_str)

def get_string_offset(edge):
    onset_label = edge.find('labels[@name="SEND"]')
    onset_str = onset_label.xpath('@valueString')[0]
    return int(onset_str)
