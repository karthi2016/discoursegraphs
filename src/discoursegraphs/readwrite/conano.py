#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Arne Neumann

"""
This script parses a Conano XML file, extracts all connectives and
writes them to an output file. You can choose between three output
formats.

- normal: one connective per line
- relations: each line contains one connective and the type of relation
  it belongs to (tab-separated)
- units: prints the connective as well as its units, e.g.

'''
=====
Aber

Dass darunter einige wenige leiden müssen , ist leider unvermeidbar .

Aber das muss irgendwann ein Ende haben .
'''
"""

import sys
from collections import OrderedDict
from lxml import etree
import argparse
import re
import pudb  # TODO: rm debug

from discoursegraphs import DiscourseDocumentGraph
from discoursegraphs.util import ensure_unicode

REDUCE_WHITESPACE_RE = re.compile(' +')


class ConanoDocumentGraph(DiscourseDocumentGraph):
    """
    represents a Conano XML file as a multidigraph.

    Attributes
    ----------
    tokens : list of int
        a list of node IDs (int) which represent the tokens in the
        order they occur in the text
    root : str
        name of the document root node ID
        (default: 'conano:root_node')    
    """
    def __init__(self, conano_file, name=None):
        """
        reads a Conano XML file and converts it into a multidigraph.
        
        Parameters
        ----------
        conano_file : str or _ElementTree
            relative or absolute path to a Conano XML file (or an
            ``lxml.etree._ElementTree`` representing the file)
        name : str or None
            the name or ID of the graph to be generated. If no name is
            given, the basename of the input file is used.
        """
        # super calls __init__() of base class DiscourseDocumentGraph
        super(DiscourseDocumentGraph, self).__init__()

        if name is not None:
            self.name = os.path.basename(conano_filepath)
        self.root = 'conano:root_node'
        self.add_node(self.root, layers={'conano'})

        if isinstance(conano_file, etree._ElementTree):
            self.tree = conano_file
        else:
            self.tree = etree.parse(conano_file)
        conano_plaintext = etree.tostring(self.tree, encoding='utf8',
                                          method='text')
        tokens = conano_plaintext.split()

        self.tokens = []
        for i, token in enumerate(tokens):
            self.__add_token_to_document(token, i)
            self.tokens.append(i)

    def __add_token_to_document(self, token, token_id):
        """
        adds a token to the document graph as a node with the given ID.
        adds an edge of type ``contains`` from the root node to the
        token node.

        TODO: only add edges from root to token if the token isn't part
        of a unit span...

        Parameters
        ----------
        token : str
            the token to be added to the document graph
        token_id : int
            the node ID of the token to be added, which must not yet
            exist in the document graph
        """
        self.add_node(
            token_id,
            layers={'conano', 'conano:token'},
            attr_dict={'conano:token': ensure_unicode(token)})

        self.add_edge(self.root, token_id,
                      layers={'conano', 'conano:token'},
                      edge_type='contains')


def get_connectives(tree):
    """
    extracts connectives from a Conano XML file.

    Note: There can be more than one connective with the same ID (e.g.
    'je' and 'desto')

    Parameters
    ----------
    tree : lxml.etree._ElementTree
        an element tree representing the Conano XML file to be parsed

    Returns
    -------
    connectives : OrderedDict
        an ordered dictionary which maps from a connective ID (int) to a
        list of dictionaries.
        each dict represents one connective by its features ('text' maps
        to the connective (str), 'relation' maps to the relation (str)
        the connective is part of and 'modifier' maps to the modifier
        (str or None) of the connective
    """
    connectives = OrderedDict()
    connective_elements = tree.findall('//connective')
    for element in connective_elements:
        try:
            conn_id = int(element.attrib['id'])
            conn_feats = {
                'text': get_connective_string(element),
                'relation': element.attrib['relation'],
                'modifier': get_modifier(element)}

            if conn_id in connectives:
                connectives[conn_id].append(conn_feats)
            else:
                connectives[conn_id] = [conn_feats]

        except KeyError as e:
            sys.stderr.write(
                ("Something's wrong in file {0}.\nThere's no {1} "
                 "attribute in element:\n{2}"
                 "\n".format(tree.docinfo.URL, e, etree.tostring(element))))
    return connectives


def get_units(tree):
    """
    extracts connectives and their internal and external units from a
    Conano XML file.

    Parameters
    ----------
    tree : lxml.etree._ElementTree
        an element tree representing the Conano XML file to be parsed

    Returns
    -------
    ext_units : OrderedDict
        an ordered dictionary which maps from a connective ID (int) to
        the external unit (str) of that connective
    int_units : OrderedDict
        an ordered dictionary which maps from a connective ID (int) to
        the internal unit (str) of that connective
    """
    ext_units = OrderedDict()
    int_units = OrderedDict()
    for unit in tree.findall('//unit'):
        unit_str = etree.tostring(unit, encoding='utf8',
                                  method='text').replace('\n', ' ')
        cleaned_str = REDUCE_WHITESPACE_RE.sub(' ', unit_str).strip()

        if unit.attrib['type'] == 'ext':
            ext_units[int(unit.attrib['id'])] = cleaned_str
        else:
            int_units[int(unit.attrib['id'])] = cleaned_str
    return ext_units, int_units


def get_connective_string(connective_element):
    """
    given an etree element representing a connective, returns the
    connective (str).

    Parameters
    ----------
    connective_element : lxml.etree._Element
        An etree elements that contains a connective, which might
        additionally be modified, e.g.
        <connective id="5" relation="consequence">
            <modifier>auch</modifier>
            deshalb
        </connective>

    Results
    -------
    result : str
        a string representing the (modified) connective,
        e.g. 'und' or 'auch deshalb'
    """
    if connective_element.text is None:  # has a modifier
        modifier = connective_element.getchildren()[0]
        return ' '.join([modifier.text.strip(), modifier.tail.strip()])

    else:
        return connective_element.text.strip()


def get_modifier(connective_element):
    """
    returns the modifier (str) of a connective or None, if the
    connective has none.

    Parameters
    ----------
    connective_element : lxml.etree._Element
        An etree elements that contains a connective

    Results
    -------
    result : str or None
        a string representing the modifier or None
    """
    if connective_element.xpath('modifier'):
        return connective_element.getchildren()[0].text.strip()
    else:
        return None


def get_ancestor_units(unit_element):
    pudb.set_trace()
    results = ["{0}-{1}".format(unit_element.attrib['type'],
                                unit_element.attrib['id'])]
    parent_element = unit_element.getparent()
    if parent_element.tag == 'discourse':
        return results
    elif parent_element.tag == 'unit':
        results.append(get_ancestor_units(parent_element))
    else:
        raise ValueError(
            ('Connective {0} embedded in unknown element '
             '{1}!'.format(results[0], parent_element.tag)))


def write_connectives(connectives, outfile):
    """
    writes connectives to output file (one connective per line).
    """
    with outfile:
        for cid, clist in connectives.items():
            for connective in clist:
                conn_str = connective['text'].encode('utf8')
                outfile.write(conn_str + '\n')


def write_relations(connectives, outfile):
    """
    Writes connectives and their relations to an output file. Each line
    will contain one connective and the relation it belongs to (tab-separated).
    """
    with outfile:
        for cid, clist in connectives.items():
            for connective in clist:
                conn_str = connective['text'].encode('utf8')
                relation = connective['relation'].encode('utf8')
                outfile.write(conn_str + '\t' + relation + '\n')


def write_units(conano_etree, connectives, outfile):
    """
    """
    ext_units, int_units = get_units(conano_etree)
    with outfile:
        for cid, clist in connectives.items():
            for connective in clist:
                conn_str = connective['text'].encode('utf8')
                try:
                    extunit = ext_units[cid]
                except KeyError:
                    sys.stderr.write(
                        ("{0} has no ext-unit with ID {1}"
                         "\n".format(conano_etree.docinfo.URL, cid)))
                try:
                    intunit = int_units[cid]
                except KeyError:
                    sys.stderr.write(
                        ("{0} has no int-unit with ID {1}"
                         "\n".format(conano_etree.docinfo.URL, cid)))
                outfile.write('=====\n' + conn_str + '\n\nEXTERN: ' +
                              extunit + '\n\nINTERN: ' + intunit + '\n\n\n')


def cli():
    """
    command line interface for extracting connectives from Conano XML
    files.
    """
    desc = ("This script extracts connectives (and its relation "
            "type, and int/ext-units) from Conano XML files.")
    infile_help = ("Conano XML file to be parsed. If no filename is given: "
                   "read from stdin.")
    outfile_help = ("the file that shall contain the connectives. If no "
                    "filename is given: write to stdout.")
    outformat_help = ("output file format: 'normal', 'relations' or 'units'\n"
                      "Defaults to normal, which just prints the connectives.")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help=infile_help)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout, help=outfile_help)

    parser.add_argument('-f', '--format', dest='outformat',
                        help=outformat_help)

    args = parser.parse_args()
    conano_file = args.infile
    outfile = args.outfile

    try:
        tree = etree.parse(conano_file)
        connectives = get_connectives(tree)

        if args.outformat in (None, 'normal'):
            write_connectives(connectives, outfile)

        elif args.outformat == 'relations':
            write_relations(connectives, outfile)

        elif args.outformat == 'units':
            write_units(tree, connectives, outfile)

        elif args.outformat == 'dot':
            from networkx import write_dot
            conano_docgraph = ConanoDocumentGraph(tree)
            write_dot(conano_docgraph, outfile)

        else:
            sys.stderr.write("Unsupported output format.\n")
            parser.print_help()
            sys.exit(1)

    except etree.XMLSyntaxError as e:
        sys.stderr.write("Can't parse file {0}. {1}\n".format(conano_file, e))


if __name__ == "__main__":
    cli()