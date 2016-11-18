#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Arne Neumann <discoursegraphs.programming@arne.cl>

"""
This module contains code that is used by more than one RST format importer.
Some code is taken from github.com/EducationalTestingService/discourse-parsing
(MIT license).
"""

import re

from discoursegraphs.readwrite.ptb import PTB_BRACKET_ESCAPE


SUBTREE_TYPES = ('root', 'nucleus', 'satellite')


def fix_rst_treebank_tree_str(rst_tree_str):
    '''
    This removes some unexplained comments in two files that cannot be parsed.
    - data/RSTtrees-WSJ-main-1.0/TRAINING/wsj_2353.out.dis
    - data/RSTtrees-WSJ-main-1.0/TRAINING/wsj_2367.out.dis

    source: github.com/EducationalTestingService/discourse-parsing
    original license: MIT
    '''
    return re.sub(r'\)//TT_ERR', ')', rst_tree_str)


def convert_parens_in_rst_tree_str(rst_tree_str):
    '''
    This converts any brackets and parentheses in the EDUs of the RST discourse
    treebank to look like Penn Treebank tokens (e.g., -LRB-),
    so that the NLTK tree API doesn't crash when trying to read in the
    RST trees.

    source: github.com/EducationalTestingService/discourse-parsing
    original license: MIT
    '''
    for bracket_type, bracket_replacement in PTB_BRACKET_ESCAPE.items():
        rst_tree_str = \
            re.sub('(_![^_(?=!)]*)\\{}([^_(?=!)]*_!)'.format(bracket_type),
                   '\\g<1>{}\\g<2>'.format(bracket_replacement),
                   rst_tree_str)
    return rst_tree_str

