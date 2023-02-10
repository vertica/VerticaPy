"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""

#
#
# Modules
#
# Standard Python Modules
import shutil, re, sys, warnings, random, itertools, datetime, time, html, os
from collections.abc import Iterable
from typing import Union, Literal

# VerticaPy Modules
import verticapy as vp
from verticapy.errors import *
from verticapy.sql._utils._format import quote_ident

# Other Modules
import numpy as np
import pandas as pd

# IPython - Optional
try:
    from IPython.display import HTML, display
except:
    pass

#
#
# Functions to use to simplify the coding.
#
def bin_spatial_to_str(
    category: str, column: str = "{}",
):
    map_dict = {
        "vmap": f"MAPTOSTRING({column})",
        "binary": f"TO_HEX({column})",
        "spatial": f"ST_AsText({column})",
    }
    if category in map_dict:
        return map_dict[column]
    return column


def color_dict(d: dict, idx: int = 0):
    if "color" in d:
        if isinstance(d["color"], str):
            return d["color"]
        else:
            return d["color"][idx % len(d["color"])]
    else:
        from verticapy.plotting._colors import gen_colors

        return gen_colors()[idx % len(gen_colors())]


def find_val_in_dict(x: str, d: dict, return_key: bool = False):
    for elem in d:
        if quote_ident(x).lower() == quote_ident(elem).lower():
            if return_key:
                return elem
            return d[elem]
    raise NameError(f'Key "{x}" was not found in {d}.')


def get_match_index(x: str, col_list: list, str_check: bool = True):
    for idx, col in enumerate(col_list):
        if (str_check and quote_ident(x.lower()) == quote_ident(col.lower())) or (
            x == col
        ):
            return idx
    return None


def updated_dict(
    d1: dict, d2: dict, color_idx: int = 0,
):
    d = {}
    for elem in d1:
        d[elem] = d1[elem]
    for elem in d2:
        if elem == "color":
            if isinstance(d2["color"], str):
                d["color"] = d2["color"]
            elif color_idx < 0:
                d["color"] = [elem for elem in d2["color"]]
            else:
                d["color"] = d2["color"][color_idx % len(d2["color"])]
        else:
            d[elem] = d2[elem]
    return d
