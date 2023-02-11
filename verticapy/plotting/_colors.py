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
# Standard Modules
import copy
from random import shuffle

# MATPLOTLIB
import matplotlib.colors as plt_colors
from matplotlib.colors import LinearSegmentedColormap

# VerticaPy Modules
from verticapy._config.config import OPTIONS


def get_color(d: dict, idx: int = 0):
    if "color" in d:
        if isinstance(d["color"], str):
            return d["color"]
        else:
            return d["color"][idx % len(d["color"])]
    else:
        return gen_colors()[idx % len(gen_colors())]


def gen_cmap(color: str = "", reverse: bool = False):
    if not (color):
        cm1 = LinearSegmentedColormap.from_list(
            "verticapy", ["#FFFFFF", gen_colors()[0]], N=1000
        )
        cm2 = LinearSegmentedColormap.from_list(
            "verticapy", [gen_colors()[1], "#FFFFFF", gen_colors()[0]], N=1000
        )
        return (cm1, cm2)
    else:
        if isinstance(color, list):
            return LinearSegmentedColormap.from_list("verticapy", color, N=1000)
        elif reverse:
            return LinearSegmentedColormap.from_list(
                "verticapy", [color, "#FFFFFF"], N=1000
            )
        else:
            return LinearSegmentedColormap.from_list(
                "verticapy", ["#FFFFFF", color], N=1000
            )


def gen_colors():
    if not (OPTIONS["colors"]) or not (isinstance(OPTIONS["colors"], list)):
        if not (OPTIONS["colors"]):
            colors = COLORS_OPTIONS["default"]
        else:
            colors = copy.deepcopy(OPTIONS["colors"])
        all_colors = [item for item in plt_colors.cnames]
        shuffle(all_colors)
        for c in all_colors:
            if c not in colors:
                colors += [c]
        return colors
    else:
        return OPTIONS["colors"]
