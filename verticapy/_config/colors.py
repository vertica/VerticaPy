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
import copy
from random import shuffle

import matplotlib.colors as plt

from verticapy._config.config import COLORS_OPTIONS, _options


def get_color(d: dict = {}, idx: int = 0):
    if "color" in d:
        if isinstance(d["color"], str):
            return d["color"]
        else:
            return d["color"][idx % len(d["color"])]
    elif not (d):
        if not (_options["colors"]) or not (isinstance(_options["colors"], list)):
            if not (_options["colors"]):
                colors = COLORS_OPTIONS["default"]
            else:
                colors = copy.deepcopy(_options["colors"])
            all_colors = [item for item in plt.cnames]
            shuffle(all_colors)
            for c in all_colors:
                if c not in colors:
                    colors += [c]
            return colors
        else:
            return _options["colors"]
    else:
        return get_color()[idx % len(get_color())]


def get_cmap(color: str = "", reverse: bool = False):
    if not (color):
        cm1 = plt.LinearSegmentedColormap.from_list(
            "verticapy", ["#FFFFFF", get_color()[0]], N=1000
        )
        cm2 = plt.LinearSegmentedColormap.from_list(
            "verticapy", [get_color()[1], "#FFFFFF", get_color()[0]], N=1000
        )
        return (cm1, cm2)
    else:
        if isinstance(color, list):
            return plt.LinearSegmentedColormap.from_list("verticapy", color, N=1000)
        elif reverse:
            return plt.LinearSegmentedColormap.from_list(
                "verticapy", [color, "#FFFFFF"], N=1000
            )
        else:
            return plt.LinearSegmentedColormap.from_list(
                "verticapy", ["#FFFFFF", color], N=1000
            )
