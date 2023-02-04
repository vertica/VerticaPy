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
import verticapy

COLORS_OPTIONS = {
    "rgb": ["red", "green", "blue", "orange", "yellow", "gray"],
    "sunset": ["#36688D", "#F3CD05", "#F49F05", "#F18904", "#BDA589"],
    "retro": ["#A7414A", "#282726", "#6A8A82", "#A37C27", "#563838"],
    "shimbg": ["#0444BF", "#0584F2", "#0AAFF1", "#EDF259", "#A79674"],
    "swamp": ["#6465A5", "#6975A6", "#F3E96B", "#F28A30", "#F05837"],
    "med": ["#ABA6BF", "#595775", "#583E2E", "#F1E0D6", "#BF9887"],
    "orchid": ["#192E5B", "#1D65A6", "#72A2C0", "#00743F", "#F2A104"],
    "magenta": ["#DAA2DA", "#DBB4DA", "#DE8CF0", "#BED905", "#93A806"],
    "orange": ["#A3586D", "#5C4A72", "#F3B05A", "#F4874B", "#F46A4E"],
    "vintage": ["#80ADD7", "#0ABDA0", "#EBF2EA", "#D4DCA9", "#BF9D7A"],
    "vivid": ["#C0334D", "#D6618F", "#F3D4A0", "#F1931B", "#8F715B"],
    "berries": ["#BB1924", "#EE6C81", "#F092A5", "#777CA8", "#AFBADC"],
    "refreshing": ["#003D73", "#0878A4", "#1ECFD6", "#EDD170", "#C05640"],
    "summer": ["#728CA3", "#73C0F4", "#E6EFF3", "#F3E4C6", "#8F4F06"],
    "tropical": ["#7B8937", "#6B7436", "#F4D9C1", "#D72F01", "#F09E8C"],
    "india": ["#F1445B", "#65734B", "#94A453", "#D9C3B1", "#F03625"],
    "default": ["#FE5016", "#263133", "#0073E7", "#FDE159", "#33C180", "#FF454F"],
}

def gen_cmap(color: str = "", reverse: bool = False):
    if not (color):
        cm1 = LinearSegmentedColormap.from_list(
            "vml", ["#FFFFFF", gen_colors()[0]], N=1000
        )
        cm2 = LinearSegmentedColormap.from_list(
            "vml", [gen_colors()[1], "#FFFFFF", gen_colors()[0]], N=1000
        )
        return (cm1, cm2)
    else:
        if isinstance(color, list):
            return LinearSegmentedColormap.from_list("vml", color, N=1000)
        elif reverse:
            return LinearSegmentedColormap.from_list("vml", [color, "#FFFFFF"], N=1000)
        else:
            return LinearSegmentedColormap.from_list("vml", ["#FFFFFF", color], N=1000)


def gen_colors():
    if not (verticapy.OPTIONS["colors"]) or not (
        isinstance(verticapy.OPTIONS["colors"], list)
    ):
        if not (verticapy.OPTIONS["colors"]):
            colors = COLORS_OPTIONS["default"]
        else:
            colors = copy.deepcopy(verticapy.OPTIONS["colors"])
        all_colors = [item for item in plt_colors.cnames]
        shuffle(all_colors)
        for c in all_colors:
            if c not in colors:
                colors += [c]
        return colors
    else:
        return verticapy.OPTIONS["colors"]
