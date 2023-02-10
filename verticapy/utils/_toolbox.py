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
