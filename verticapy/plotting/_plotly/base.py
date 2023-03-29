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
from verticapy.plotting.base import PlottingBase
from plotly.graph_objs._figure import Figure
import plotly.graph_objects as go
import numpy as np


class PlotlyBase(PlottingBase):
    @staticmethod
    def _convert_labels_and_get_counts(pivot_array):
        pivot_array = np.where(pivot_array == None, "NULL", pivot_array)
        pivot_array = pivot_array.astype("<U21")
        pivot_array = pivot_array.astype(str)
        pivot_array[1:-1, :] = np.char.add(pivot_array[1:-1, :], "__")
        if pivot_array.shape[0] > 3:
            pivot_array[1:-1, :] = np.char.add(
                np.char.add(pivot_array[1:-1, :], pivot_array[:-2, :]),
                pivot_array[:-3, :],
            )
        else:
            pivot_array[1:-1, :] = np.char.add(
                pivot_array[1:-1, :], pivot_array[:-2, :]
            )
        labels_count = {}
        labels_father = {}
        for j in range(pivot_array.shape[0] - 1):
            for i in range(len(pivot_array[0])):
                current_label = pivot_array[-2][i]
                if current_label not in labels_count:
                    labels_count[current_label] = 0
                labels_count[current_label] += int(pivot_array[-1][i])
                if pivot_array.shape[0] > 2:
                    labels_father[current_label] = pivot_array[-3][i]
                else:
                    labels_father[current_label] = ""
            pivot_array = np.delete(pivot_array, -2, axis=0)
        labels = [s.split("__")[0] for s in list(labels_father.keys())]
        ids = list(labels_count.keys())
        parents = list(labels_father.values())
        values = list(labels_count.values())
        return ids, labels, parents, values

    @staticmethod
    def _get_fig(fig) -> tuple[Figure]:
        if fig:
            return fig
        else:
            return go.Figure()
