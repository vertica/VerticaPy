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
from typing import Literal

import verticapy._config.config as conf

import verticapy.plotting._matplotlib as vpy_matplotlib_plt
import verticapy.plotting._plotly as vpy_plotly_plt


class PlottingUtils:
    @staticmethod
    def _get_plotting_lib(
        matplotlib_kwargs: dict = {}, plotly_kwargs: dict = {}, style_kwargs: dict = {}
    ) -> tuple[Literal[vpy_plotly_plt, vpy_matplotlib_plt], dict]:
        if conf.get_option("plotting_lib") == "plotly":
            vpy_plt = vpy_plotly_plt
            kwargs = {**plotly_kwargs, **style_kwargs}
        else:
            vpy_plt = vpy_matplotlib_plt
            kwargs = {**matplotlib_kwargs, **style_kwargs}
        return vpy_plt, kwargs
