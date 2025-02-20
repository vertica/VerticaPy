"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
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
from typing import Literal, Optional

import numpy as np

from verticapy._typing import HChart
from verticapy.plotting._highcharts.base import HighchartsBase


class BarChart(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "chart": {"type": "column"},
            "legend": {"enabled": False},
            "colors": [self.get_colors(idx=0)],
            "xAxis": {
                "type": "category",
                "title": {"text": self.layout["column"]},
                "categories": self.layout["labels"],
            },
            "yAxis": {"title": {"text": self.layout["method_of"]}},
            "tooltip": {"headerFormat": "", "pointFormat": "{point.y}"},
        }
        self.init_style_bar = {"pointPadding": self.data["bargap"] / 2}

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a BarChart using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        chart.add_data_set(
            [float(val) for val in self.data["y"]],
            "bar",
            self.layout["column"],
            **self.init_style_bar,
        )
        return chart


class BarChart2D(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "chart": {"type": "column"},
            "xAxis": {"type": "category"},
            "colors": self.get_colors(),
            "xAxis": {
                "title": {"text": self.layout["columns"][0]},
                "categories": self.layout["x_labels"],
            },
            "yAxis": {"title": {"text": self.layout["method_of"]}},
            "legend": {"enabled": True, "title": {"text": self.layout["columns"][1]}},
        }
        self.init_style_stacked = {"plotOptions": {"series": {"stacking": "normal"}}}

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a 2D BarChart using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        for idx, label in enumerate(self.layout["y_labels"]):
            chart.add_data_set(list(self.data["X"][:, idx]), "bar", name=str(label))
        if self.layout["kind"] == "stacked":
            chart.set_dict_options(self.init_style_stacked)
        return chart


class DrillDownBarChart(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["rollup"]:
        return "rollup"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "chart": {"type": "column"},
            "title": {"text": ""},
            "subtitle": {"text": ""},
            "xAxis": {"type": "category"},
            "yAxis": {"title": {"text": self.layout["method_of"]}},
            "legend": {"enabled": False},
            "plotOptions": {
                "series": {"borderWidth": 0, "dataLabels": {"enabled": True}}
            },
            "tooltip": {
                "headerFormat": "",
                "pointFormat": '<span style="color:{point.color}">{point.name}</span>: <b>{point.y}</b><br/>',
            },
            "colors": self.get_colors(),
        }

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a 2D BarChart using the HC API.
        """
        kind = "bar" if self._kind == "barh" else "column"
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        chart.add_JSsource("https://code.highcharts.com/6/modules/drilldown.js")
        n_groups = len(self.data["groups"])
        
        if n_groups < 3:
            print("less than 3 groups")
            init_group = np.column_stack(self.data["groups"][1])
            data = []
            for row in init_group:
                data += [
                    {"name": str(row[0]), "y": float(row[1]), "drilldown": str(row[0])}
                ]
            print("data: \n", data)
            chart.add_data_set(data, kind, colorByPoint=True)
            print("SELF DATA ALL: \n", self.data)
            print("self.data['groups'][0]:", self.data["groups"][0])
            try:
                print("self.data['groups'][1]:", self.data["groups"][1])
            except:
                pass
            drilldown_group = np.column_stack(self.data["groups"][0])
            print("drilldown_group:", drilldown_group)
            uniques = np.unique(drilldown_group[:, 0])
            for c in uniques:
                data = drilldown_group[drilldown_group[:, 0] == c].tolist()
                data = [(str(x[1]), float(x[2])) for x in data]
                print(f"data for each {c}:", data)
                chart.add_drilldown_data_set(data, kind, str(c), name=str(c))
        else:
            # Top-level data
            data = [
                {'name': 'Asia',   'y': 4000000000, 'drilldown': 'Asia'},
                {'name': 'Africa', 'y': 1300000000, 'drilldown': 'Africa'},
                {'name': 'Europe', 'y': 800000000,  'drilldown': 'Europe'}
            ]
            chart.add_data_set(data, kind, colorByPoint=True)

            # First drilldown level: Countries for each continent
            # For each country we list: [Continent, Country, Population]
            self.data['groups'][0] = [
                # Continent names (one per country)
                ['Asia',   'Asia',   'Asia',
                'Africa', 'Africa', 'Africa',
                'Europe', 'Europe', 'Europe'],
                # Country names
                ['China',  'India',  'Indonesia',
                'Nigeria','Ethiopia','Egypt',
                'Germany','France', 'UK'],
                # Population values (as strings)
                ['1600000000', '1200000000', '1200000000',
                '500000000',  '400000000',  '400000000',
                '300000000',  '250000000',  '250000000']
            ]
            self.data['groups'][1] = [
                # Continent repeated for each breakdown entry
                ['Asia', 'Asia',    # China
                'Asia', 'Asia',    # India
                'Asia', 'Asia',    # Indonesia
                'Africa', 'Africa',# Nigeria
                'Africa', 'Africa',# Ethiopia
                'Africa', 'Africa',# Egypt
                'Europe', 'Europe',# Germany
                'Europe', 'Europe',# France
                'Europe', 'Europe'],# UK
                # Country names (repeated for each breakdown)
                ['China', 'China',
                'India', 'India',
                'Indonesia', 'Indonesia',
                'Nigeria', 'Nigeria',
                'Ethiopia', 'Ethiopia',
                'Egypt', 'Egypt',
                'Germany', 'Germany',
                'France', 'France',
                'UK', 'UK'],
                # Breakdown labels (e.g. Urban and Rural)
                ['Urban', 'Rural'] * 9,
                # Population breakdown values (as strings; must sum to the country's total)
                [
                    # China: Total 1,600,000,000 → Urban 800M, Rural 800M
                    '800000000', '800000000',
                    # India: Total 1,200,000,000 → Urban 400M, Rural 800M
                    '400000000', '800000000',
                    # Indonesia: Total 1,200,000,000 → Urban 600M, Rural 600M
                    '600000000', '600000000',
                    # Nigeria: Total 500,000,000 → Urban 200M, Rural 300M
                    '200000000', '300000000',
                    # Ethiopia: Total 400,000,000 → Urban 150M, Rural 250M
                    '150000000', '250000000',
                    # Egypt: Total 400,000,000 → Urban 250M, Rural 150M
                    '250000000', '150000000',
                    # Germany: Total 300,000,000 → Urban 200M, Rural 100M
                    '200000000', '100000000',
                    # France: Total 250,000,000 → Urban 150M, Rural 100M
                    '150000000', '100000000',
                    # UK: Total 250,000,000 → Urban 160M, Rural 90000000
                    '160000000', '90000000'
                ]
            ]
            chart.add_data_set(data, kind, colorByPoint=True)


            # First drilldown level
            drilldown_group = np.column_stack(self.data["groups"][0])
            uniques = np.unique(drilldown_group[:, 0])
            for c in uniques:
                data = drilldown_group[drilldown_group[:, 0] == c].tolist()
                # Add drilldown key for next level
                data_points = [{
                    'name': str(x[1]), 
                    'y': float(x[2]), 
                    'drilldown': f"{c}-{x[1]}"
                } for x in data]
                chart.add_drilldown_data_set(data_points, kind, str(c), name=str(c))

            # Second drilldown level
            drilldown_level2 = np.column_stack(self.data["groups"][1])
            unique_keys = np.unique(drilldown_level2[:, 0:2], axis=0)
            for key in unique_keys:
                parent_id, x_axis_label = key[0], key[1]
                drilldown_id = f"{parent_id}-{x_axis_label}"
                mask = (drilldown_level2[:, 0] == parent_id) & (drilldown_level2[:, 1] == x_axis_label)
                data_subset = drilldown_level2[mask]
                data = [{'name': str(row[2]), 'y': float(row[3])} for row in data_subset]
                chart.add_drilldown_data_set(data, kind, drilldown_id, name=f"{x_axis_label}")

                
        return chart