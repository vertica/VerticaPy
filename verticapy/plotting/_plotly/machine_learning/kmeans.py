"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
import scipy.spatial as scipy_st

from verticapy.plotting._plotly.base import PlotlyBase


class VoronoiPlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["voronoi"]:
        return "voronoi"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 2)

    # Styling Methods.

    def _init_style(self) -> None:
        """Must be overridden in child class"""
        self.init_layout_style = {
            "width": 700,
            "height": 450,
            "xaxis": dict(
                title=self.layout["columns"][0],
                showgrid=False,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                zeroline=False,
            ),
            "yaxis": dict(
                title=self.layout["columns"][1],
                showgrid=False,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                zeroline=False,
            ),
        }
        self.init_line_style = {
            "mode": "lines",
            "line": dict(color="black", width=3),
            "showlegend": False,
        }

        self.init_scatter_style = {
            "mode": "markers",
            "marker_color": "red",
            "marker_size": 5,
        }
        self.init_cluster_scatter_style = {
            "mode": "markers",
            "marker_symbol": "x",
            "marker_size": 12,
        }
        self.init_heatmap_style = {
            "colorscale": "rainbow",
            "opacity": 0.4,
            "hoverinfo": "none",
            "showscale": False,
        }

    def _get_voronoi_lines(self, points: np.array) -> tuple[list, list]:
        vor = scipy_st.Voronoi(points)
        center = vor.points.mean(axis=0)
        ptp_bound = vor.points.ptp(axis=0)
        finite_segments = []
        infinite_segments = []
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite_segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if vor.furthest_site:
                    direction = -direction
                far_point = vor.vertices[i] + direction * max(ptp_bound) * 2

                infinite_segments.append([vor.vertices[i], far_point])
        return finite_segments, infinite_segments

    # Draw.

    def draw(
        self,
        plot_crosses: bool = True,
        resolution: int = 500,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a KMeans Voronoi plot using the Plotly API.
        """
        fig = self._get_fig(fig)
        colors = self.get_colors()
        buffer = 0.5
        cluster_points = self.data["clusters"]
        if "colors" in style_kwargs:
            colors = (
                style_kwargs["colors"] + colors
                if isinstance(style_kwargs["colors"], list)
                else [style_kwargs["colors"]] + colors
            )
            colors = colors[: len(cluster_points)]
            style_kwargs.pop("colors")
            self.init_heatmap_style["colorscale"] = colors
        all_points = self.data["X"]
        x_range = [
            cluster_points[:, 0].min() - buffer,
            cluster_points[:, 0].max() + buffer,
        ]
        y_range = [
            cluster_points[:, 1].min() - buffer,
            cluster_points[:, 1].max() + buffer,
        ]
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        distance_arrays = []
        for sample_point in cluster_points:
            distance_arrays.append(
                np.linalg.norm(
                    np.vstack([X.ravel(), Y.ravel()]).T - sample_point, axis=1
                ).reshape(X.shape)
            )
        classified_array = np.argmin(np.dstack(distance_arrays), axis=-1)
        fig = fig.add_trace(
            go.Heatmap(
                x=x_vals, y=y_vals, z=classified_array, **self.init_heatmap_style
            )
        )
        if plot_crosses:
            fig.add_trace(
                go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    **self.init_cluster_scatter_style,
                    name="Clusters",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=all_points[:, 0],
                y=all_points[:, 1],
                **self.init_scatter_style,
                name="All points",
            )
        )
        finite_segments, infinite_segments = self._get_voronoi_lines(cluster_points)
        for i in range(len(finite_segments)):
            fig.add_trace(
                go.Scatter(
                    x=finite_segments[i][:, 0],
                    y=finite_segments[i][:, 1],
                    **self.init_line_style,
                )
            )
        for i in range(len(infinite_segments)):
            fig.add_trace(
                go.Scatter(
                    x=[infinite_segments[i][0][0], infinite_segments[i][1][0]],
                    y=[infinite_segments[i][0][1], infinite_segments[i][1][1]],
                    **self.init_line_style,
                )
            )
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        fig.update_layout(
            xaxis=dict(
                range=[
                    cluster_points[:, 0].min() - buffer,
                    cluster_points[:, 0].max() + buffer,
                ]
            ),
            yaxis=dict(
                range=[
                    cluster_points[:, 1].min() - buffer,
                    cluster_points[:, 1].max() + buffer,
                ]
            ),
        )
        return fig
