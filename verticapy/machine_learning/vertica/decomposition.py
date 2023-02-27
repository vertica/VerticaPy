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
import numpy as np

from verticapy._config.colors import get_cmap, get_colors
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.tablesample.base import TableSample

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.vertica.base import Decomposition

from verticapy.plotting._matplotlib.mlplot import plot_var


class PCA(Decomposition):
    """
Creates a PCA (Principal Component Analysis) object using the Vertica PCA
algorithm on the data.
 
Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
n_components: int, optional
	The number of components to keep in the model. If this value is not 
    provided, all components are kept. The maximum number of components 
    is the number of non-zero singular values returned by the internal 
    call to SVD. This number is less than or equal to SVD (number of 
    columns, number of rows). 
scale: bool, optional
	A Boolean value that specifies whether to standardize the columns 
    during the preparation step.
method: str, optional
	The method to use to calculate PCA.
		lapack: Lapack definition.
	"""

    @property
    def _vertica_fit_sql(self) -> Literal["PCA"]:
        return "PCA"

    @property
    def _vertica_transform_sql(self) -> Literal["APPLY_PCA"]:
        return "APPLY_PCA"

    @property
    def _vertica_inverse_transform_sql(self) -> Literal["APPLY_INVERSE_PCA"]:
        return "APPLY_INVERSE_PCA"

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["DECOMPOSITION"]:
        return "DECOMPOSITION"

    @property
    def _model_type(self) -> Literal["PCA"]:
        return "PCA"

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        n_components: int = 0,
        scale: bool = False,
        method: Literal["lapack"] = "lapack",
    ):
        self.model_name = name
        self.parameters = {
            "n_components": n_components,
            "scale": scale,
            "method": str(method).lower(),
        }

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.principal_components_ = self.get_attr("principal_components").to_numpy()
        self.mean_ = np.array(self.get_attr("columns")["mean"])
        cos2 = self.get_attr("principal_components").to_list()
        for i in range(len(cos2)):
            cos2[i] = [v ** 2 for v in cos2[i]]
            total = sum(cos2[i])
            cos2[i] = [v / total for v in cos2[i]]
        values = {"index": self.X}
        for idx, v in enumerate(self.get_attr("principal_components").values):
            if v != "index":
                values[v] = [c[idx - 1] for c in cos2]
        self.cos2_ = TableSample(values)
        return None

    def to_memmodel(self) -> mm.PCA:
        """
        Converts the model to an InMemory object which
        can be used to do different types of predictions.
        """
        return mm.PCA(self.principal_components_, self.mean_)


class MCA(PCA):
    """
Creates a MCA (multiple correspondence analysis) object using the Vertica 
PCA algorithm on the data. It uses the property that the MCA is a PCA 
applied to a complete disjunctive table. The input relation is transformed 
to a TCDT (transformed complete disjunctive table) before applying the PCA.
 
Parameters
----------
name: str
    Name of the the model. The model will be stored in the database.
    """

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _vertica_fit_sql(self) -> Literal["PCA"]:
        return "PCA"

    @property
    def _vertica_transform_sql(self) -> Literal["APPLY_PCA"]:
        return "APPLY_PCA"

    @property
    def _vertica_inverse_transform_sql(self) -> Literal["APPLY_INVERSE_PCA"]:
        return "APPLY_INVERSE_PCA"

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["DECOMPOSITION"]:
        return "DECOMPOSITION"

    @property
    def _model_type(self) -> Literal["MCA"]:
        return "MCA"

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, name: str):
        self.model_name = name
        self.parameters = {}

    def plot_var(
        self,
        dimensions: tuple = (1, 2),
        method: Literal["auto", "cos2", "contrib"] = "auto",
        ax=None,
        **style_kwds,
    ):
        """
    Draws the MCA (multiple correspondence analysis) graph.

    Parameters
    ----------
    dimensions: tuple, optional
        Tuple of two IDs of the model's components.
    method: str, optional
        Method used to draw the plot.
            auto   : Only the variables are displayed.
            cos2   : The cos2 is used as CMAP.
            contrib: The feature contribution is used as CMAP.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        x = self.get_attr("principal_components")[f"PC{dimensions[0]}"]
        y = self.get_attr("principal_components")[f"PC{dimensions[1]}"]
        n = len(self.cos2_[f"PC{dimensions[0]}"])
        if method in ("cos2", "contrib"):
            if method == "cos2":
                c = [
                    self.cos2_[f"PC{dimensions[0]}"][i]
                    + self.cos2_[f"PC{dimensions[1]}"][i]
                    for i in range(n)
                ]
            else:
                sum_1, sum_2 = (
                    sum(self.cos2_[f"PC{dimensions[0]}"]),
                    sum(self.cos2_[f"PC{dimensions[1]}"]),
                )
                c = [
                    0.5
                    * 100
                    * (
                        self.cos2_[f"PC{dimensions[0]}"][i] / sum_1
                        + self.cos2_[f"PC{dimensions[1]}"][i] / sum_2
                    )
                    for i in range(n)
                ]
            style_kwds["c"] = c
            if "cmap" not in style_kwds:
                style_kwds["cmap"] = get_cmap(
                    color=[get_colors()[0], get_colors()[1], get_colors()[2],]
                )
        explained_variance = self.get_attr("singular_values")["explained_variance"]
        return plot_var(
            x,
            y,
            self.X,
            (
                explained_variance[dimensions[0] - 1],
                explained_variance[dimensions[1] - 1],
            ),
            dimensions,
            method,
            ax,
            **style_kwds,
        )

    def plot_contrib(self, dimension: int = 1, ax=None, **style_kwds):
        """
    Draws a decomposition contribution plot of the input dimension.

    Parameters
    ----------
    dimension: int, optional
        Integer representing the IDs of the model's component.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        contrib = self.get_attr("principal_components")[f"PC{dimension}"]
        contrib = [elem ** 2 for elem in contrib]
        total = sum(contrib)
        contrib = [100 * elem / total for elem in contrib]
        n = len(contrib)
        variables, contribution = zip(
            *sorted(zip(self.X, contrib), key=lambda t: t[1], reverse=True)
        )
        contrib = TableSample(
            {"row_nb": [i + 1 for i in range(n)], "contrib": contribution}
        ).to_vdf()
        contrib["row_nb_2"] = contrib["row_nb"] + 0.5
        ax = contrib["row_nb"].hist(
            method="avg", of="contrib", max_cardinality=1, h=1, ax=ax, **style_kwds
        )
        ax = contrib["contrib"].plot(ts="row_nb_2", ax=ax, color="black")
        ax.set_xlim(1, n + 1)
        ax.set_xticks([i + 1.5 for i in range(n)])
        ax.set_xticklabels(variables)
        ax.set_ylabel("Cos2 - Quality of Representation")
        ax.set_xlabel("")
        ax.set_title(f"Contribution of variables to Dim {dimension}")
        ax.plot([1, n + 1], [1 / n * 100, 1 / n * 100], c="r", linestyle="--")
        for i in range(n):
            ax.text(
                i + 1.5, contribution[i] + 1, f"{round(contribution[i], 1)}%",
            )
        return ax

    def plot_cos2(self, dimensions: tuple = (1, 2), ax=None, **style_kwds):
        """
    Draws a MCA (multiple correspondence analysis) cos2 plot of 
    the two input dimensions.

    Parameters
    ----------
    dimensions: tuple, optional
        Tuple of two IDs of the model's components.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        cos2_1 = self.cos2_[f"PC{dimensions[0]}"]
        cos2_2 = self.cos2_[f"PC{dimensions[1]}"]
        n = len(cos2_1)
        quality = []
        for i in range(n):
            quality += [cos2_1[i] + cos2_2[i]]
        variables, quality = zip(
            *sorted(zip(self.X, quality), key=lambda t: t[1], reverse=True)
        )
        quality = TableSample({"variables": variables, "quality": quality}).to_vdf()
        ax = quality["variables"].hist(
            method="avg", of="quality", max_cardinality=n, ax=ax, **style_kwds
        )
        ax.set_ylabel("Cos2 - Quality of Representation")
        ax.set_xlabel("")
        ax.set_title(f"Cos2 of variables to Dim {dimensions[0]}-{dimensions[1]}")
        return ax


class SVD(Decomposition):
    """
Creates an SVD (Singular Value Decomposition) object using the Vertica SVD
algorithm on the data.
 
Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
n_components: int, optional
	The number of components to keep in the model. If this value is not 
    provided, all components are kept. The maximum number of components 
    is the number of non-zero singular values returned by the internal 
    call to SVD. This number is less than or equal to SVD (number of 
    columns, number of rows).
method: str, optional
	The method to use to calculate SVD.
		lapack: Lapack definition.
	"""

    @property
    def _vertica_fit_sql(self) -> Literal["SVD"]:
        return "SVD"

    @property
    def _vertica_transform_sql(self) -> Literal["APPLY_SVD"]:
        return "APPLY_SVD"

    @property
    def _vertica_inverse_transform_sql(self) -> Literal["APPLY_INVERSE_SVD"]:
        return "APPLY_INVERSE_SVD"

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["DECOMPOSITION"]:
        return "DECOMPOSITION"

    @property
    def _model_type(self) -> Literal["SVD"]:
        return "SVD"

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self, name: str, n_components: int = 0, method: Literal["lapack"] = "lapack"
    ):
        self.model_name = name
        self.parameters = {
            "n_components": n_components,
            "method": str(method).lower(),
        }

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.vectors_ = self.get_attr("right_singular_vectors").to_numpy()
        self.values_ = np.array(self.get_attr("singular_values")["value"])
        return None

    def to_memmodel(self) -> mm.SVD:
        """
        Converts the model to an InMemory object which
        can be used to do different types of predictions.
        """
        return mm.SVD(self.vectors_, self.values_)
