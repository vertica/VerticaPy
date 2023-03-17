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
from typing import Literal, Optional
import numpy as np

from matplotlib.axes import Axes

from verticapy._config.colors import get_colors
from verticapy._typing import PythonNumber, SQLColumns, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, quote_ident
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.vertica.preprocessing import Preprocessing

from verticapy.plotting.base import PlottingBase
import verticapy.plotting._matplotlib as vpy_matplotlib_plt

"""
General Classes.
"""


class Decomposition(Preprocessing):

    # I/O Methods.

    def deploySQL(
        self,
        X: SQLColumns = [],
        n_components: int = 0,
        cutoff: PythonNumber = 1,
        key_columns: SQLColumns = [],
        exclude_columns: SQLColumns = [],
    ) -> str:
        """
        Returns the SQL code needed to deploy the model. 

        Parameters
        ----------
        X: SQLColumns, optional
            List of the columns used to deploy the model. 
            If empty,  the model predictors will be used.
        n_components: int, optional
            Number  of  components to return.  If  set to 
            0,  all  the  components  will  be  deployed.
        cutoff: PythonNumber, optional
            Specifies  the minimum accumulated  explained 
            variance.  Components  are  taken  until  the 
            accumulated  explained  variance reaches this 
            value.
        key_columns: SQLColumns, optional
            Predictors   used    during   the   algorithm 
            computation  which will be deployed with  the 
            principal components.
        exclude_columns: SQLColumns, optional
            Columns to exclude from the prediction.

        Returns
        -------
        str
            the SQL code needed to deploy the model.
        """
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        if isinstance(exclude_columns, str):
            exclude_columns = [exclude_columns]
        if isinstance(X, str):
            X = [X]
        if not (X):
            X = self.X
        else:
            X = [quote_ident(elem) for elem in X]
        fun = self._vertica_transform_sql
        sql = f"""{self._vertica_transform_sql}({', '.join(X)} 
                                            USING PARAMETERS
                                            model_name = '{self.model_name}',
                                            match_by_pos = 'true'"""
        if key_columns:
            key_columns = ", ".join([quote_ident(col) for col in key_columns])
            sql += f", key_columns = '{key_columns}'"
        if exclude_columns:
            exclude_columns = ", ".join([quote_ident(col) for col in exclude_columns])
            sql += f", exclude_columns = '{exclude_columns}'"
        if n_components:
            sql += f", num_components = {n_components}"
        else:
            sql += f", cutoff = {cutoff}"
        sql += ")"
        return clean_query(sql)

    # Model Evaluation Methods.

    def score(
        self,
        X: SQLColumns = [],
        input_relation: str = "",
        method: Literal["avg", "median"] = "avg",
        p: int = 2,
    ) -> TableSample:
        """
        Returns  the  decomposition  score  on  a  dataset 
        for  each  transformed column.  It is the  average 
        / median of the p-distance between the real column 
        and  its  result after applying the  decomposition 
        model and its inverse.  

        Parameters
        ----------
        X: SQLColumns, optional
            List of the columns used to deploy the model. 
            If empty, the model  predictors will be used.
        input_relation: str, optional
            Input  Relation.  If  empty,  the model input 
            relation will be used.
        method: str, optional
            Distance Method used to do the scoring.
                avg    : The average is used as 
                         aggregation.
                median : The median  is used as 
                         aggregation.
        p: int, optional
            The p of the p-distance.

        Returns
        -------
        TableSample
            PCA scores.
        """
        if isinstance(X, str):
            X = [X]
        if not (X):
            X = self.X
        if not (input_relation):
            input_relation = self.input_relation
        method = str(method).upper()
        if method == "MEDIAN":
            method = "APPROXIMATE_MEDIAN"
        if self._model_type in ("PCA", "SVD"):
            n_components = self.parameters["n_components"]
            if not (n_components):
                n_components = len(X)
        else:
            n_components = len(X)
        col_init_1 = [f"{X[idx]} AS col_init{idx}" for idx in range(len(X))]
        col_init_2 = [f"col_init{idx}" for idx in range(len(X))]
        cols = [f"col{idx + 1}" for idx in range(n_components)]
        query = f"""SELECT 
                        {self._vertica_transform_sql}
                        ({', '.join(self.X)} 
                            USING PARAMETERS 
                            model_name = '{self.model_name}', 
                            key_columns = '{', '.join(self.X)}', 
                            num_components = {n_components}) OVER () 
                    FROM {input_relation}"""
        query = f"""
            SELECT 
                {', '.join(col_init_1 + cols)} 
            FROM ({query}) VERTICAPY_SUBTABLE"""
        query = f"""
            SELECT 
                {self._vertica_inverse_transform_sql}
                ({', '.join(col_init_2 + cols)} 
                    USING PARAMETERS 
                    model_name = '{self.model_name}', 
                    key_columns = '{', '.join(col_init_2)}', 
                    exclude_columns = '{', '.join(col_init_2)}', 
                    num_components = {n_components}) OVER () 
            FROM ({query}) y"""
        p_distances = [
            f"""{method}(POWER(ABS(POWER({X[idx]}, {p}) 
                         - POWER(col_init{idx}, {p})), {1 / p})) 
                         AS {X[idx]}"""
            for idx in range(len(X))
        ]
        query = f"""
            SELECT 
                'Score' AS 'index', 
                {', '.join(p_distances)} 
            FROM ({query}) z"""
        return TableSample.read_sql(query, title="Getting Model Score.").transpose()

    # Prediction / Transformation Methods.

    def transform(
        self,
        vdf: SQLRelation = None,
        X: SQLColumns = [],
        n_components: int = 0,
        cutoff: PythonNumber = 1,
    ) -> vDataFrame:
        """
        Applies the model on a vDataFrame.

        Parameters
        ----------
        vdf: SQLRelation, optional
            Input  vDataFrame.   You can  also  specify 
            a  customized   relation,   but   you  must 
            enclose  it  with  an  alias.  For  example 
            "(SELECT 1) x"    is     correct    whereas 
            "(SELECT 1)" and "SELECT 1" are incorrect.
        X: SQLColumns, optional
            List of the input vDataColumns.
        n_components: int, optional
            Number  of components to return.  If set to 
            0, all the components will be deployed.
        cutoff: PythonNumber, optional
            Specifies the minimum accumulated explained 
            variance.  Components  are taken until  the 
            accumulated explained variance reaches this 
            value.

        Returns
        -------
        vDataFrame
            object result of the model transformation.
        """
        if isinstance(X, str):
            X = [X]
        if not (vdf):
            vdf = self.input_relation
        if not (X):
            X = self.X
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = vdf._format_colnames(X)
        relation = vdf._genSQL()
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        columns = self.deploySQL(
            all_columns, n_components, cutoff, exclude_columns, exclude_columns,
        )
        main_relation = f"(SELECT {columns} FROM {relation}) VERTICAPY_SUBTABLE"
        return vDataFrame(main_relation)

    # Plotting Methods.

    def plot(
        self, dimensions: tuple = (1, 2), ax: Optional[Axes] = None, **style_kwargs
    ) -> Axes:
        """
        Draws a decomposition scatter plot.

        Parameters
        ----------
        dimensions: tuple, optional
            Tuple of two elements representing the 
            IDs of the model's components.
        ax: Axes, optional
            The axes to plot on.
        **style_kwargs
            Any optional  parameter to pass to the 
            Matplotlib functions.

        Returns
        -------
        Axes
            Axes.
        """
        vdf = vDataFrame(self.input_relation)
        ax = self.transform(vdf).scatter(
            columns=[f"col{dimensions[0]}", f"col{dimensions[1]}"],
            max_nb_points=100000,
            ax=ax,
            **style_kwargs,
        )
        if not (self.explained_variance_[dimensions[0] - 1]):
            dimensions_1 = ""
        else:
            dimensions_1 = (
                f"({round(self.explained_variance_[dimensions[0] - 1] * 100, 1)}%)"
            )
        ax.set_xlabel(f"Dim{dimensions[0]} {dimensions_1}")
        ax.set_ylabel(f"Dim{dimensions[0]} {dimensions_1}")
        return ax

    def plot_circle(
        self, dimensions: tuple = (1, 2), ax: Optional[Axes] = None, **style_kwargs
    ) -> Axes:
        """
        Draws a decomposition circle.

        Parameters
        ----------
        dimensions: tuple, optional
            Tuple of two elements representing the IDs 
            of the model's components.
        ax: Axes, optional
            The axes to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to  the 
            Matplotlib functions.

        Returns
        -------
        Axes
            Axes.
        """
        if self._model_type == "SVD":
            x = self.vectors_[:, dimensions[0] - 1]
            y = self.vectors_[:, dimensions[1] - 1]
        else:
            x = self.principal_components_[:, dimensions[0] - 1]
            y = self.principal_components_[:, dimensions[1] - 1]
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax,}, style_kwargs=style_kwargs,
        )
        data = {
            "x": x,
            "y": y,
            "explained_variance": [
                self.explained_variance_[dimensions[0] - 1],
                self.explained_variance_[dimensions[1] - 1],
            ],
            "dim": dimensions,
        }
        layout = {
            "columns": self.X,
        }
        return vpy_matplotlib_plt.PCACirclePlot(data=data, layout=layout).draw(**kwargs)

    def plot_scree(self, ax: Optional[Axes] = None, **style_kwargs) -> Axes:
        """
        Draws a decomposition scree plot.

        Parameters
        ----------
        ax: Axes, optional
            The axes to plot on.
        **style_kwargs
            Any optional parameter to pass to the 
            Matplotlib functions.

        Returns
        -------
        Axes
            Axes.
        """
        n = len(self.explained_variance_)
        explained_variance = [100 * x for x in self.explained_variance_]
        information = TableSample(
            {
                "dimensions": [i + 1 for i in range(n)],
                "percentage_explained_variance": explained_variance,
            }
        ).to_vdf()
        information["dimensions_center"] = information["dimensions"] + 0.5
        ax = information["dimensions"].bar(
            method="avg",
            of="percentage_explained_variance",
            h=1,
            max_cardinality=1,
            ax=ax,
            **style_kwargs,
        )
        ax = information["percentage_explained_variance"].plot(
            ts="dimensions_center", ax=ax, color="black"
        )
        ax.set_xlim(1, n + 1)
        ax.set_xticks([i + 1.5 for i in range(n)])
        ax.set_xticklabels([i + 1 for i in range(n)])
        ax.set_ylabel('"percentage_explained_variance"')
        ax.set_xlabel('"dimensions"')
        for i in range(n):
            text_str = f"{round(explained_variance[i], 1)}%"
            ax.text(
                i + 1.5, explained_variance[i] + 1, text_str,
            )
        return ax


"""
Algorithms used for decomposition.
"""


class PCA(Decomposition):
    """
    Creates a PCA  (Principal Component Analysis) object 
    using the Vertica PCA algorithm on the data.
     
    Parameters
    ----------
    name: str
    	Name  of the  model.  The  model will be  stored 
        in the DB.
    n_components: int, optional
    	The  number of  components to keep in the model. 
        If  this value  is not provided,  all components 
        are kept.  The  maximum number of components  is 
        the number of  non-zero singular values returned 
        by the internal call to SVD. This number is less 
        than or equal to SVD  (number of columns, number 
        of rows). 
    scale: bool, optional
    	A  Boolean  value  that  specifies   whether  to 
        standardize  the columns during the  preparation 
        step.
    method: str, optional
    	The method to use to calculate PCA.
    		lapack: Lapack definition.
	"""

    # Properties.

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

    @property
    def _attributes(self) -> list[str]:
        return ["principal_components_", "mean_", "cos2_", "explained_variance_"]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        n_components: int = 0,
        scale: bool = False,
        method: Literal["lapack"] = "lapack",
    ) -> None:
        self.model_name = name
        self.parameters = {
            "n_components": n_components,
            "scale": scale,
            "method": str(method).lower(),
        }
        return None

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.principal_components_ = self.get_vertica_attributes(
            "principal_components"
        ).to_numpy()
        self.mean_ = np.array(self.get_vertica_attributes("columns")["mean"])
        self.explained_variance_ = np.array(
            self.get_vertica_attributes("singular_values")["explained_variance"]
        )
        cos2 = self.get_vertica_attributes("principal_components").to_list()
        for i in range(len(cos2)):
            cos2[i] = [v ** 2 for v in cos2[i]]
            total = sum(cos2[i])
            cos2[i] = [v / total for v in cos2[i]]
        values = {"index": self.X}
        for idx, v in enumerate(
            self.get_vertica_attributes("principal_components").values
        ):
            if v != "index":
                values[v] = [c[idx - 1] for c in cos2]
        self.cos2_ = TableSample(values).to_numpy()
        return None

    # I/O Methods.

    def to_memmodel(self) -> mm.PCA:
        """
        Converts  the model  to an InMemory object  which
        can be used to do different types of predictions.
        """
        return mm.PCA(self.principal_components_, self.mean_)


class MCA(PCA):
    """
    Creates a MCA  (multiple correspondence analysis) object 
    using  the Vertica PCA  algorithm  on the data. It  uses 
    the property that the MCA is a PCA applied to a complete 
    disjunctive  table.  The  input relation is  transformed 
    to  a  TCDT  (transformed  complete  disjunctive  table) 
    before applying the PCA.
     
    Parameters
    ----------
    name: str
        Name of the model.  The model will be stored in the 
        database.
    """

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _is_using_native(self) -> Literal[True]:
        return True

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

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, name: str) -> None:
        self.model_name = name
        self.parameters = {}
        return None

    # Plotting Methods.

    def plot_contrib(
        self, dimension: int = 1, ax: Optional[Axes] = None, **style_kwargs
    ) -> Axes:
        """
        Draws a decomposition  contribution plot of the input 
        dimension.

        Parameters
        ----------
        dimension: int, optional
            Integer  representing  the  IDs  of the  model's 
            component.
        ax: Axes, optional
            The axes to plot on.
        **style_kwargs
            Any optional parameter to pass to the Matplotlib 
            functions.

        Returns
        -------
        Axes
            Axes.
        """
        contrib = self.principal_components_[:, dimension - 1]
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
        ax = contrib["row_nb"].bar(
            method="avg", of="contrib", max_cardinality=1, h=1, ax=ax, **style_kwargs
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

    def plot_cos2(
        self, dimensions: tuple = (1, 2), ax: Optional[Axes] = None, **style_kwargs
    ) -> Axes:
        """
        Draws a MCA (multiple correspondence analysis) cos2 
        plot of the two input dimensions.

        Parameters
        ----------
        dimensions: tuple, optional
            Tuple of two IDs of the model's components.
        ax: Axes, optional
            The axes to plot on.
        **style_kwargs
            Any optional parameter to pass to the Matplotlib 
            functions.

        Returns
        -------
        Axes
            Axes.
        """
        cos2_1 = self.cos2_[:, dimensions[0] - 1]
        cos2_2 = self.cos2_[:, dimensions[1] - 1]
        n = len(cos2_1)
        quality = []
        for i in range(n):
            quality += [cos2_1[i] + cos2_2[i]]
        variables, quality = zip(
            *sorted(zip(self.X, quality), key=lambda t: t[1], reverse=True)
        )
        quality = TableSample({"variables": variables, "quality": quality}).to_vdf()
        ax = quality["variables"].bar(
            method="avg", of="quality", max_cardinality=n, ax=ax, **style_kwargs
        )
        ax.set_ylabel("Cos2 - Quality of Representation")
        ax.set_xlabel("")
        ax.set_title(f"Cos2 of variables to Dim {dimensions[0]}-{dimensions[1]}")
        return ax

    def plot_var(
        self,
        dimensions: tuple = (1, 2),
        method: Literal["auto", "cos2", "contrib"] = "auto",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws  the  MCA  (multiple correspondence analysis) 
        graph.

        Parameters
        ----------
        dimensions: tuple, optional
            Tuple  of  two IDs  of  the model's  components.
        method: str, optional
            Method used to draw the plot.
                auto    : Only the  variables are displayed.
                cos2    : The cos2 is used as CMAP.
                contrib : The feature  contribution is  used 
                          as CMAP.
        ax: Axes, optional
            The axes to plot on.
        **style_kwargs
            Any optional parameter to pass to the Matplotlib 
            functions.

        Returns
        -------
        Axes
            Axes.
        """
        x = self.principal_components_[:, dimensions[0] - 1]
        y = self.principal_components_[:, dimensions[1] - 1]
        n = len(self.cos2_[:, dimensions[0] - 1])
        if method in ("cos2", "contrib"):
            if method == "cos2":
                c = [
                    self.cos2_[:, dimensions[0] - 1][i]
                    + self.cos2_[:, dimensions[1] - 1][i]
                    for i in range(n)
                ]
            else:
                sum_1, sum_2 = (
                    sum(self.cos2_[:, dimensions[0] - 1]),
                    sum(self.cos2_[:, dimensions[1] - 1]),
                )
                c = [
                    0.5
                    * 100
                    * (
                        self.cos2_[:, dimensions[0] - 1][i] / sum_1
                        + self.cos2_[:, dimensions[1] - 1][i] / sum_2
                    )
                    for i in range(n)
                ]
            style_kwargs["c"] = c
            if "cmap" not in style_kwargs:
                style_kwargs["cmap"] = PlottingBase().get_cmap(
                    color=[get_colors(idx=0), get_colors(idx=1), get_colors(idx=2),]
                )
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax,}, style_kwargs=style_kwargs,
        )
        data = {
            "x": x,
            "y": y,
            "explained_variance": [
                self.explained_variance_[dimensions[0] - 1],
                self.explained_variance_[dimensions[1] - 1],
            ],
            "dim": dimensions,
        }
        layout = {
            "columns": self.X,
            "method": method,
        }
        return vpy_matplotlib_plt.PCAVarPlot(data=data, layout=layout).draw(**kwargs)


class SVD(Decomposition):
    """
    Creates  an  SVD  (Singular  Value  Decomposition) 
    object using the Vertica SVD algorithm on the data.
     
    Parameters
    ----------
    name: str
    	Name  of  the model.  The model will be stored 
        in the DB.
    n_components: int, optional
    	The number  of components to keep in the model. 
        If this value  is not provided,  all components 
        are kept.  The maximum number of  components is 
        the number of non-zero singular values returned 
        by  the  internal call to SVD. This  number  is 
        less  than or equal to SVD (number of  columns, 
        number of rows).
    method: str, optional
    	The method to use to calculate SVD.
    		lapack: Lapack definition.
	"""

    # Properties.

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

    @property
    def _attributes(self) -> list[str]:
        return ["vectors_", "values_", "explained_variance_"]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self, name: str, n_components: int = 0, method: Literal["lapack"] = "lapack"
    ) -> None:
        self.model_name = name
        self.parameters = {
            "n_components": n_components,
            "method": str(method).lower(),
        }
        return None

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.vectors_ = self.get_vertica_attributes("right_singular_vectors").to_numpy()
        self.values_ = np.array(self.get_vertica_attributes("singular_values")["value"])
        self.explained_variance_ = np.array(
            self.get_vertica_attributes("singular_values")["explained_variance"]
        )
        return None

    # I/O Methods.

    def to_memmodel(self) -> mm.SVD:
        """
        Converts  the model  to an InMemory object  which
        can be used to do different types of predictions.
        """
        return mm.SVD(self.vectors_, self.values_)
