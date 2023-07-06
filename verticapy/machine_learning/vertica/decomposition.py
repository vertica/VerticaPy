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

from verticapy._typing import (
    NoneType,
    PlottingObject,
    PythonNumber,
    SQLColumns,
    SQLRelation,
)
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, format_type, quote_ident
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.vertica.preprocessing import Preprocessing

"""
General Classes.
"""


class Decomposition(Preprocessing):
    # I/O Methods.

    def deploySQL(
        self,
        X: Optional[SQLColumns] = None,
        n_components: int = 0,
        cutoff: PythonNumber = 1,
        key_columns: Optional[SQLColumns] = None,
        exclude_columns: Optional[SQLColumns] = None,
    ) -> str:
        """
        Returns the SQL code needed to deploy the model.

        Parameters
        ----------
        X: SQLColumns, optional
            List of the columns used to deploy the model.
            If empty,  the model predictors are used.
        n_components: int, optional
            Number  of  components to return.  If  set to
            0,  all  the  components  are deployed.
        cutoff: PythonNumber, optional
            Specifies  the minimum accumulated  explained
            variance.  Components  are  taken  until  the
            accumulated  explained  variance reaches this
            value.
        key_columns: SQLColumns, optional
            Predictors   used    during   the   algorithm
            computation  that will be deployed with  the
            principal components.
        exclude_columns: SQLColumns, optional
            Columns to exclude from the prediction.

        Returns
        -------
        str
            the SQL code needed to deploy the model.
        """
        exclude_columns, key_columns = format_type(
            exclude_columns, key_columns, dtype=list
        )
        X = format_type(X, dtype=list, na_out=self.X)
        X = quote_ident(X)
        sql = f"""{self._vertica_transform_sql}({', '.join(X)} 
                                            USING PARAMETERS
                                            model_name = '{self.model_name}',
                                            match_by_pos = 'true'"""
        if key_columns:
            key_columns = ", ".join(quote_ident(key_columns))
            sql += f", key_columns = '{key_columns}'"
        if exclude_columns:
            exclude_columns = ", ".join(quote_ident(exclude_columns))
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
        X: Optional[SQLColumns] = None,
        input_relation: Optional[str] = None,
        metric: Literal["avg", "median"] = "avg",
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
            If empty, the model  predictors are used.
        input_relation: str, optional
            Input  Relation.  If  empty,  the model input
            relation are used.
        metric: str, optional
            Distance metric used to do the scoring.
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
        if isinstance(X, NoneType):
            X = self.X
        X = format_type(X, dtype=list)
        if not input_relation:
            input_relation = self.input_relation
        metric = str(metric).upper()
        if metric == "MEDIAN":
            metric = "APPROXIMATE_MEDIAN"
        if self._model_type in ("PCA", "SVD"):
            n_components = self.parameters["n_components"]
            if not n_components:
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
            f"""{metric}(POWER(ABS(POWER({X[idx]}, {p}) 
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
        X: Optional[SQLColumns] = None,
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
            enclose  it  with  an  alias.  For example:
            "(SELECT 1) x"    is     valid    whereas
            "(SELECT 1)" and "SELECT 1" are invalid.
        X: SQLColumns, optional
            List of the input vDataColumns.
        n_components: int, optional
            Number  of components to return.  If set to
            0, all the components are deployed.
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
        if isinstance(X, NoneType):
            X = self.X
        X = format_type(X, dtype=list)
        if not vdf:
            vdf = self.input_relation
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = vdf.format_colnames(X)
        exclude_columns = vdf.get_columns(exclude_columns=X)
        all_columns = vdf.get_columns()
        columns = self.deploySQL(
            all_columns,
            n_components,
            cutoff,
            exclude_columns,
            exclude_columns,
        )
        main_relation = f"(SELECT {columns} FROM {vdf}) VERTICAPY_SUBTABLE"
        return vDataFrame(main_relation)

    # Plotting Methods.

    def plot(
        self,
        dimensions: tuple = (1, 2),
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws a decomposition scatter plot.

        Parameters
        ----------
        dimensions: tuple, optional
            Tuple of two elements representing the
            IDs of the model's components.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional  parameter to pass to the
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vdf = self.transform(vDataFrame(self.input_relation))
        dim_perc = []
        for d in dimensions:
            if not self.explained_variance_[d - 1]:
                dim_perc += [""]
            else:
                dim_perc += [f" ({round(self.explained_variance_[d - 1] * 100, 1)}%)"]
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="ScatterPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.ScatterPlot(
            vdf=vdf,
            columns=[f"col{dimensions[0]}", f"col{dimensions[1]}"],
            max_nb_points=100000,
            misc_layout={
                "columns": [
                    f"Dim{dimensions[0]}{dim_perc[0]}",
                    f"Dim{dimensions[1]}{dim_perc[1]}",
                ]
            },
        ).draw(**kwargs)

    def plot_circle(
        self,
        dimensions: tuple = (1, 2),
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws a decomposition circle.

        Parameters
        ----------
        dimensions: tuple, optional
            Tuple of two elements representing the IDs
            of the model's components.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to  the
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        if self._model_type == "SVD":
            x = self.vectors_[:, dimensions[0] - 1]
            y = self.vectors_[:, dimensions[1] - 1]
        else:
            x = self.principal_components_[:, dimensions[0] - 1]
            y = self.principal_components_[:, dimensions[1] - 1]
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="PCACirclePlot",
            chart=chart,
            style_kwargs=style_kwargs,
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
        return vpy_plt.PCACirclePlot(data=data, layout=layout).draw(**kwargs)

    def plot_scree(
        self, chart: Optional[PlottingObject] = None, **style_kwargs
    ) -> PlottingObject:
        """
        Draws a decomposition scree plot.

        Parameters
        ----------
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="PCAScreePlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        n = len(self.explained_variance_)
        data = {
            "x": np.array([i + 1 for i in range(n)]),
            "y": 100 * self.explained_variance_,
            "adj_width": 0.94,
        }
        layout = {
            "labels": [i + 1 for i in range(n)],
            "x_label": "dimensions",
            "y_label": "percentage_explained_variance (%)",
            "title": None,
            "plot_scree": True,
            "plot_line": False,
        }
        return vpy_plt.PCAScreePlot(data=data, layout=layout).draw(**kwargs)


"""
Algorithms used for decomposition.
"""


class PCA(Decomposition):
    """
    Creates a PCA  (Principal Component Analysis) object
    using the Vertica PCA algorithm.

    Parameters
    ----------
    name: str, optional
        Name  of the  model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
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
        The method used to calculate PCA.
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
        name: str = None,
        overwrite_model: bool = False,
        n_components: int = 0,
        scale: bool = False,
        method: Literal["lapack"] = "lapack",
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_components": n_components,
            "scale": scale,
            "method": str(method).lower(),
        }

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
            cos2[i] = [v**2 for v in cos2[i]]
            total = sum(cos2[i])
            cos2[i] = [v / total for v in cos2[i]]
        values = {"index": self.X}
        for idx, v in enumerate(
            self.get_vertica_attributes("principal_components").values
        ):
            if v != "index":
                values[v] = [c[idx - 1] for c in cos2]
        self.cos2_ = TableSample(values).to_numpy()

    # I/O Methods.

    def to_memmodel(self) -> mm.PCA:
        """
        Converts  the model  to an InMemory object  that
        can be used for different types of predictions.
        """
        return mm.PCA(self.principal_components_, self.mean_)


class MCA(PCA):
    """
    Creates a MCA  (multiple correspondence analysis) object
    using  the Vertica PCA  algorithm. MCA is a PCA applied
    to a complete disjunctive table.  The  input relation is
    transformed to a TCDT (transformed  complete  disjunctive
    table) before applying the PCA.

    Parameters
    ----------
    name: str, optional
        Name of the model.  The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
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
    def _model_subcategory(self) -> Literal["DECOMPOSITION"]:
        return "DECOMPOSITION"

    @property
    def _model_type(self) -> Literal["MCA"]:
        return "MCA"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {}

    # Plotting Methods.

    def plot_contrib(
        self, dimension: int = 1, chart: Optional[PlottingObject] = None, **style_kwargs
    ) -> PlottingObject:
        """
        Draws a decomposition  contribution plot of the input
        dimension.

        Parameters
        ----------
        dimension: int, optional
            Integer  representing  the  IDs  of the  model's
            component.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the Plotting
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        contrib = self.principal_components_[:, dimension - 1] ** 2
        contrib = 100 * contrib / contrib.sum()
        variables, contribution = zip(
            *sorted(zip(self.X, contrib), key=lambda t: t[1], reverse=True)
        )
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="PCAScreePlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        n = len(contribution)
        data = {
            "x": np.array([i + 1 for i in range(n)]),
            "y": contribution,
            "adj_width": 0.94,
        }
        layout = {
            "labels": variables,
            "x_label": None,
            "y_label": "Contribution (%)",
            "title": f"Contribution of variables to Dim {dimension}",
            "plot_scree": True,
            "plot_line": True,
        }
        return vpy_plt.PCAScreePlot(data=data, layout=layout).draw(**kwargs)

    def plot_cos2(
        self,
        dimensions: tuple = (1, 2),
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws a MCA (multiple correspondence analysis) cos2
        plot of the two input dimensions.

        Parameters
        ----------
        dimensions: tuple, optional
            Tuple of two IDs of the model's components.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the Plotting
            functions.

        Returns
        -------
        obj
            Plotting Object.
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
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="PCAScreePlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        n = len(self.explained_variance_)
        data = {
            "x": np.array([i + 1 for i in range(n)]),
            "y": 100 * np.array(quality),
            "adj_width": 1.0,
        }
        layout = {
            "labels": variables,
            "x_label": None,
            "y_label": "Cos2 - Quality of Representation (%)",
            "title": f"Cos2 of variables to Dim {dimensions[0]}-{dimensions[1]}",
            "plot_scree": False,
            "plot_line": False,
        }
        return vpy_plt.PCAScreePlot(data=data, layout=layout).draw(**kwargs)

    def plot_var(
        self,
        dimensions: tuple = (1, 2),
        method: Literal["auto", "cos2", "contrib"] = "auto",
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
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
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the Plotting
            functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        x = self.principal_components_[:, dimensions[0] - 1]
        y = self.principal_components_[:, dimensions[1] - 1]
        n = len(self.cos2_[:, dimensions[0] - 1])
        c = None
        has_category = False
        if method in ("cos2", "contrib"):
            has_category = True
            if method == "cos2":
                c = np.array(
                    [
                        self.cos2_[:, dimensions[0] - 1][i]
                        + self.cos2_[:, dimensions[1] - 1][i]
                        for i in range(n)
                    ]
                )
            else:
                sum_1, sum_2 = (
                    sum(self.cos2_[:, dimensions[0] - 1]),
                    sum(self.cos2_[:, dimensions[1] - 1]),
                )
                c = np.array(
                    [
                        0.5
                        * 100
                        * (
                            self.cos2_[:, dimensions[0] - 1][i] / sum_1
                            + self.cos2_[:, dimensions[1] - 1][i] / sum_2
                        )
                        for i in range(n)
                    ]
                )
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="PCAVarPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        data = {
            "x": x,
            "y": y,
            "c": c,
            "explained_variance": [
                self.explained_variance_[dimensions[0] - 1],
                self.explained_variance_[dimensions[1] - 1],
            ],
            "dim": dimensions,
        }
        layout = {
            "columns": self.X,
            "method": method,
            "has_category": has_category,
        }
        return vpy_plt.PCAVarPlot(data=data, layout=layout).draw(**kwargs)


class SVD(Decomposition):
    """
    Creates  an  SVD  (Singular  Value  Decomposition)
    object using the Vertica SVD algorithm.

    Parameters
    ----------
    name: str, optional
        Name  of the model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    n_components: int, optional
        The number  of components to keep in the model.
        If this value  is not provided,  all components
        are kept.  The maximum number of  components is
        the number of non-zero singular values returned
        by  the  internal call to SVD. This  number  is
        less  than or equal to SVD (number of  columns,
        number of rows).
    method: str, optional
        The method used to calculate SVD.
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
        self,
        name: str = None,
        overwrite_model: bool = False,
        n_components: int = 0,
        method: Literal["lapack"] = "lapack",
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_components": n_components,
            "method": str(method).lower(),
        }

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

    # I/O Methods.

    def to_memmodel(self) -> mm.SVD:
        """
        Converts  the model  to an InMemory object  that
        can be used for different types of predictions.
        """
        return mm.SVD(self.vectors_, self.values_)
