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

        - lapack:
            Lapack definition.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    principal_components_: numpy.array
        Matrix of the principal components.
    mean_: numpy.array
        List of the averages of each input feature.
    cos2_: numpy.array
        Quality of representation of each observation in
        the principal component space. A high cos2 value
        indicates that the observation is well-represented
        in the reduced-dimensional space defined by the
        principal components, while a low value suggests
        poor representation.
    explained_variance_: numpy.array
        Represents the proportion of the total variance in
        the original dataset that is captured by a specific
        principal component or a combination of principal
        components.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.decomposition.Decomposition.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.decomposition.Decomposition.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, we will use the winequality dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_winequality()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    We can drop the "color" column as it is varchar type.

    .. code-block::

        data.drop("color")

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_winequality()
        data.drop("color")

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``PCA`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import PCA

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import PCA

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = PCA(
            n_components = 3,
        )

    You can select the number of components by the ``n_component``
    parameter. If it is not provided, then all are considered.

    .. hint::

        In :py:mod:`verticapy` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(data)

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database.

    Scores
    ^^^^^^^

    The decomposition  score  on  the  dataset for  each
    transformed column can be calculated by:

    .. ipython:: python

        model.score()

    For more details on the function, check out
    :py:meth:`verticapy.machine_learning.PCA.score`

    You can also fetch the explained variance by:

    .. ipython:: python

        model.explained_variance_

    Principal Components
    ^^^^^^^^^^^^^^^^^^^^^

    To get the transformed dataset in the form of principal
    components:

    .. ipython:: python

        model.transform(data)

    Please refer to :py:meth:`verticapy.machine_learning.PCA.transform`
    for more details on transforming a :py:class:`vDataFrame`.

    Similarly, you can perform the inverse tranform to get
    the original features using:

    .. code-block:: python

        model.inverse_transform(data_transformed)

    The variable ``data_transformed`` includes the PCA components.

    Plots - PCA
    ^^^^^^^^^^^^

    You can plot the first two components conveniently using:

    .. code-block:: python

        model.plot()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_pca_plot.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_pca_plot.html

    Plots - Scree
    ^^^^^^^^^^^^^^

    You can also plot the Scree plot:

    .. code-block:: python

        model.plot_scree()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")
        fig = model.plot_scree()
        html_text = fig.htmlcontent.replace("container", "ml_vertica_PCA_scree")
        with open("SPHINX_DIRECTORY/figures/machine_learning_vertica_pca_plot_scree.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_pca_plot_scree.html

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'n_components': 3})

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    **To Memmodel**

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``MemModel`` objects serve as in-memory representations of machine
        learning models. They can be used for both in-database and in-memory
        prediction tasks. These objects can be pickled in the same way that
        you would pickle a ``scikit-learn`` model.

    The preceding methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **SQL**

    To get the SQL query use below:

    .. ipython:: python

        model.to_sql()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[3.8, 0.3, 0.02, 11, 0.03, 20, 113, 0.99, 3, 0.4, 12, 6, 0]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.decomposition.PCA.to_python`
        method is used to retrieve the Principal Component values.
        For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
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

    .. important::

        This algorithm is not Vertica Native and relies solely
        on SQL for attribute computation. While this model does
        not take advantage of the benefits provided by a model
        management system, including versioning and tracking,
        the SQL code it generates can still be used to create a
        pipeline.

    Parameters
    ----------
    name: str, optional
        Name of the model.  The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    principal_components_: numpy.array
        Matrix of the principal components.
    mean_: numpy.array
        List of the averages of each input feature.
    cos2_: numpy.array
        Quality of representation of each observation in
        the principal component space. A high cos2 value
        indicates that the observation is well-represented
        in the reduced-dimensional space defined by the
        principal components, while a low value suggests
        poor representation.
    explained_variance_: numpy.array
        Represents the proportion of the total variance in
        the original dataset that is captured by a specific
        principal component or a combination of principal
        components.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.decomposition.Decomposition.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.decomposition.Decomposition.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, we will use the Titanic dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_titanic()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_titanic()

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``MCA`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import MCA

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = MCA()

    You can select the number of components by the ``n_component``
    parameter. If it is not provided, then all are considered.

    .. important::

        As this model is not native, it solely relies on SQL statements to
        compute various attributes, storing them within the object. No data
        is saved in the database.

    Model Training
    ^^^^^^^^^^^^^^^

    Before fitting the model, we need to calculate the Transformed Completely
    Disjontive Table before fitting the model:

    .. ipython:: python
        :okwarning:

        tcdt = data[["survived", "pclass", "sex"]].cdt()

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(tcdt)

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database.

    Scores
    ^^^^^^

    The decomposition  score  on  the  dataset for  each
    transformed column can be calculated by:

    .. ipython:: python

        model.score()

    For more details on the function, check out
    :py:meth:`verticapy.machine_learning.MCA.score`

    You can also fetch the explained variance by:

    .. ipython:: python

        model.explained_variance_

    Principal Components
    ^^^^^^^^^^^^^^^^^^^^^^

    To get the transformed dataset in the form of principal
    components:

    .. ipython:: python

        model.transform(tcdt)

    Please refer to :py:meth:`verticapy.machine_learning.MCA.transform`
    for more details on transforming a :py:class:`vDataFrame`.

    Similarly, you can perform the inverse tranform to get
    the original features using:

    .. code-block:: python

        model.inverse_transform(data_transformed)

    The variable ``data_transformed`` includes the MCA components.

    Plots - MCA
    ^^^^^^^^^^^^

    You can plot the first two dimensions conveniently using:

    .. code-block:: python

        model.plot()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_mca_plot.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_mca_plot.html

    Plots - Scree
    ^^^^^^^^^^^^^^

    You can also plot the Scree plot:

    .. code-block:: python

        model.plot_scree()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")
        fig = model.plot_scree()
        html_text = fig.htmlcontent.replace("container", "ml_vertica_MCA_scree")
        with open("SPHINX_DIRECTORY/figures/machine_learning_vertica_mca_plot_scree.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_mca_plot_scree.html

    Plots - Decomposition Circle
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    You can also plot the Decomposition Circles:

    .. code-block:: python

        model.plot_circle()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot_circle()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_mca_plot_circle.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_mca_plot_circle.html

    Model Register
    ^^^^^^^^^^^^^^

    As this model is not native, it does not support model management and
    versioning. However, it is possible to use the SQL code it generates
    for deployment.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    **To Memmodel**

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``MemModel`` objects serve as in-memory representations of machine
        learning models. They can be used for both in-database and in-memory
        prediction tasks. These objects can be pickled in the same way that
        you would pickle a ``scikit-learn`` model.

    The preceding methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **SQL**

    To get the SQL query use below:

    .. ipython:: python

        model.to_sql()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[0, 1, 0, 1, 1, 0, 1]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.decomposition.MCA.to_python`
        method is used to retrieve the Principal Component values.
        For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
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

        - lapack:
            Lapack definition.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    values_: numpy.array
        Matrix of the right singular vectors.
    values_: numpy.array
        Array of the singular values for each input
        feature.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.decomposition.Decomposition.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.decomposition.Decomposition.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, we will use the winequality dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_winequality()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html

    We can drop the "color" column as it is varchar type.

    .. code-block::

        data.drop("color")

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_winequality()
        data.drop("color")

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``SVD`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import SVD

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import SVD

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = SVD(
            n_components = 3,
        )

    You can select the number of components by the ``n_component``
    parameter. If it is not provided, then all are considered.

    .. hint::

        In :py:mod:`verticapy` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(data)

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database.

    Scores
    ^^^^^^

    The decomposition  score  on  the  dataset for  each
    transformed column can be calculated by:

    .. ipython:: python

        model.score()

    For more details on the function, check out
    :py:meth:`verticapy.machine_learning.SVD.score`

    You can also fetch the explained variance by:

    .. ipython:: python

        model.explained_variance_

    Principal Components
    ^^^^^^^^^^^^^^^^^^^^^^

    To get the transformed dataset in the form of principal
    components:

    .. ipython:: python

        model.transform(data)

    Please refer to :py:meth:`verticapy.machine_learning.SVD.transform`
    for more details on transforming a :py:class:`vDataFrame`.

    Similarly, you can perform the inverse tranform to get
    the original features using:

    .. code-block:: python

        model.inverse_transform(data_transformed)

    The variable ``data_transformed`` includes the PCA components.

    Plots - SVD
    ^^^^^^^^^^^^

    You can plot the first two dimensions conveniently using:

    .. code-block:: python

        model.plot()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_svd_plot.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_svd_plot.html

    Plots - Scree
    ^^^^^^^^^^^^^^

    You can also plot the Scree plot:

    .. code-block:: python

        model.plot_scree()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")
        fig = model.plot_scree()
        html_text = fig.htmlcontent.replace("container", "ml_vertica_SVD_scree")
        with open("SPHINX_DIRECTORY/figures/machine_learning_vertica_svd_plot_scree.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_svd_plot_scree.html

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'n_components': 3})

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    **To Memmodel**

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``MemModel`` objects serve as in-memory representations of machine
        learning models. They can be used for both in-database and in-memory
        prediction tasks. These objects can be pickled in the same way that
        you would pickle a ``scikit-learn`` model.

    The preceding methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **SQL**

    To get the SQL query use below:

    .. ipython:: python

        model.to_sql()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[3.8, 0.3, 0.02, 11, 0.03, 20, 113, 0.99, 3, 0.4, 12, 6, 0]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.decomposition.SVD.to_python`
        method is used to retrieve the Principal Component values.
        For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
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
