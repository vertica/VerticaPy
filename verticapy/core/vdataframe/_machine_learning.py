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
import math
import random
import warnings
from itertools import combinations_with_replacement
from typing import Literal, Optional, Union, TYPE_CHECKING

import scipy.stats as scipy_st

import verticapy._config.config as conf
from verticapy._typing import PythonScalar, SQLColumns
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type, quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._random import _seeded_random_function

from verticapy.core.tablesample.base import TableSample

from verticapy.core.vdataframe._scaler import vDFScaler

from verticapy.machine_learning.memmodel.tree import NonBinaryTree
from verticapy.machine_learning.metrics import FUNCTIONS_DICTIONNARY

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFMachineLearning(vDFScaler):
    @save_verticapy_logs
    def add_duplicates(
        self, weight: Union[int, str], use_gcd: bool = True
    ) -> "vDataFrame":
        """
        Duplicates the :py:class:`~vDataFrame` using
        the input weight.

        Parameters
        ----------
        weight: str | integer
            :py:class:`~vDataColumn` or ``integer``
            representing the weight.
        use_gcd: bool
            If set to True,  uses the GCD (Greatest Common
            Divisor) to reduce all common weights to avoid
            unnecessary duplicates.

        Returns
        -------
        vDataFrame
            the output :py:class:`~vDataFrame`.

        Examples
        ---------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        Let us create a :py:class:`~vDataFrame`
        with multiple columns:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "cats": ["A", "B", "C"],
                    "reps": [2, 4, 8],
                },
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_add_duplicates.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_add_duplicates.html

        We can add duplicates by the weight column:

        .. code-block:: python

            vdf.add_duplicates("reps")

        .. ipython:: python
            :suppress:

            result = vdf.add_duplicates("reps")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_add_duplicates_result.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_add_duplicates_result.html

        .. note::

            VerticaPy will find the greatest common divisor (gcd)
            of the weight column to normalize the weights by it,
            ensuring a meaningful minimum number of occurrences.
            It will then duplicate the different values.
            This function can be highly valuable in machine
            learning for preprocessing and increasing the weight
            of specific rows.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.sample` :
                Sampling the Dataset.
        """
        if isinstance(weight, str):
            weight = self.format_colnames(weight)
            assert self[weight].category() == "int", TypeError(
                "The weight vDataColumn category must be "
                f"'integer', found {self[weight].category()}."
            )
            L = sorted(self[weight].distinct())
            gcd, max_value, n = L[0], L[-1], len(L)
            assert gcd >= 0, ValueError(
                "The weight vDataColumn must only include positive integers."
            )
            if use_gcd:
                if gcd != 1:
                    for i in range(1, n):
                        if gcd != 1:
                            gcd = math.gcd(gcd, L[i])
                        else:
                            break
            else:
                gcd = 1
            columns = self.get_columns(exclude_columns=[weight])
            vdf = self.search(self[weight] != 0, usecols=columns)
            for i in range(2, int(max_value / gcd) + 1):
                vdf = vdf.append(
                    self.search((self[weight] / gcd) >= i, usecols=columns)
                )
        else:
            assert weight >= 2 and isinstance(weight, int), ValueError(
                "The weight must be an integer greater or equal to 2."
            )
            vdf = self.copy()
            for i in range(2, weight + 1):
                vdf = vdf.append(self)
        return vdf

    @save_verticapy_logs
    def cdt(
        self,
        columns: Optional[SQLColumns] = None,
        max_cardinality: int = 20,
        nbins: int = 10,
        tcdt: bool = True,
        drop_transf_cols: bool = True,
    ) -> "vDataFrame":
        """
        Returns the complete  disjunctive table of  the vDataFrame.
        Numerical  features  are transformed  to categorical using
        the ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.discretize` method.
        Applying PCA on TCDT leads to MCA (Multiple correspondence
        analysis).

        .. warning::

            This method can become computationally
            expensive  when used with  categorical
            variables with many categories.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names.
        max_cardinality: int, optional
            For  any categorical variable, keeps the most  frequent
            categories and merges the less frequent categories into
            a new unique category.
        nbins: int, optional
            Number of bins used for the discretization (must be > 1).
        tcdt: bool, optional
            If  set  to  True,   returns  the  transformed  complete
            disjunctive table (TCDT).
        drop_transf_cols: bool, optional
            If  set  to  True,  drops the  columns used  during  the
            transformation.

        Returns
        -------
        vDataFrame
            the CDT relation.

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        Let us create a :py:class:`~vDataFrame`
        with multiple columns:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "id": [0, 1, 2, 3, 4, 5],
                    "cats": ["A", "B", "C", "A", "B", "C"],
                    "vals": [2, 4, 8, 1, 4, 2],
                },
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_cdt.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_cdt.html

        We can create the complete disjunctive
        table of the :py:class:`~vDataFrame`:

        .. code-block:: python

            vdf.cdt(columns=["cats", "vals"], tcdt = False)

        .. ipython:: python
            :suppress:

            result = vdf.cdt(columns=["cats", "vals"], tcdt = False)
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_cdt_result.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_cdt_result.html

        Same can be done to create the transformed
        complete disjunctive table of the
        :py:class:`~vDataFrame`:

        .. code-block:: python

            vdf.cdt(columns=["cats", "vals"], tcdt = True)

        .. ipython:: python
            :suppress:

            result = vdf.cdt(columns=["cats", "vals"], tcdt = True)
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_tcdt_result.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_tcdt_result.html

        .. note::

            This method can be useful to build an MCA (Multiple
            Correspondence Analysis) model based on a PCA
            (Principal Component Analysis) one. The transformed
            complete disjunctive table refers to a table used
            in MCA, where the original categorical data is
            transformed into binary indicators to represent
            the absence or presence of categories.

        .. seealso::

            | :py:class:`~verticapy.machine_learning.vertica.PCA` :
                Principal Component Analysis.
        """
        columns = format_type(columns, dtype=list)
        if len(columns) > 0:
            columns = self.format_colnames(columns)
        else:
            columns = self.get_columns()
        vdf = self.copy()
        columns_to_drop = []
        for elem in columns:
            if vdf[elem].isbool():
                vdf[elem].astype("int")
            elif vdf[elem].isnum():
                vdf[elem].discretize(nbins=nbins)
                columns_to_drop += [elem]
            elif vdf[elem].isdate():
                vdf[elem].drop()
            else:
                vdf[elem].discretize(method="topk", k=max_cardinality)
                columns_to_drop += [elem]
        new_columns = vdf.get_columns()
        vdf.one_hot_encode(
            columns=columns,
            max_cardinality=max(max_cardinality, nbins) + 2,
            drop_first=False,
        )
        new_columns = vdf.get_columns(exclude_columns=new_columns)
        if drop_transf_cols:
            vdf.drop(columns=columns_to_drop)
        if tcdt:
            for elem in new_columns:
                sum_cat = vdf[elem].sum()
                vdf[elem].apply(f"{{}} / {sum_cat} - 1")
        return vdf

    @save_verticapy_logs
    def chaid(
        self,
        response: str,
        columns: SQLColumns,
        nbins: int = 4,
        method: Literal["smart", "same_width"] = "same_width",
        RFmodel_params: Optional[dict] = None,
        **kwargs,
    ) -> NonBinaryTree:
        """
        Returns a CHAID (Chi-square Automatic Interaction Detector)
        tree. CHAID is a decision tree technique based on adjusted
        significance testing (Bonferroni test).

        Parameters
        ----------
        response: str
            Categorical response vDataColumn.
        columns: SQLColumns
            List of the vDataColumn names. The maximum number of
            categories  for  each   categorical   column  is  16;
            categorical  columns  with a higher cardinality  are
            discarded.
        nbins: int, optional
            Integer in the range [2,16], the number of bins used
            to discretize the numerical features.
        method: str, optional
            The  method  with which to discretize the  numerical
            vDataColumns, one of the following:

            - same_width:
                Computes  bins of regular  width.
            - smart:
                Uses a  random forest model on a
                response column to find the best
                interval for discretization.

        RFmodel_params: dict, optional
            Dictionary  of the  parameters of the random  forest
            model used to compute  the best splits  when 'method'
            is 'smart'. If the response column is numerical (but
            not  of type int or bool), this function trains  and
            uses  a random  forest  regressor.  Otherwise,  this
            function trains a random forest classifier.
            For example,  to train a random forest with 20 trees
            and a maximum depth of 10, use:
            ``{"n_estimators": 20, "max_depth": 10}``

        Returns
        -------
        NonBinaryTree
            An independent model containing the result.

        Examples
        ---------
        For this example, we will use the Titanic dataset.

        .. ipython:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample
            datasets that are ideal for training
            and testing purposes. You can explore
            the full list of available datasets in
            the :ref:`api.datasets`, which provides
            detailed information on each dataset and
            how to use them effectively. These datasets
            are invaluable resources for honing your
            data analysis and machine learning skills
            within the VerticaPy environment.

        We can conveniently get the CHAID tree:

        .. code-block::

            tree = data.chaid(
                response = "survived",
                columns = ["sex", "pclass"],
            )
            tree.plot_tree()

        .. ipython:: python
            :suppress:

            tree = data.chaid(
                response = "survived",
                columns = ["sex", "pclass"]
            )
            res = tree.plot_tree()
            res.render(filename='figures/core_vDataFrame_ml_chaid_tree', format='png')


        .. image:: /../figures/core_vDataFrame_ml_chaid_tree.png

        .. note::

            The result is an :py:class:`~verticapy.machine_learning.memmodel.InMemoryModel`
            model. Refer to :py:class:`~verticapy.machine_learning.memmodel.NonBinaryTree`
            for more information on the different methods
            and possibilities.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.chaid_columns` :
                Returns the columns picked by the CHAID algorithm
        """
        RFmodel_params = format_type(RFmodel_params, dtype=dict)
        if "process" not in kwargs or kwargs["process"]:
            columns = format_type(columns, dtype=list)
            assert 2 <= nbins <= 16, ValueError(
                "Parameter 'nbins' must be between 2 and 16, inclusive."
            )
            columns = self.chaid_columns(columns)
            if not columns:
                raise ValueError("No column to process.")
        idx = 0 if ("node_id" not in kwargs) else kwargs["node_id"]
        p = self.pivot_table_chi2(response, columns, nbins, method, RFmodel_params)
        categories, split_predictor, is_numerical, chi2 = (
            p["categories"][0],
            p["index"][0],
            p["is_numerical"][0],
            p["chi2"][0],
        )
        split_predictor_idx = self.get_match_index(
            split_predictor,
            columns
            if "process" not in kwargs or kwargs["process"]
            else kwargs["columns_init"],
        )
        tree = {
            "split_predictor": split_predictor,
            "split_predictor_idx": split_predictor_idx,
            "split_is_numerical": is_numerical,
            "chi2": chi2,
            "is_leaf": False,
            "node_id": idx,
        }
        if is_numerical:
            if categories:
                if ";" in categories[0]:
                    categories = sorted(
                        [float(c.split(";")[1][0:-1]) for c in categories]
                    )
                    ctype = "float"
                else:
                    categories = sorted([int(c) for c in categories])
                    ctype = "int"
            else:
                categories, ctype = [], "int"
        if "process" not in kwargs or kwargs["process"]:
            classes = self[response].distinct()
        else:
            classes = kwargs["classes"]
        if len(columns) == 1:
            if categories:
                if is_numerical:
                    column = "(CASE "
                    for c in categories:
                        column += f"WHEN {split_predictor} <= {c} THEN {c} "
                    column += f"ELSE NULL END)::{ctype} AS {split_predictor}"
                else:
                    column = split_predictor
                result = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataframe.chaid')*/ 
                            {split_predictor}, 
                            {response}, 
                            (cnt / SUM(cnt) 
                                OVER (PARTITION BY {split_predictor}))::float 
                                AS proba 
                        FROM 
                            (SELECT 
                                {column}, 
                                {response}, 
                                COUNT(*) AS cnt 
                             FROM {self} 
                             WHERE {split_predictor} IS NOT NULL 
                               AND {response} IS NOT NULL 
                             GROUP BY 1, 2) x 
                        ORDER BY 1;""",
                    title="Computing the CHAID tree probability.",
                    method="fetchall",
                    sql_push_ext=self._vars["sql_push_ext"],
                    symbol=self._vars["symbol"],
                )
            else:
                result = []
            children = {}
            for c in categories:
                children[c] = {}
                for cl in classes:
                    children[c][cl] = 0.0
            for elem in result:
                children[elem[0]][elem[1]] = elem[2]
            for elem in children:
                idx += 1
                children[elem] = {
                    "prediction": [children[elem][c] for c in children[elem]],
                    "is_leaf": True,
                    "node_id": idx,
                }
            tree["children"] = children
            if "process" not in kwargs or kwargs["process"]:
                return NonBinaryTree(tree=tree, classes=classes)
            return tree, idx
        else:
            tree["children"] = {}
            columns_tmp = columns.copy()
            columns_tmp.remove(split_predictor)
            for c in categories:
                if is_numerical:
                    vdf = self.search(
                        f"""{split_predictor} <= {c}
                        AND {split_predictor} IS NOT NULL
                        AND {response} IS NOT NULL""",
                        usecols=columns_tmp + [response],
                    )
                else:
                    vdf = self.search(
                        f"""{split_predictor} = '{c}'
                        AND {split_predictor} IS NOT NULL
                        AND {response} IS NOT NULL""",
                        usecols=columns_tmp + [response],
                    )
                tree["children"][c], idx = vdf.chaid(
                    response,
                    columns_tmp,
                    nbins,
                    method,
                    RFmodel_params,
                    process=False,
                    columns_init=columns,
                    classes=classes,
                    node_id=idx + 1,
                )
            if "process" not in kwargs or kwargs["process"]:
                return NonBinaryTree(tree=tree, classes=classes)
            return tree, idx

    @save_verticapy_logs
    def chaid_columns(
        self, columns: Optional[SQLColumns] = None, max_cardinality: int = 16
    ) -> list[str]:
        """
        Function used to simplify the code. It returns
        the columns picked by the CHAID algorithm.

        Parameters
        ----------
        columns: SQLColumns
            List of the vDataColumn names.
        max_cardinality: int, optional
            The maximum number of categories for each
            categorical column. Categorical columns
            with a higher cardinality are discarded.

        Returns
        -------
        list
            columns picked by the CHAID algorithm.

        Examples
        ---------
        For this example, we will use the Titanic dataset.

        .. ipython:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample
            datasets that are ideal for training
            and testing purposes. You can explore
            the full list of available datasets in
            the :ref:`api.datasets`, which provides
            detailed information on each dataset and
            how to use them effectively. These datasets
            are invaluable resources for honing your
            data analysis and machine learning skills
            within the VerticaPy environment.

        We can conveniently get the CHAID columns:

        .. ipython:: python

            data.chaid_columns()

        .. note::

            This function is invoked by the CHAID function at
            each step to obtain the potential columns used for
            the split.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.chaid` :
                Returns a CHAID (Chi-square Automatic Interaction Detector) tree.
        """
        columns = format_type(columns, dtype=list)
        columns_tmp = columns.copy()
        if not columns_tmp:
            columns_tmp = self.get_columns()
            remove_cols = []
            for col in columns_tmp:
                if self[col].category() not in ("float", "int", "text") or (
                    self[col].category() == "text"
                    and self[col].nunique() > max_cardinality
                ):
                    remove_cols += [col]
        else:
            remove_cols = []
            columns_tmp = self.format_colnames(columns_tmp)
            for col in columns_tmp:
                if self[col].category() not in ("float", "int", "text") or (
                    self[col].category() == "text"
                    and self[col].nunique() > max_cardinality
                ):
                    remove_cols += [col]
                    if self[col].category() not in ("float", "int", "text"):
                        warning_message = (
                            f"vDataColumn '{col}' is of category '{self[col].category()}'. "
                            "This method only accepts categorical & numerical inputs. "
                            "This vDataColumn was ignored."
                        )
                    else:
                        warning_message = (
                            f"vDataColumn '{col}' has a too high cardinality "
                            f"(> {max_cardinality}). This vDataColumn was ignored."
                        )
                    warnings.warn(warning_message, Warning)
        for col in remove_cols:
            columns_tmp.remove(col)
        return columns_tmp

    @save_verticapy_logs
    def outliers(
        self,
        columns: Optional[SQLColumns] = None,
        name: str = "distribution_outliers",
        threshold: float = 3.0,
        robust: bool = False,
    ) -> "vDataFrame":
        """
        Adds a new :py:class:`~vDataColumn` labeled
        with 0 or 1, where 1 indicates that the record
        is a global outlier.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of the :py:class:`~vDataColumn` names.
            If empty, all  numerical :py:class:`~vDataColumn`
            are used.
        name: str, optional
            Name of the new :py:class:`~vDataColumn`.
        threshold: float, optional
            Threshold equal to the critical score.
        robust: bool
            If set to True, uses the Robust Z-Score
            instead of the Z-Score.

        Returns
        -------
        vDataFrame
            self

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        Let us create a :py:class:`~vDataFrame`
        that has some outliers:

        .. ipython:: python

            import numpy as np

            data = np.random.normal(
                loc = 0,
                scale = 1,
                size = 10,
            )
            data = np.append(data, [100])
            vdf = vp.vDataFrame({"vals": data})

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_outliers.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_outliers.html

        Now we can see which values are outliers
        by using the ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.outliers`
        method:

        .. code-block:: python

            vdf.outliers()

        .. ipython:: python
            :suppress:

            result = vdf.outliers()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_outliers_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_outliers_2.html

        .. note::

            This function can only identify global outliers in
            the distribution. For other types of outliers, it
            is recommended to create machine learning models.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.outliers_plot` :
                Plots the outliers.
        """
        columns = format_type(columns, dtype=list)
        columns = self.format_colnames(columns) if (columns) else self.numcol()
        if not robust:
            result = self.aggregate(func=["std", "avg"], columns=columns).values
        else:
            result = self.aggregate(
                func=["mad", "approx_median"], columns=columns
            ).values
        conditions = []
        for idx, col in enumerate(result["index"]):
            if not robust:
                conditions += [
                    f"""
                    ABS({col} - {result['avg'][idx]}) 
                    / NULLIFZERO({result['std'][idx]}) 
                    > {threshold}"""
                ]
            else:
                conditions += [
                    f"""
                    ABS({col} - {result['approx_median'][idx]}) 
                    / NULLIFZERO({result['mad'][idx]} * 1.4826) 
                    > {threshold}"""
                ]
        self.eval(name, f"(CASE WHEN {' OR '.join(conditions)} THEN 1 ELSE 0 END)")
        return self

    @save_verticapy_logs
    def pivot_table_chi2(
        self,
        response: str,
        columns: Optional[SQLColumns] = None,
        nbins: int = 16,
        method: Literal["smart", "same_width"] = "same_width",
        RFmodel_params: Optional[dict] = None,
    ) -> TableSample:
        """
        Returns the chi-square term using the pivot
        table of the response :py:class:`~vDataColumn`
        against the input :py:class:`~vDataColumn`.

        Parameters
        ----------
        response: str
            Categorical response :py:class:`~vDataColumn`.
        columns: SQLColumns
            List of the vDataColumn names. The maximum
            number of categories for  each categorical
            column is 16; categorical  columns  with a
            higher cardinality are discarded.
        nbins: int, optional
            Integer in the range [2,16], the number of
            bins used to discretize the numerical features.
        method: str, optional
            The  method  with which to discretize the
            numerical :py:class:`~vDataColumn`, one of the
            following:

            - same_width:
                Computes  bins of regular  width.
            - smart:
                Uses a  random forest model on a
                response column to find the best
                interval for discretization.

        RFmodel_params: dict, optional
            Dictionary  of the  parameters of the
            random  forest model used to compute
            the best splits  when ``method = smart``.
            If the response column is numerical (but
            not  of type ``int`` or ``bool``), this
            function trains  and uses a
            ``RandomForestRegressor``. Otherwise, this
            function trains a ``RandomForestClassifier``.
            For example,  to train a random forest with
            20 trees and a maximum depth of 10, use:
            ``{"n_estimators": 20, "max_depth": 10}``

        Returns
        -------
        TableSample
            result.

        Examples
        ---------

        For this example, we will use the Titanic dataset.

        .. ipython:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample
            datasets that are ideal for training
            and testing purposes. You can explore
            the full list of available datasets in
            the :ref:`api.datasets`, which provides
            detailed information on each dataset and
            how to use them effectively. These datasets
            are invaluable resources for honing your
            data analysis and machine learning skills
            within the VerticaPy environment.

        We can conveniently get the chi-squared
        term using the pivot table:

        .. code-block:: python

            data.pivot_table_chi2(response = "survived")

        .. ipython:: python
            :suppress:

            result = data.pivot_table_chi2(response = "survived")
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_pivot_table_chi2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_pivot_table_chi2.html

        .. note::

            This function is employed to construct CHAID trees.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.chaid` :
                Returns a CHAID (Chi-square Automatic Interaction Detector) tree.
        """
        RFmodel_params = format_type(RFmodel_params, dtype=dict)
        columns = format_type(columns, dtype=list)
        columns, response = self.format_colnames(columns, response)
        assert 2 <= nbins <= 16, ValueError(
            "Parameter 'nbins' must be between 2 and 16, inclusive."
        )
        columns = self.chaid_columns(columns)
        for col in columns:
            if quote_ident(response) == quote_ident(col):
                columns.remove(col)
                break
        if not columns:
            raise ValueError("No column to process.")
        if self.shape()[0] == 0:
            return {
                "index": columns,
                "chi2": [0.0 for col in columns],
                "categories": [[] for col in columns],
                "is_numerical": [self[col].isnum() for col in columns],
            }
        vdf = self.copy()
        for col in columns:
            if vdf[col].isnum():
                vdf[col].discretize(
                    method=method,
                    nbins=nbins,
                    response=response,
                    RFmodel_params=RFmodel_params,
                )
        response = vdf.format_colnames(response)
        if response in columns:
            columns.remove(response)
        chi2_list = []
        for col in columns:
            tmp_res = vdf._pivot_table(
                columns=[col, response], max_cardinality=(10000, 100)
            ).to_numpy()
            i = 0
            all_chi2 = []
            for row in tmp_res:
                j = 0
                for col_in_row in row:
                    all_chi2 += [
                        col_in_row**2 / (sum(tmp_res[i]) * sum(tmp_res[:, j]))
                    ]
                    j += 1
                i += 1
            val = sum(sum(tmp_res)) * (sum(all_chi2) - 1)
            k, r = tmp_res.shape
            dof = (k - 1) * (r - 1)
            pval = scipy_st.chi2.sf(val, dof)
            chi2_list += [(col, val, pval, dof, vdf[col].distinct(), self[col].isnum())]
        chi2_list = sorted(chi2_list, key=lambda tup: tup[1], reverse=True)
        result = {
            "index": [chi2[0] for chi2 in chi2_list],
            "chi2": [chi2[1] for chi2 in chi2_list],
            "p_value": [chi2[2] for chi2 in chi2_list],
            "dof": [chi2[3] for chi2 in chi2_list],
            "categories": [chi2[4] for chi2 in chi2_list],
            "is_numerical": [chi2[5] for chi2 in chi2_list],
        }
        return TableSample(result)

    @save_verticapy_logs
    def polynomial_comb(
        self, columns: Optional[SQLColumns] = None, r: int = 2
    ) -> "vDataFrame":
        """
        Returns a vDataFrame containing the different
        product combinations  of   the  input
        :py:class:`~vDataColumn`. This function is ideal for
        bivariate analysis.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the :py:class:`~vDataColumn` names.
            If empty, all numerical :py:class:`~vDataColumn`
            are used.
        r: int, optional
            Degree of the polynomial.

        Returns
        -------
        vDataFrame
            the Polynomial object.

        Examples
        --------

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        Let us create a :py:class:`~vDataFrame`
        with multiple columns:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": [0, 7, 8],
                    "col3": [3, 11, 93],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_polynomial_comb.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_polynomial_comb.html

        We can create a new :py:class:`~vDataFrame` that
        has a combination of the original columns using
        the ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.polynomial_comb`
        method:

        .. ipython:: python

            new_vdf = vdf.polynomial_comb(r = 2)

        .. ipython:: python
            :suppress:

            result = new_vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_polynomial_comb_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_polynomial_comb_2.html

        .. note::

            This function is highly useful for data preparation, as
            certain combinations of variables may be relevant for
            predicting a specific column. It can be beneficial to
            combine it with a correlation matrix to determine if
            any of the created combinations can influence the
            response column.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.corr` :
                Computes the correlation matrix.
        """
        columns = format_type(columns, dtype=list)
        if len(columns) == 0:
            numcol = self.numcol()
        else:
            numcol = self.format_colnames(columns)
        vdf = self.copy()
        all_comb = combinations_with_replacement(numcol, r=r)
        for elem in all_comb:
            name = "_".join(elem)
            vdf.eval(name.replace('"', ""), expr=" * ".join(elem))
        return vdf

    @save_verticapy_logs
    def recommend(
        self,
        unique_id: str,
        item_id: str,
        method: Literal["count", "avg", "median"] = "count",
        rating: Union[str, tuple] = "",
        ts: Optional[str] = None,
        start_date: PythonScalar = "",
        end_date: PythonScalar = "",
    ) -> "vDataFrame":
        """
        Recommend items based on the Collaborative
        Filtering (CF) technique.  The implementation
        is the same as APRIORI algorithm, but is
        limited to pairs of items.

        Parameters
        ----------
        unique_id: str
            Input :py:class:`~vDataColumn` corresponding
            to a unique ID. It serves as a primary key
            in another dataset. In our context, it
            represents an operation, such as a basket ID,
            which includes multiple sub-transactions.
        item_id: str
            Input :py:class:`~vDataColumn` corresponding
            to an item ID. It is a secondary key used to
            compute the different pairs.
        method: str, optional
            Method used to recommend.

            - count:
                Each item will be recommended based on
                frequencies of the  different pairs of
                items.
            - avg:
                Each item will be recommended based on
                the  average rating of  the  different
                item  pairs  with  a differing  second
                element.
            - median:
                Each item will be recommended based on
                the  median  rating of  the  different
                item  pairs  with a  differing  second
                element.

        rating: str | tuple, optional
            Input :py:class:`~vDataColumn` including the
            items rating. If the ``rating`` type is
            ``tuple``, it must be composed of 3 elements:

            (r_vdf, r_item_id, r_name) where:

            **r_vdf** is an input :py:class:`~vDataFrame`.

            **r_item_id** is an  input :py:class:`~vDataColumn`
            which must includes the same id as ``item_id``.

            **r_name** is an input :py:class:`~vDataColumn`
            including the items rating.

        ts: str, optional
            TS (Time Series) :py:class:`~vDataColumn`
            used to order the data. The
            :py:class:`~vDataColumn` type must be date
            (date, datetime, timestamp...) or numerical.
        start_date: str | PythonNumber | date, optional
            Input Start Date. For example, ``time = '03-11-1993'``
            will filter the data when ``ts``  is less
            than November 1993 the 3rd.
        end_date: str | PythonNumber | date, optional
            Input End Date.  For example,  ``time = '03-11-1993'``
            will filter the data when ``ts`` is greater
            than November 1993 the 3rd.

        Returns
        -------
        vDataFrame
            The :py:class:`~vDataFrame`
            of the recommendation.

        Examples
        --------

        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        Let us create a :py:class:`~vDataFrame`
        which has some purchase transaction data:

        - transaction_id:
            Unique ID for a transaction.

        - item_id:
            The unique ID for different items that
            were purchased.

        - rating:
            Rating provided by the user for the item
            purchased.

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "transaction_id": [1, 1, 1, 2, 2, 3, 3, 3],
                    "item_id": ["A", "B", "C", "B", "C", "A", "B", "C"],
                    "rating": [8, 5, 1, 6, 2, 9, 4, 3],
                },
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_suggest.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_suggest.html

        We can easily create the recommend
        table from the above data:

        .. ipython:: python

            recommendations = vdf.recommend(
                unique_id = "transaction_id",
                item_id = "item_id",
                method = "avg",
                rating = "rating",
            )

        .. ipython:: python
            :suppress:

            result = recommendations
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_suggest_result.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_suggest_result.html

        .. note::

            This function is highly useful for basket
            analysis and can be employed to derive
            valuable recommendations.

        Let's look at another example
        involving timestamp values:

        .. ipython:: python

            # Create a vDataFrame with the transaction data
            vdf = vp.vDataFrame(
                {
                    "transaction_id": [1, 1, 1, 2, 2, 3, 3, 3],
                    "item_id": ["A", "B", "C", "B", "C", "A", "B", "C"],
                    "rating": [8, 5, 1, 6, 2, 9, 4, 3],
                    "date": [
                        "2021-1-1",
                        "2021-1-1",
                        "2021-1-1",
                        "2021-1-4",
                        "2021-1-4",
                        "2021-1-21",
                        "2021-1-21",
                        "2021-1-21",
                    ],
                },
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_suggest_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_suggest_2.html

        Then we can use the timestamp column
        to filter the recommendation results:

        .. ipython:: python

            recommendations = vdf.recommend(
                unique_id = "transaction_id",
                item_id = "item_id",
                method = "avg",
                rating = "rating",
                ts = "date",
                start_date = "2021-1-1",
                end_date = "2021-1-5",
            )

        .. ipython:: python
            :suppress:

            result = recommendations
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_suggest_result_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_suggest_result_2.html

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.add_duplicates` :
                Add duplicates of values based on weights.
        """
        unique_id, item_id, ts = self.format_colnames(unique_id, item_id, ts)
        vdf = self.copy()
        assert (
            method == "count" or rating
        ), f"Method '{method}' can not be used if parameter 'rating' is empty."
        if rating:
            assert isinstance(rating, str) or len(rating) == 3, ValueError(
                "Parameter 'rating' must be of type str or composed of "
                "exactly 3 elements: (r_vdf, r_item_id, r_name)."
            )
            assert (
                method != "count"
            ), "Method 'count' can not be used if parameter 'rating' is defined."
            rating = self.format_colnames(rating)
        if ts:
            if start_date and end_date:
                vdf = self.search(f"{ts} BETWEEN '{start_date}' AND '{end_date}'")
            elif start_date:
                vdf = self.search(f"{ts} >= '{start_date}'")
            elif end_date:
                vdf = self.search(f"{ts} <= '{end_date}'")
        vdf = (
            vdf.join(
                vdf,
                how="left",
                on={unique_id: unique_id},
                expr1=[f"{item_id} AS item1"],
                expr2=[f"{item_id} AS item2"],
            )
            .groupby(["item1", "item2"], ["COUNT(*) AS cnt"])
            .search("item1 != item2 AND cnt > 1")
        )
        order_columns = "cnt DESC"
        if method in ("avg", "median"):
            fun = "AVG" if method == "avg" else "APPROXIMATE_MEDIAN"
            if isinstance(rating, str):
                r_vdf = self.groupby([item_id], [f"{fun}({rating}) AS score"])
                r_item_id = item_id
                r_name = "score"
            else:
                r_vdf, r_item_id, r_name = rating
                r_vdf = r_vdf.groupby([r_item_id], [f"{fun}({r_name}) AS {r_name}"])
            vdf = vdf.join(
                r_vdf,
                how="left",
                on={"item1": r_item_id},
                expr2=[f"{r_name} AS score1"],
            ).join(
                r_vdf,
                how="left",
                on={"item2": r_item_id},
                expr2=[f"{r_name} AS score2"],
            )
            order_columns = "score2 DESC, score1 DESC, cnt DESC"
        vdf["rank"] = f"ROW_NUMBER() OVER (PARTITION BY item1 ORDER BY {order_columns})"
        return vdf

    @save_verticapy_logs
    def score(
        self,
        y_true: str,
        y_score: str,
        metric: Literal[tuple(FUNCTIONS_DICTIONNARY)],
    ) -> float:
        """
        Computes the score using the input columns
        and the input metric.

        Parameters
        ----------
        y_true: str
            Response column.
        y_score: str
            Prediction.
        metric: str
            The metric used to compute the score.

            **For Classification**

            - accuracy:
                Accuracy.

                .. math::

                    Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}

            - auc:
                Area Under the Curve (ROC).

                .. math::

                    AUC = \int_{0}^{1} TPR(FPR) \, dFPR

            - ba:
                Balanced Accuracy.

                .. math::

                    BA = \\frac{TPR + TNR}{2}

            - best_cutoff:
                Cutoff  which  optimised
                the ROC Curve prediction.

            - bm:
                Informedness.

                .. math::

                    BA = TPR + TNR - 1

            - csi:
                Critical  Success  Index.

                .. math::

                    index  = \\frac{TP}{TP + FN + FP}

            - f1:
                F1 Score

                .. math::

                    F_1 Score = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall}

            - fdr:
                False Discovery Rate.

                .. math::

                    FDR = 1 - PPV

            - fm:
                Fowlkes-Mallows index.

                .. math::

                    FM = \\sqrt{PPV * TPR}

            - fnr:
                False Negative Rate.

                .. math::

                    FNR = \\frac{FN}{FN + TP}

            - for:
                False Omission Rate.

                .. math::

                    FOR = 1 - NPV

            - fpr:
                False Positive Rate.

                .. math::

                    \\frac{FP}{FP + TN}

            - logloss:
                Log Loss.

                .. math::

                    Loss = -\\frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \\right)

            - lr+:
                Positive Likelihood Ratio.

                .. math::

                    LR+ = \\frac{TPR}{FPR}

            - lr-:
                Negative Likelihood Ratio.

                .. math::

                    LR- = \\frac{FNR}{TNR}

            - dor:
                Diagnostic Odds Ratio.

                .. math::

                    DOR = \\frac{TP \\times TN}{FP \\times FN}

            - mcc:
                Matthews Correlation Coefficient.

                .. math::

                    MCC = \\frac{TP \\times TN - FP \\times FN}{\\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}

            - mk:
                Markedness.

                .. math::

                    MK = PPV + NPV - 1

            - npv:
                Negative Predictive Value.

                .. math::

                    NPV = \\frac{TN}{TN + FN}

            - prc_auc:
                Area Under the Curve (PRC).

                .. math::

                    AUC = \int_{0}^{1} Precision(Recall) \, dRecall

            - precision:
                Precision.

                .. math::

                    Precision = TP / (TP + FP)

            - pt:
                Prevalence Threshold.

                .. math::

                    threshold = \\frac{\\sqrt{FPR}}{\\sqrt{TPR} + \\sqrt{FPR}}

            - recall:
                Recall.

                .. math::

                    Recall = \\frac{TP}{TP + FN}

            - specificity:
                Specificity.

                .. math::

                    Specificity = \\frac{TN}{TN + FP}

            **For Regression**

            - max:
                Max Error.

                .. math::

                    ME = \max_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

            - mae:
                Mean Absolute Error.

                .. math::

                    MAE = \\frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

            - median:
                Median Absolute Error.

                .. math::

                    MedAE = \\text{median}_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

            - mse:
                Mean Squared Error.

                .. math::

                    MSE = \\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \\right)^2

            - msle:
                Mean Squared Log Error.

                .. math::

                    MSLE = \\frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2

            - r2:
                R squared coefficient.

                .. math::

                    R^2 = 1 - \\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \\bar{y})^2}

            - var:
                Explained Variance.

                .. math::

                    \\text{Explained Variance} = 1 - \\frac{Var(y - \hat{y})}{Var(y)}

        Returns
        -------
        float
            score.

        Examples
        ---------

        Let us build a quick ML model and calculate
        the score of its predictions.

        Load data for machine learning
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        For this example, we will
        use the winequality dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_winequality()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html

        .. note::

            VerticaPy offers a wide range of sample
            datasets that are ideal for training
            and testing purposes. You can explore
            the full list of available datasets in
            the :ref:`api.datasets`, which provides
            detailed information on each dataset and
            how to use them effectively. These datasets
            are invaluable resources for honing your
            data analysis and machine learning skills
            within the VerticaPy environment.

        You can easily divide your dataset into training
        and testing subsets using the
        ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.train_test_split` method.

        .. code-block:: python

            data = vpd.load_winequality()
            train, test = data.train_test_split(test_size = 0.2)

        .. ipython:: python
            :suppress:

                import verticapy as vp
                import verticapy.datasets as vpd
                data = vpd.load_winequality()
                train, test = data.train_test_split(test_size = 0.2)

        Model Initialization
        ^^^^^^^^^^^^^^^^^^^^^

        First we import the
        :py:class:`~verticapy.vertica.machine_learning.LinearRegression`
        model:

        .. code-block::

            from verticapy.machine_learning.vertica import LinearRegression

        Then we can create the model:

        .. code-block::

            model = LinearRegression(
                tol = 1e-6,
                max_iter = 100,
                solver = 'newton',
                fit_intercept = True,
            )

        .. ipython:: python
            :suppress:

            from verticapy.machine_learning.vertica import LinearRegression
            model = LinearRegression(
                tol = 1e-6,
                max_iter = 100,
                solver = 'newton',
                fit_intercept = True,
            )

        Model Training
        ^^^^^^^^^^^^^^^

        We can now fit the model:

        .. ipython:: python

            model.fit(
                train,
                [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                    "chlorides",
                    "density"
                ],
                "quality",
                test,
            )

        Prediction
        ^^^^^^^^^^^

        Prediction is straight-forward:

        .. ipython:: python
            :suppress:

            result = model.predict(
                test,
                [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                    "chlorides",
                    "density"
                ],
                "prediction",
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_score_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. code-block:: python

            result = model.predict(
                test,
                [
                    "fixed_acidity",
                    "volatile_acidity",
                    "citric_acid",
                    "residual_sugar",
                    "chlorides",
                    "density"
                ],
                "prediction",
            )

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_score_1.html

        Score
        ^^^^^^

        Finally we can calculate the scores:

        .. ipython:: python

            # R2
            result.score("quality", "prediction", metric = "r2")
            #  MSE
            result.score("quality", "prediction", metric = "mse")
            #  Max Error
            result.score("quality", "prediction", metric = "max")

        .. note::

            If the prediction is already part of the dataset,
            there is no need to use a model to compute a
            prediction column. Use your column directly.

        .. seealso::

            | :py:class:`~verticapy.machine_learning.vertica.LinearRegression` :
                Linear Regression model.
        """
        y_true, y_score = self.format_colnames(y_true, y_score)
        args = [y_true, y_score, self._genSQL()]
        return FUNCTIONS_DICTIONNARY[metric](*args)

    @save_verticapy_logs
    def sessionize(
        self,
        ts: str,
        by: Optional[SQLColumns] = None,
        session_threshold: str = "30 minutes",
        name: str = "session_id",
    ) -> "vDataFrame":
        """
        Adds a new :py:class:`~vDataColumn` to the
        :py:class:`~vDataFrame` that corresponds to
        sessions (user activity during  a specific
        time). A session ends when ``ts - lag(ts)``
        is greater than a specific threshold.

        Parameters
        ----------
        ts: str
            :py:class:`~vDataColumn` used  as timeline.
            It is used to order the data. It can be a
            numerical or type date (``date``, ``datetime``,
            ``timestamp``...) :py:class:`~vDataColumn`.
        by: SQLColumns, optional
            :py:class:`~vDataColumn` used in the partition.
        session_threshold: str, optional
            This parameter is the threshold that determines
            the end of the session. For example, if it is set to
            '10 minutes', the session  ends after 10 minutes  of
            inactivity.
        name: str, optional
            The session name.

        Returns
        -------
        vDataFrame
            self

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        Let us create a :py:class:`~vDataFrame`
        with multiple columns:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "time": [
                        "1993-11-03 00:00:00",
                        "1993-11-03 00:14:00",
                        "1993-11-03 00:07:00",
                        "1993-11-03 01:00:00",
                        "1993-11-03 01:05:05",
                        "1993-11-03 01:15:05",
                        "1993-11-03 01:45:01",
                    ],
                    "val": [0., 1., 2., 4., 5., 5.5, 6., 8],
                },
            )

        We can make sure the correct data type is assigned:

        .. code-block:: python

            vdf["time"].astype("datetime")

        .. ipython:: python
            :suppress:

            vdf["time"].astype("datetime")
            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_sessionize.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_sessionize.html

        Now we can create sessions based on
        custom time intervals:

        .. code-block:: python

            vdf.sessionize(
                ts = "time",
                session_threshold = "15 minutes",
            )

        .. ipython:: python
            :suppress:

            vdf.sessionize(
                ts = "time",
                session_threshold = "15 minutes"
            )
            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_ml_sessionize_result.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_ml_sessionize_result.html

        .. note::

            This method is particularly useful when dealing with
            clickstream analytics, enabling the creation of sessions
            as part of data preparation for machine learning.
            It helps understand user behaviors and interactions as
            they navigate through applications, for example.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.analytic` :
                Use an advanced analytical function on one or
                two specific :py:class:`~vDataColumn`.
        """
        by = format_type(by, dtype=list)
        by, ts = self.format_colnames(by, ts)
        partition = ""
        if by:
            partition = f"PARTITION BY {', '.join(by)}"
        expr = f"""CONDITIONAL_TRUE_EVENT(
                    {ts}::timestamp - LAG({ts}::timestamp) 
                  > '{session_threshold}') 
                  OVER ({partition} ORDER BY {ts})"""
        return self.eval(name=name, expr=expr)

    @save_verticapy_logs
    def train_test_split(
        self,
        test_size: float = 0.33,
        order_by: Union[None, str, list, dict] = None,
        random_state: int = None,
    ) -> tuple["vDataframe", "vDataFrame"]:
        """
        Creates two vDataFrames (train/test), which can be used
        to  evaluate a model. The intersection between the train
        and test set is empty only if you specify a unique
        ``order_by``.

        Parameters
        ----------
        test_size: float, optional
            Proportion of the test set  compared to the training
            set.
        order_by: str | dict | list, optional
            List of the :py:class:`~vDataColumn` used to sort the data,
            using asc order or a ``dictionary`` of all sorting methods.
            For example,  to sort by "column1" ASC and "column2"
            DESC, write: ``{"column1": "asc", "column2": "desc"}``
            Without this parameter,  the seeded random number used
            to split the data into train and test cannot guarantee
            that no collision will occur. Using this parameter
            avoids the possibility of collisions.
        random_state: int, optional
            Integer used to seed the randomness.

        Returns
        -------
        tuple
            (train vDataFrame, test vDataFrame)

        Examples
        --------
        For this example, we will use the Titanic dataset.

        .. ipython:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample
            datasets that are ideal for training
            and testing purposes. You can explore
            the full list of available datasets in
            the :ref:`api.datasets`, which provides
            detailed information on each dataset and
            how to use them effectively. These datasets
            are invaluable resources for honing your
            data analysis and machine learning skills
            within the VerticaPy environment.

        Let us check the size of the dataset:

        .. ipython:: python

            data.shape()

        Now, we can easily divide the dataset into
        training and testing subsets using the
        ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.train_test_split`
        method.

        .. ipython:: python

            train, test = data.train_test_split(test_size = 0.2)

        We can confirm the size of the train set:

        .. ipython:: python

            train.shape()

        And test set:

        .. ipython:: python

            test.shape()

        .. warning::

            This function utilizes seeded random values
            to ensure reproducibility, requiring an index
            to prevent collisions. It is highly recommended
            to save the results as tables using the
            ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.to_db` method to expedite
            the process and ensure consistent results.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.sample` :
                Downsamples the input :py:class:`~vDataFrame`.
        """
        order_by = format_type(order_by, dtype=list)
        order_by = self._get_sort_syntax(order_by)
        if not random_state:
            random_state = conf.get_option("random_state")
        random_seed = (
            random_state
            if isinstance(random_state, int)
            else random.randint(int(-10e6), int(10e6))
        )
        random_func = _seeded_random_function(random_seed)
        q = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.train_test_split')*/ 
                    APPROXIMATE_PERCENTILE({random_func} 
                        USING PARAMETERS percentile = {test_size}) 
                FROM {self}""",
            title="Computing the seeded numbers quantile.",
            method="fetchfirstelem",
        )
        test_table = f"""
            SELECT * 
            FROM {self} 
            WHERE {random_func} <= {q}{order_by}"""
        train_table = f"""
            SELECT * 
            FROM {self} 
            WHERE {random_func} > {q}{order_by}"""
        return (
            create_new_vdf(train_table),
            create_new_vdf(test_table),
        )
