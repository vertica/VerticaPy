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
# Standard Python Modules
import warnings, math
from typing import Literal, Union

# VerticaPy Modules
from verticapy._utils._collect import save_verticapy_logs
from verticapy.errors import ParameterError
from verticapy._utils._cast import to_varchar
from verticapy._config.config import OPTIONS
from verticapy.sql.drop import drop
from verticapy._utils._sql._execute import _executeSQL
from verticapy._utils._gen import gen_tmp_name


class vDFENCODE:
    @save_verticapy_logs
    def one_hot_encode(
        self,
        columns: Union[str, list] = [],
        max_cardinality: int = 12,
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ):
        """
    Encodes the vDataColumns using the One Hot Encoding algorithm.

    Parameters
    ----------
    columns: str / list, optional
        List of the vDataColumns to use to train the One Hot Encoding model. If empty, 
        only the vDataColumns having a cardinality lesser than 'max_cardinality' will 
        be used.
    max_cardinality: int, optional
        Cardinality threshold to use to determine if the vDataColumn will be taken into
        account during the encoding. This parameter is used only if the parameter 
        'columns' is empty.
    prefix_sep: str, optional
        Prefix delimitor of the dummies names.
    drop_first: bool, optional
        Drops the first dummy to avoid the creation of correlated features.
    use_numbers_as_suffix: bool, optional
        Uses numbers as suffix instead of the vDataColumns categories.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame[].decode       : Encodes the vDataColumn using a user defined Encoding.
    vDataFrame[].discretize   : Discretizes the vDataColumn.
    vDataFrame[].get_dummies  : Computes the vDataColumns result of One Hot Encoding.
    vDataFrame[].label_encode : Encodes the vDataColumn using the Label Encoding.
    vDataFrame[].mean_encode  : Encodes the vDataColumn using the Mean Encoding of a response.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns)
        if not (columns):
            columns = self.get_columns()
        cols_hand = True if (columns) else False
        for column in columns:
            if self[column].nunique(True) < max_cardinality:
                self[column].get_dummies(
                    "", prefix_sep, drop_first, use_numbers_as_suffix
                )
            elif cols_hand and OPTIONS["print_info"]:
                warning_message = (
                    f"The vDataColumn '{column}' was ignored because of "
                    "its high cardinality.\nIncrease the parameter "
                    "'max_cardinality' to solve this issue or use "
                    "directly the vDataColumn get_dummies method."
                )
                warnings.warn(warning_message, Warning)
        return self

    get_dummies = one_hot_encode


class vDCENCODE:
    @save_verticapy_logs
    def cut(
        self,
        breaks: list,
        labels: list = [],
        include_lowest: bool = True,
        right: bool = True,
    ):
        """
    Discretizes the vDataColumn using the input list. 

    Parameters
    ----------
    breaks: list
        List of values used to cut the vDataColumn.
    labels: list, optional
        Labels used to name the new categories. If empty, names will be generated.
    include_lowest: bool, optional
        If set to True, the lowest element of the list will be included.
    right: bool, optional
        How the intervals should be closed. If set to True, the intervals will be
        closed on the right.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].apply : Applies a function to the input vDataColumn.
        """
        assert self.isnum() or self.isdate(), TypeError(
            "cut only works on numerical / date-like vDataColumns."
        )
        assert len(breaks) >= 2, ParameterError(
            "Length of parameter 'breaks' must be greater or equal to 2."
        )
        assert len(breaks) == len(labels) + 1 or not (labels), ParameterError(
            "Length of parameter breaks must be equal to the length of parameter "
            "'labels' + 1 or parameter 'labels' must be empty."
        )
        conditions, column = [], self.alias
        for idx in range(len(breaks) - 1):
            first_elem, second_elem = breaks[idx], breaks[idx + 1]
            if right:
                op1, op2, close_l, close_r = "<", "<=", "]", "]"
            else:
                op1, op2, close_l, close_r = "<=", "<", "[", "["
            if idx == 0 and include_lowest:
                op1, close_l = "<=", "["
            elif idx == 0:
                op1, close_l = "<", "]"
            if labels:
                label = labels[idx]
            else:
                label = f"{close_l}{first_elem};{second_elem}{close_r}"
            conditions += [
                f"'{first_elem}' {op1} {column} AND {column} {op2} '{second_elem}' THEN '{label}'"
            ]
        expr = "CASE WHEN " + " WHEN ".join(conditions) + " END"
        self.apply(func=expr)

    @save_verticapy_logs
    def discretize(
        self,
        method: Literal["auto", "smart", "same_width", "same_freq", "topk"] = "auto",
        h: Union[int, float] = 0,
        nbins: int = -1,
        k: int = 6,
        new_category: str = "Others",
        RFmodel_params: dict = {},
        response: str = "",
        return_enum_trans: bool = False,
    ):
        """
    Discretizes the vDataColumn using the input method.

    Parameters
    ----------
    method: str, optional
        The method to use to discretize the vDataColumn.
            auto       : Uses method 'same_width' for numerical vDataColumns, cast 
                the other types to varchar.
            same_freq  : Computes bins with the same number of elements.
            same_width : Computes regular width bins.
            smart      : Uses the Random Forest on a response column to find the most 
                relevant interval to use for the discretization.
            topk       : Keeps the topk most frequent categories and merge the other 
                into one unique category.
    h: int / float, optional
        The interval size to convert to use to convert the vDataColumn. If this parameter 
        is equal to 0, an optimised interval will be computed.
    nbins: int, optional
        Number of bins used for the discretization (must be > 1)
    k: int, optional
        The integer k of the 'topk' method.
    new_category: str, optional
        The name of the merging category when using the 'topk' method.
    RFmodel_params: dict, optional
        Dictionary of the Random Forest model parameters used to compute the best splits 
        when 'method' is set to 'smart'. A RF Regressor will be trained if the response
        is numerical (except ints and bools), a RF Classifier otherwise.
        Example: Write {"n_estimators": 20, "max_depth": 10} to train a Random Forest with
        20 trees and a maximum depth of 10.
    response: str, optional
        Response vDataColumn when method is set to 'smart'.
    return_enum_trans: bool, optional
        Returns the transformation instead of the vDataFrame parent and do not apply
        it. This parameter is very useful for testing to be able to look at the final 
        transformation.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].decode       : Encodes the vDataColumn with user defined Encoding.
    vDataFrame[].get_dummies  : Encodes the vDataColumn with One-Hot Encoding.
    vDataFrame[].label_encode : Encodes the vDataColumn with Label Encoding.
    vDataFrame[].mean_encode  : Encodes the vDataColumn using the mean encoding of a response.
        """
        from verticapy.learn.ensemble import (
            RandomForestRegressor,
            RandomForestClassifier,
        )

        if self.isnum() and method == "smart":
            schema = OPTIONS["temp_schema"]
            if not (schema):
                schema = "public"
            tmp_view_name = gen_tmp_name(schema=schema, name="view")
            tmp_model_name = gen_tmp_name(schema=schema, name="model")
            assert nbins >= 2, ParameterError(
                "Parameter 'nbins' must be greater or equals to 2 in case "
                "of discretization using the method 'smart'."
            )
            assert response, ParameterError(
                "Parameter 'response' can not be empty in case of "
                "discretization using the method 'smart'."
            )
            response = self.parent.format_colnames(response)
            drop(tmp_view_name, method="view")
            self.parent.to_db(tmp_view_name)
            drop(tmp_model_name, method="model")
            if self.parent[response].category() == "float":
                model = RandomForestRegressor(tmp_model_name)
            else:
                model = RandomForestClassifier(tmp_model_name)
            model.set_params({"n_estimators": 20, "max_depth": 8, "nbins": 100})
            model.set_params(RFmodel_params)
            parameters = model.get_params()
            try:
                model.fit(tmp_view_name, [self.alias], response)
                query = [
                    f"""
                    (SELECT 
                        READ_TREE(USING PARAMETERS 
                            model_name = '{tmp_model_name}', 
                            tree_id = {i}, 
                            format = 'tabular'))"""
                    for i in range(parameters["n_estimators"])
                ]
                query = f"""
                    SELECT 
                        /*+LABEL('vDataColumn.discretize')*/ split_value 
                    FROM 
                        (SELECT 
                            split_value, 
                            MAX(weighted_information_gain) 
                        FROM ({' UNION ALL '.join(query)}) VERTICAPY_SUBTABLE 
                        WHERE split_value IS NOT NULL 
                        GROUP BY 1 ORDER BY 2 DESC LIMIT {nbins - 1}) VERTICAPY_SUBTABLE 
                    ORDER BY split_value::float"""
                result = _executeSQL(
                    query=query,
                    title="Computing the optimized histogram nbins using Random Forest.",
                    method="fetchall",
                    sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
                )
                result = [x[0] for x in result]
            finally:
                drop(tmp_view_name, method="view")
                drop(tmp_model_name, method="model")
            result = [self.min()] + result + [self.max()]
        elif method == "topk":
            assert k >= 2, ParameterError(
                "Parameter 'k' must be greater or equals to 2 in "
                "case of discretization using the method 'topk'"
            )
            distinct = self.topk(k).values["index"]
            category_str = to_varchar(self.category())
            X_str = ", ".join([f"""'{str(x).replace("'", "''")}'""" for x in distinct])
            new_category_str = new_category.replace("'", "''")
            trans = (
                f"""(CASE 
                        WHEN {category_str} IN ({X_str})
                        THEN {category_str} || '' 
                        ELSE '{new_category_str}' 
                     END)""",
                "varchar",
                "text",
            )
        elif self.isnum() and method == "same_freq":
            assert nbins >= 2, ParameterError(
                "Parameter 'nbins' must be greater or equals to 2 in case "
                "of discretization using the method 'same_freq'"
            )
            count = self.count()
            nb = int(float(count / int(nbins)))
            assert nb != 0, Exception(
                "Not enough values to compute the Equal Frequency discretization"
            )
            total, query, nth_elems = nb, [], []
            while total < int(float(count / int(nbins))) * int(nbins):
                nth_elems += [str(total)]
                total += nb
            possibilities = ", ".join(["1"] + nth_elems + [str(count)])
            where = f"WHERE _verticapy_row_nb_ IN ({possibilities})"
            query = f"""
                SELECT /*+LABEL('vDataColumn.discretize')*/ 
                    {self.alias} 
                FROM (SELECT 
                        {self.alias}, 
                        ROW_NUMBER() OVER (ORDER BY {self.alias}) AS _verticapy_row_nb_ 
                      FROM {self.parent.__genSQL__()} 
                      WHERE {self.alias} IS NOT NULL) VERTICAPY_SUBTABLE {where}"""
            result = _executeSQL(
                query=query,
                title="Computing the equal frequency histogram bins.",
                method="fetchall",
                sql_push_ext=self.parent._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self.parent._VERTICAPY_VARIABLES_["symbol"],
            )
            result = [elem[0] for elem in result]
        elif self.isnum() and method in ("same_width", "auto"):
            if not (h) or h <= 0:
                if nbins <= 0:
                    h = self.numh()
                else:
                    h = (self.max() - self.min()) * 1.01 / nbins
                if h > 0.01:
                    h = round(h, 2)
                elif h > 0.0001:
                    h = round(h, 4)
                elif h > 0.000001:
                    h = round(h, 6)
                if self.category() == "int":
                    h = int(max(math.floor(h), 1))
            floor_end = -1 if (self.category() == "int") else ""
            if (h > 1) or (self.category() == "float"):
                trans = (
                    f"'[' || FLOOR({{}} / {h}) * {h} || ';' || (FLOOR({{}} / {h}) * {h} + {h}{floor_end}) || ']'",
                    "varchar",
                    "text",
                )
            else:
                trans = ("FLOOR({}) || ''", "varchar", "text")
        else:
            trans = ("{} || ''", "varchar", "text")
        if (self.isnum() and method == "same_freq") or (
            self.isnum() and method == "smart"
        ):
            n = len(result)
            trans = "(CASE "
            for i in range(1, n):
                trans += f"""
                    WHEN {{}} 
                        BETWEEN {result[i - 1]} 
                        AND {result[i]} 
                    THEN '[{result[i - 1]};{result[i]}]' """
            trans += " ELSE NULL END)"
            trans = (trans, "varchar", "text")
        if return_enum_trans:
            return trans
        else:
            self.transformations += [trans]
            sauv = {}
            for elem in self.catalog:
                sauv[elem] = self.catalog[elem]
            self.parent.__update_catalog__(erase=True, columns=[self.alias])
            try:
                if "count" in sauv:
                    self.catalog["count"] = sauv["count"]
                    self.catalog["percent"] = (
                        100 * sauv["count"] / self.parent.shape()[0]
                    )
            except:
                pass
            self.parent.__add_to_history__(
                f"[Discretize]: The vDataColumn {self.alias} was discretized."
            )
        return self.parent

    @save_verticapy_logs
    def one_hot_encode(
        self,
        prefix: str = "",
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ):
        """
    Encodes the vDataColumn with the One-Hot Encoding algorithm.

    Parameters
    ----------
    prefix: str, optional
        Prefix of the dummies.
    prefix_sep: str, optional
        Prefix delimitor of the dummies.
    drop_first: bool, optional
        Drops the first dummy to avoid the creation of correlated features.
    use_numbers_as_suffix: bool, optional
        Uses numbers as suffix instead of the vDataColumns categories.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].decode       : Encodes the vDataColumn with user defined Encoding.
    vDataFrame[].discretize   : Discretizes the vDataColumn.
    vDataFrame[].label_encode : Encodes the vDataColumn with Label Encoding.
    vDataFrame[].mean_encode  : Encodes the vDataColumn using the mean encoding of a response.
        """
        from verticapy.core.vdataframe.base import vDataColumn

        distinct_elements = self.distinct()
        if distinct_elements not in ([0, 1], [1, 0]) or self.isbool():
            all_new_features = []
            if not (prefix):
                prefix = self.alias.replace('"', "") + prefix_sep.replace('"', "_")
            else:
                prefix = prefix.replace('"', "_") + prefix_sep.replace('"', "_")
            n = 1 if drop_first else 0
            for k in range(len(distinct_elements) - n):
                distinct_elements_k = str(distinct_elements[k]).replace('"', "_")
                if use_numbers_as_suffix:
                    name = f'"{prefix}{k}"'
                else:
                    name = f'"{prefix}{distinct_elements_k}"'
                assert not (self.parent.is_colname_in(name)), NameError(
                    "A vDataColumn has already the alias of one of "
                    f"the dummies ({name}).\nIt can be the result "
                    "of using previously the method on the vDataColumn "
                    "or simply because of ambiguous columns naming."
                    "\nBy changing one of the parameters ('prefix', "
                    "'prefix_sep'), you'll be able to solve this "
                    "issue."
                )
            for k in range(len(distinct_elements) - n):
                distinct_elements_k = str(distinct_elements[k]).replace("'", "''")
                if use_numbers_as_suffix:
                    name = f'"{prefix}{k}"'
                else:
                    name = f'"{prefix}{distinct_elements_k}"'
                name = (
                    name.replace(" ", "_")
                    .replace("/", "_")
                    .replace(",", "_")
                    .replace("'", "_")
                )
                expr = f"DECODE({{}}, '{distinct_elements_k}', 1, 0)"
                transformations = self.transformations + [(expr, "bool", "int")]
                new_vDataColumn = vDataColumn(
                    name,
                    parent=self.parent,
                    transformations=transformations,
                    catalog={
                        "min": 0,
                        "max": 1,
                        "count": self.parent.shape()[0],
                        "percent": 100.0,
                        "unique": 2,
                        "approx_unique": 2,
                        "prod": 0,
                    },
                )
                setattr(self.parent, name, new_vDataColumn)
                setattr(self.parent, name.replace('"', ""), new_vDataColumn)
                self.parent._VERTICAPY_VARIABLES_["columns"] += [name]
                all_new_features += [name]
            conj = "s were " if len(all_new_features) > 1 else " was "
            self.parent.__add_to_history__(
                "[Get Dummies]: One hot encoder was applied to the vDataColumn "
                f"{self.alias}\n{len(all_new_features)} feature{conj}created: "
                f"{', '.join(all_new_features)}."
            )
        return self.parent

    get_dummies = one_hot_encode

    @save_verticapy_logs
    def label_encode(self):
        """
    Encodes the vDataColumn using a bijection from the different categories to
    [0, n - 1] (n being the vDataColumn cardinality).

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].decode       : Encodes the vDataColumn with a user defined Encoding.
    vDataFrame[].discretize   : Discretizes the vDataColumn.
    vDataFrame[].get_dummies  : Encodes the vDataColumn with One-Hot Encoding.
    vDataFrame[].mean_encode  : Encodes the vDataColumn using the mean encoding of a response.
        """
        if self.category() in ["date", "float"]:
            warning_message = (
                "label_encode is only available for categorical variables."
            )
            warnings.warn(warning_message, Warning)
        else:
            distinct_elements = self.distinct()
            expr = ["DECODE({}"]
            text_info = "\n"
            for k in range(len(distinct_elements)):
                distinct_elements_k = str(distinct_elements[k]).replace("'", "''")
                expr += [f"'{distinct_elements_k}', {k}"]
                text_info += f"\t{distinct_elements[k]} => {k}"
            expr = f"{', '.join(expr)}, {len(distinct_elements)})"
            self.transformations += [(expr, "int", "int")]
            self.parent.__update_catalog__(erase=True, columns=[self.alias])
            self.catalog["count"] = self.parent.shape()[0]
            self.catalog["percent"] = 100
            self.parent.__add_to_history__(
                "[Label Encoding]: Label Encoding was applied to the vDataColumn"
                f" {self.alias} using the following mapping:{text_info}"
            )
        return self.parent

    @save_verticapy_logs
    def mean_encode(self, response: str):
        """
    Encodes the vDataColumn using the average of the response partitioned by the 
    different vDataColumn categories.

    Parameters
    ----------
    response: str
        Response vDataColumn.

    Returns
    -------
    vDataFrame
        self.parent

    See Also
    --------
    vDataFrame[].decode       : Encodes the vDataColumn using a user-defined encoding.
    vDataFrame[].discretize   : Discretizes the vDataColumn.
    vDataFrame[].label_encode : Encodes the vDataColumn with Label Encoding.
    vDataFrame[].get_dummies  : Encodes the vDataColumn with One-Hot Encoding.
        """
        response = self.parent.format_colnames(response)
        assert self.parent[response].isnum(), TypeError(
            "The response column must be numerical to use a mean encoding"
        )
        max_floor = len(self.parent[response].transformations) - len(
            self.transformations
        )
        for k in range(max_floor):
            self.transformations += [("{}", self.ctype(), self.category())]
        self.transformations += [
            (f"AVG({response}) OVER (PARTITION BY {{}})", "int", "float",)
        ]
        self.parent.__update_catalog__(erase=True, columns=[self.alias])
        self.parent.__add_to_history__(
            f"[Mean Encode]: The vDataColumn {self.alias} was transformed "
            f"using a mean encoding with {response} as Response Column."
        )
        if OPTIONS["print_info"]:
            print("The mean encoding was successfully done.")
        return self.parent
