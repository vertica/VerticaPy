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
import random, warnings, datetime
from typing import Union, Literal

# Other modules
import numpy as np
import scipy.stats as scipy_st

# VerticaPy Modules
from verticapy.core.tablesample import tablesample
from verticapy.learn.memmodel import memModel
from verticapy._utils._collect import save_verticapy_logs
from verticapy.errors import ParameterError
from verticapy.sql.read import vDataFrameSQL
from verticapy._utils._sql import _executeSQL
from verticapy.sql._utils._format import quote_ident
from verticapy._config.config import OPTIONS


class vDFML:
    @save_verticapy_logs
    def chaid(
        self,
        response: str,
        columns: Union[str, list],
        nbins: int = 4,
        method: Literal["smart", "same_width"] = "same_width",
        RFmodel_params: dict = {},
        **kwds,
    ):
        """
    Returns a CHAID (Chi-square Automatic Interaction Detector) tree.
    CHAID is a decision tree technique based on adjusted significance testing 
    (Bonferroni test).

    Parameters
    ----------
    response: str
        Categorical response vColumn.
    columns: str / list
        List of the vColumn names. The maximum number of categories for each
        categorical column is 16; categorical columns with a higher cardinality
        are discarded.
    nbins: int, optional
        Integer in the range [2,16], the number of bins used 
        to discretize the numerical features.
    method: str, optional
        The method with which to discretize the numerical vColumns, 
        one of the following:
            same_width : Computes bins of regular width.
            smart      : Uses a random forest model on a response column to find the best
                interval for discretization.
    RFmodel_params: dict, optional
        Dictionary of the parameters of the random forest model used to compute the best splits 
        when 'method' is 'smart'. If the response column is numerical (but not of type int or bool), 
        this function trains and uses a random forest regressor. Otherwise, this function 
        trains a random forest classifier.
        For example, to train a random forest with 20 trees and a maximum depth of 10, use:
            {"n_estimators": 20, "max_depth": 10}

    Returns
    -------
    memModel
        An independent model containing the result. For more information, see
        learn.memmodel.
        """
        from verticapy.machine_learning._utils import get_match_index

        if "process" not in kwds or kwds["process"]:
            if isinstance(columns, str):
                columns = [columns]
            assert 2 <= nbins <= 16, ParameterError(
                "Parameter 'nbins' must be between 2 and 16, inclusive."
            )
            columns = self.chaid_columns(columns)
            if not (columns):
                raise ValueError("No column to process.")
        idx = 0 if ("node_id" not in kwds) else kwds["node_id"]
        p = self.pivot_table_chi2(response, columns, nbins, method, RFmodel_params)
        categories, split_predictor, is_numerical, chi2 = (
            p["categories"][0],
            p["index"][0],
            p["is_numerical"][0],
            p["chi2"][0],
        )
        split_predictor_idx = get_match_index(
            split_predictor,
            columns
            if "process" not in kwds or kwds["process"]
            else kwds["columns_init"],
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
        if "process" not in kwds or kwds["process"]:
            classes = self[response].distinct()
        else:
            classes = kwds["classes"]
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
                             FROM {self.__genSQL__()} 
                             WHERE {split_predictor} IS NOT NULL 
                               AND {response} IS NOT NULL 
                             GROUP BY 1, 2) x 
                        ORDER BY 1;""",
                    title="Computing the CHAID tree probability.",
                    method="fetchall",
                    sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                    symbol=self._VERTICAPY_VARIABLES_["symbol"],
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
            if "process" not in kwds or kwds["process"]:
                return memModel("CHAID", attributes={"tree": tree, "classes": classes})
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
            if "process" not in kwds or kwds["process"]:
                return memModel("CHAID", attributes={"tree": tree, "classes": classes})
            return tree, idx

    @save_verticapy_logs
    def chaid_columns(self, columns: list = [], max_cardinality: int = 16):
        """
    Function used to simplify the code. It returns the columns picked by
    the CHAID algorithm.

    Parameters
    ----------
    columns: list
        List of the vColumn names.
    max_cardinality: int, optional
        The maximum number of categories for each categorical column. Categorical 
        columns with a higher cardinality are discarded.

    Returns
    -------
    list
        columns picked by the CHAID algorithm
        """
        columns_tmp = columns.copy()
        if not (columns_tmp):
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
                            f"vColumn '{col}' is of category '{self[col].category()}'. "
                            "This method only accepts categorical & numerical inputs. "
                            "This vColumn was ignored."
                        )
                    else:
                        warning_message = (
                            f"vColumn '{col}' has a too high cardinality "
                            f"(> {max_cardinality}). This vColumn was ignored."
                        )
                    warnings.warn(warning_message, Warning)
        for col in remove_cols:
            columns_tmp.remove(col)
        return columns_tmp

    @save_verticapy_logs
    def outliers(
        self,
        columns: Union[str, list] = [],
        name: str = "distribution_outliers",
        threshold: float = 3.0,
        robust: bool = False,
    ):
        """
    Adds a new vColumn labeled with 0 and 1. 1 means that the record is a global 
    outlier.

    Parameters
    ----------
    columns: str / list, optional
        List of the vColumns names. If empty, all numerical vColumns will be 
        used.
    name: str, optional
        Name of the new vColumn.
    threshold: float, optional
        Threshold equals to the critical score.
    robust: bool
        If set to True, the score used will be the Robust Z-Score instead of 
        the Z-Score.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.normalize : Normalizes the input vColumns.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = self.format_colnames(columns) if (columns) else self.numcol()
        if not (robust):
            result = self.aggregate(func=["std", "avg"], columns=columns).values
        else:
            result = self.aggregate(
                func=["mad", "approx_median"], columns=columns
            ).values
        conditions = []
        for idx, col in enumerate(result["index"]):
            if not (robust):
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
        columns: Union[str, list] = [],
        nbins: int = 16,
        method: Literal["smart", "same_width"] = "same_width",
        RFmodel_params: dict = {},
    ):
        """
    Returns the chi-square term using the pivot table of the response vColumn 
    against the input vColumns.

    Parameters
    ----------
    response: str
        Categorical response vColumn.
    columns: str / list, optional
        List of the vColumn names. The maximum number of categories for each
        categorical columns is 16. Categorical columns with a higher cardinality
        are discarded.
    nbins: int, optional
        Integer in the range [2,16], the number of bins used to discretize 
        the numerical features.
    method: str, optional
        The method to use to discretize the numerical vColumns.
            same_width : Computes bins of regular width.
            smart      : Uses a random forest model on a response column to find the best
                interval for discretization.
    RFmodel_params: dict, optional
        Dictionary of the parameters of the random forest model used to compute the best splits 
        when 'method' is 'smart'. If the response column is numerical (but not of type int or bool), 
        this function trains and uses a random forest regressor.  Otherwise, this function 
        trains a random forest classifier.
        For example, to train a random forest with 20 trees and a maximum depth of 10, use:
            {"n_estimators": 20, "max_depth": 10}

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns, response = self.format_colnames(columns, response)
        assert 2 <= nbins <= 16, ParameterError(
            "Parameter 'nbins' must be between 2 and 16, inclusive."
        )
        columns = self.chaid_columns(columns)
        for col in columns:
            if quote_ident(response) == quote_ident(col):
                columns.remove(col)
                break
        if not (columns):
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
            tmp_res = vdf.pivot_table(
                columns=[col, response], max_cardinality=(10000, 100), show=False
            ).to_numpy()[:, 1:]
            tmp_res = np.where(tmp_res == "", "0", tmp_res)
            tmp_res = tmp_res.astype(float)
            i = 0
            all_chi2 = []
            for row in tmp_res:
                j = 0
                for col_in_row in row:
                    all_chi2 += [
                        col_in_row ** 2 / (sum(tmp_res[i]) * sum(tmp_res[:, j]))
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
        return tablesample(result)

    @save_verticapy_logs
    def recommend(
        self,
        unique_id: str,
        item_id: str,
        method: Literal["count", "avg", "median"] = "count",
        rating: Union[str, tuple] = "",
        ts: str = "",
        start_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
        end_date: Union[str, int, float, datetime.datetime, datetime.date] = "",
    ):
        """
    Recommend items based on the Collaborative Filtering (CF) technique.
    The implementation is the same as APRIORI algorithm, but is limited to pairs 
    of items.

    Parameters
    ----------
    unique_id: str
        Input vColumn corresponding to a unique ID. It is a primary key.
    item_id: str
        Input vColumn corresponding to an item ID. It is a secondary key used to 
        compute the different pairs.
    method: str, optional
        Method used to recommend.
            count  : Each item will be recommended based on frequencies of the
                     different pairs of items.
            avg    : Each item will be recommended based on the average rating
                     of the different item pairs with a differing second element.
            median : Each item will be recommended based on the median rating
                     of the different item pairs with a differing second element.
    rating: str / tuple, optional
        Input vColumn including the items rating.
        If the 'rating' type is 'tuple', it must composed of 3 elements: 
        (r_vdf, r_item_id, r_name) where:
            r_vdf is an input vDataFrame.
            r_item_id is an input vColumn which must includes the same id as 'item_id'.
            r_name is an input vColumn including the items rating. 
    ts: str, optional
        TS (Time Series) vColumn to use to order the data. The vColumn type must be
        date like (date, datetime, timestamp...) or numerical.
    start_date: str / int / float / date, optional
        Input Start Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is lesser than November 1993 the 3rd.
    end_date: str / int / float / date, optional
        Input End Date. For example, time = '03-11-1993' will filter the data when 
        'ts' is greater than November 1993 the 3rd.

    Returns
    -------
    vDataFrame
        The vDataFrame of the recommendation.
        """
        unique_id, item_id, ts = self.format_colnames(unique_id, item_id, ts)
        vdf = self.copy()
        assert (
            method == "count" or rating
        ), f"Method '{method}' can not be used if parameter 'rating' is empty."
        if rating:
            assert isinstance(rating, str) or len(rating) == 3, ParameterError(
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
        method: str,  # TODO Literal[tuple(FUNCTIONS_DICTIONNARY)]
        nbins: int = 30,
    ):
        """
    Computes the score using the input columns and the input method.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    method: str
        The method to use to compute the score.
            --- For Classification ---
            accuracy    : Accuracy
            auc         : Area Under the Curve (ROC)
            best_cutoff : Cutoff which optimised the ROC Curve prediction.
            bm          : Informedness = tpr + tnr - 1
            csi         : Critical Success Index = tp / (tp + fn + fp)
            f1          : F1 Score 
            logloss     : Log Loss
            mcc         : Matthews Correlation Coefficient 
            mk          : Markedness = ppv + npv - 1
            npv         : Negative Predictive Value = tn / (tn + fn)
            prc_auc     : Area Under the Curve (PRC)
            precision   : Precision = tp / (tp + fp)
            recall      : Recall = tp / (tp + fn)
            specificity : Specificity = tn / (tn + fp)
            --- For Regression ---
            max    : Max Error
            mae    : Mean Absolute Error
            median : Median Absolute Error
            mse    : Mean Squared Error
            msle   : Mean Squared Log Error
            r2     : R squared coefficient
            var    : Explained Variance  
            --- Plots ---
            roc  : ROC Curve
            prc  : PRC Curve
            lift : Lift Chart
    nbins: int, optional
        Number of bins used to compute some of the metrics (AUC, PRC AUC...)

    Returns
    -------
    float / tablesample
        score / tablesample of the curve

    See Also
    --------
    vDataFrame.aggregate : Computes the vDataFrame input aggregations.
        """
        from verticapy.machine_learning.metrics import FUNCTIONS_DICTIONNARY

        y_true, y_score = self.format_colnames(y_true, y_score)
        fun = FUNCTIONS_DICTIONNARY[method]
        argv = [y_true, y_score, self.__genSQL__()]
        kwds = {}
        if method in ("accuracy", "acc"):
            kwds["pos_label"] = None
        elif method in ("best_cutoff", "best_threshold"):
            kwds["nbins"] = nbins
            kwds["best_threshold"] = True
        elif method in ("roc_curve", "roc", "prc_curve", "prc", "lift_chart", "lift"):
            kwds["nbins"] = nbins
        return FUNCTIONS_DICTIONNARY[method](*argv, **kwds)

    @save_verticapy_logs
    def train_test_split(
        self,
        test_size: float = 0.33,
        order_by: Union[str, list, dict] = {},
        random_state: int = None,
    ):
        """
    Creates 2 vDataFrame (train/test) which can be to use to evaluate a model.
    The intersection between the train and the test is empty only if a unique
    order is specified.

    Parameters
    ----------
    test_size: float, optional
        Proportion of the test set comparint to the training set.
    order_by: str / dict / list, optional
        List of the vColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
        Without this parameter, the seeded random number used to split the data
        into train and test can not garanty that no collision occurs. Use this
        parameter to avoid collisions.
    random_state: int, optional
        Integer used to seed the randomness.

    Returns
    -------
    tuple
        (train vDataFrame, test vDataFrame)
        """
        if isinstance(order_by, str):
            order_by = [order_by]
        order_by = self.__get_sort_syntax__(order_by)
        if not random_state:
            random_state = OPTIONS["random_state"]
        random_seed = (
            random_state
            if isinstance(random_state, int)
            else random.randint(-10e6, 10e6)
        )
        random_func = f"SEEDED_RANDOM({random_seed})"
        q = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.train_test_split')*/ 
                    APPROXIMATE_PERCENTILE({random_func} 
                        USING PARAMETERS percentile = {test_size}) 
                FROM {self.__genSQL__()}""",
            title="Computing the seeded numbers quantile.",
            method="fetchfirstelem",
        )
        test_table = f"""
            (SELECT * 
             FROM {self.__genSQL__()} 
             WHERE {random_func} < {q}{order_by}) x"""
        train_table = f"""
            (SELECT * 
             FROM {self.__genSQL__()} 
             WHERE {random_func} > {q}{order_by}) x"""
        return (
            vDataFrameSQL(relation=train_table),
            vDataFrameSQL(relation=test_table),
        )
