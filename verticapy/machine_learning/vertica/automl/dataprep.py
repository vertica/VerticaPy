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
import copy
import datetime
from typing import Literal, Optional

import verticapy._config.config as conf
from verticapy._typing import NoneType, TimeInterval, SQLColumns, SQLRelation
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type
from verticapy._utils._sql._sys import _executeSQL

from verticapy.sql.drop import drop


from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.base import VerticaModel
from verticapy.machine_learning.vertica.decomposition import PCA


class AutoDataPrep(VerticaModel):
    """
    Automatically find relations between the different
    features  to preprocess the data according to each
    column type.

    Parameters
    ----------
    name: str, optional
        Name of the model in which to store the output
        relation in the Vertica database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    cat_method: str, optional
        Method for encoding categorical features. This
        can  be set to 'label' for label encoding  and
        'ooe' for One-Hot Encoding.
    num_method: str, optional
        [Only used for non-time series datasets]
        Method  for  encoding numerical features.  This
        can  be  set to  'same_freq'  to  encode  using
        frequencies,   'same_width'   to  encode  using
        regular bins, or 'none' to not encode numerical
        features.
    nbins: int, optional
        [Only used for non-time series datasets]
        Number  of bins used  to  discretize  numerical
        features.
    outliers_threshold: float, optional
        [Only used for non-time series datasets]
        Method for dealing with outliers. If a number
        is used, all  elements with an absolute z-score
        greater than  the threshold  are converted
        to  NULL values.  Otherwise,  outliers  are
        treated  as regular values.
    na_method: str, optional
        Method for handling missing values.
            auto: Mean  for the numerical features  and
                  creates   a  new  category  for   the
                  categorical  vDataColumns.  For  time
                  series      datasets,      'constant'
                  interpolation is used for categorical
                  features and 'linear' for the others.
            drop: Drops the missing values.
    cat_topk: int, optional
        Keeps  the top-k  most frequent categories  and
        merges the others  into one unique category. If
        unspecified, all categories are kept.
    standardize: bool, optional
        If True, the data is standardized. The 'num_method'
        parameter must be set to 'none'.
    standardize_min_cat: int, optional
        Minimum   feature   cardinality  before   using
        standardization.
    id_method: str, optional
        Method for handling ID features.
            drop: Drops any feature detected as ID.
            none: Does not change ID features.
    apply_pca: bool, optional
        [Only used for non-time series datasets]
        If  True, a PCA  is  applied at the end of  the
        preprocessing.
    rule: TimeInterval, optional
        [Only used for time series datasets]
        Interval used to slice  the time. For example,
        setting to '5 minutes' creates records separated
        by '5 minutes' time interval. If set to auto,
        the rule is detected using aggregations.
    identify_ts: bool, optional
        If  True  and parameter 'ts' is  undefined  when
        fitting  the  model,  the  function tries to
        automatically detect the parameter 'ts'.
    save: bool, optional
        If  True,  saves  the final relation inside  the
        database.

    Attributes
    ----------
    X_in_: list
        Variables used to fit the model.
    X_out_: list
        Variables created by the model.
    sql_: str
        SQL needed to deploy the model.
    final_relation_: vDataFrame
        Relation created after fitting the model.
    """

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _vertica_fit_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_predict_sql(self) -> Literal[""]:
        return ""

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["PREPROCESSING"]:
        return "PREPROCESSING"

    @property
    def _model_type(self) -> Literal["AutoDataPrep"]:
        return "AutoDataPrep"

    @property
    def _attributes(self) -> list[str]:
        return ["X_in_", "X_out_", "sql_", "final_relation_"]

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self,
        name: Optional[str] = None,
        overwrite_model: Optional[bool] = False,
        cat_method: Literal["label", "ooe"] = "ooe",
        num_method: Literal["same_freq", "same_width", "none"] = "none",
        nbins: int = 20,
        outliers_threshold: float = 4.0,
        na_method: Literal["auto", "drop"] = "auto",
        cat_topk: int = 10,
        standardize: bool = True,
        standardize_min_cat: int = 6,
        id_method: Literal["none", "drop"] = "drop",
        apply_pca: bool = False,
        rule: TimeInterval = "auto",
        identify_ts: bool = True,
        save: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "cat_method": cat_method,
            "num_method": num_method,
            "nbins": nbins,
            "outliers_threshold": outliers_threshold,
            "na_method": na_method,
            "cat_topk": cat_topk,
            "rule": rule,
            "standardize": standardize,
            "standardize_min_cat": standardize_min_cat,
            "apply_pca": apply_pca,
            "id_method": id_method,
            "identify_ts": identify_ts,
            "save": save,
        }

    def drop(self) -> bool:
        """
        Drops the model from the Vertica database.
        """
        # it could be stored as a model or a table
        dropped_model = super().drop()
        dropped_table = drop(self.model_name, method="table")
        return dropped_model or dropped_table

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        X: Optional[SQLColumns] = None,
        ts: Optional[str] = None,
        by: Optional[SQLColumns] = None,
        return_report: bool = False,
    ) -> None:
        """
        Trains the model.

        Parameters
        ----------
        input_relation: SQLRelation
            Training Relation.
        X: SQLColumns, optional
            List of the features to preprocess.
        ts: str, optional
            Time series  vDataColumn used to order the
            data.
            The vDataColumn type must be date-like (date,
            datetime, timestamp...).
        by: SQLColumns, optional
            vDataColumns used in the partition.
        """
        if self.overwrite_model:
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        current_print_info = conf.get_option("print_info")
        conf.set_option("print_info", False)
        if by and not ts:
            raise ValueError("Parameter 'by' must be empty if 'ts' is not defined.")
        if isinstance(input_relation, str):
            vdf = vDataFrame(input_relation)
        else:
            vdf = input_relation.copy()
        if isinstance(X, NoneType):
            X = vdf.get_columns()
        X, by = format_type(X, by, dtype=list)
        if not ts and self.parameters["identify_ts"]:
            nb_date, nb_num, nb_others = 0, 0, 0
            for x in X:
                if vdf[x].isnum() and not vdf[x].isbool():
                    nb_num += 1
                elif vdf[x].isdate():
                    nb_date += 1
                    ts_tmp = x
                else:
                    nb_others += 1
                    cat_tmp = x
            if nb_date == 1 and nb_others <= 1:
                ts = ts_tmp
            if nb_date == 1 and nb_others == 1:
                by = [cat_tmp]
        X, ts, by = vdf.format_colnames(X, ts, by)
        X_diff = vdf.get_columns(exclude_columns=X)
        columns_to_drop = []
        n = vdf.shape()[0]
        for x in X:
            is_id = (
                not vdf[x].isnum()
                and not vdf[x].isdate()
                and 0.9 * n <= vdf[x].nunique()
            )
            if (
                self.parameters["id_method"] == "drop"
                and is_id
                and (not by or x not in by)
            ):
                columns_to_drop += [x]
                X_diff += [x]
            elif not is_id and (not by or x not in by):
                if not vdf[x].isdate():
                    if vdf[x].isnum():
                        if (self.parameters["outliers_threshold"]) and self.parameters[
                            "outliers_threshold"
                        ] > 0:
                            vdf[x].fill_outliers(
                                method="null",
                                threshold=self.parameters["outliers_threshold"],
                            )
                        if (
                            self.parameters["num_method"] == "none"
                            and (self.parameters["standardize"])
                            and (
                                self.parameters["standardize_min_cat"] < 2
                                or (
                                    vdf[x].nunique()
                                    > self.parameters["standardize_min_cat"]
                                )
                            )
                        ):
                            vdf[x].scale(method="zscore")
                        if self.parameters["na_method"] == "auto":
                            vdf[x].fillna(method="mean")
                        else:
                            vdf[x].dropna()
                    if (
                        vdf[x].isnum()
                        and not ts
                        and self.parameters["num_method"] in ("same_width", "same_freq")
                    ):
                        vdf[x].discretize(
                            method=self.parameters["num_method"],
                            nbins=self.parameters["nbins"],
                        )
                    elif vdf[x].nunique() > self.parameters["cat_topk"] and not (
                        vdf[x].isnum()
                    ):
                        if self.parameters["na_method"] == "auto":
                            vdf[x].fillna("NULL")
                        else:
                            vdf[x].dropna()
                        vdf[x].discretize(method="topk", k=self.parameters["cat_topk"])
                    if (
                        self.parameters["cat_method"] == "ooe" and not vdf[x].isnum()
                    ) or (
                        vdf[x].isnum()
                        and not ts
                        and self.parameters["num_method"] in ("same_width", "same_freq")
                    ):
                        vdf[x].one_hot_encode(drop_first=False)
                        columns_to_drop += [x]
                    elif (
                        self.parameters["cat_method"] == "label" and not vdf[x].isnum()
                    ) or (
                        vdf[x].isnum()
                        and not ts
                        and self.parameters["num_method"] in ("same_width", "same_freq")
                    ):
                        vdf[x].label_encode()
                elif not ts:
                    vdf[x.replace('"', "") + "_year"] = f"YEAR({x})"
                    vdf[x.replace('"', "") + "_dayofweek"] = f"DAYOFWEEK({x})"
                    vdf[x.replace('"', "") + "_month"] = f"MONTH({x})"
                    vdf[x.replace('"', "") + "_hour"] = f"HOUR({x})"
                    vdf[x.replace('"', "") + "_quarter"] = f"QUARTER({x})"
                    vdf[
                        x.replace('"', "") + "_trend"
                    ] = f"({x}::timestamp - MIN({x}::timestamp) OVER ()) / '1 second'::interval"
                    columns_to_drop += [x]
        if columns_to_drop:
            vdf.drop(columns_to_drop)
        if ts:
            if self.parameters["rule"] == "auto":
                vdf_tmp = vdf[[ts] + by]
                if by:
                    by_tmp = f"PARTITION BY {', '.join(by)} "
                else:
                    by_tmp = ""
                vdf_tmp[
                    "verticapy_time_delta"
                ] = f"""
                    ({ts}::timestamp 
                  - (LAG({ts}) OVER ({by_tmp}ORDER BY {ts}))::timestamp) 
                  / '00:00:01'"""
                vdf_tmp = vdf_tmp.groupby(["verticapy_time_delta"], ["COUNT(*) AS cnt"])
                rule = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('learn.delphi.AutoDataPrep.fit')*/
                            verticapy_time_delta 
                        FROM {vdf_tmp} 
                        ORDER BY cnt DESC 
                        LIMIT 1""",
                    method="fetchfirstelem",
                    print_time_sql=False,
                )
                rule = datetime.timedelta(seconds=rule)
            method = {}
            for elem in X:
                if elem != ts and elem not in by:
                    if vdf[elem].isnum() and not vdf[elem].isbool():
                        method[elem] = "linear"
                    else:
                        method[elem] = "ffill"
            vdf = vdf.interpolate(ts=ts, rule=rule, method=method, by=by)
            vdf.dropna()
        self.X_in_ = copy.deepcopy(X)
        self.X_out_ = vdf.get_columns(
            exclude_columns=by + [ts] + X_diff if ts else by + X_diff
        )
        self.by = by
        self.ts = ts
        if self.parameters["apply_pca"] and not ts:
            model_pca = PCA(self.model_name + "_pca")
            model_pca.drop()
            model_pca.fit(
                vdf,
                self.X_out_,
                return_report=True,
            )
            vdf = model_pca.transform()
            self.X_out_ = vdf.get_columns(
                exclude_columns=by + [ts] + X_diff if ts else by + X_diff
            )
        self.sql_ = vdf.current_relation()
        if self.parameters["save"]:
            vdf.to_db(name=self.model_name, relation_type="table", inplace=True)
        self.final_relation_ = vdf
        conf.set_option("print_info", current_print_info)
