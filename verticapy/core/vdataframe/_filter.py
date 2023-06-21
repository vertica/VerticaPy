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
import copy
import random
import warnings
from typing import Literal, Optional, Union, TYPE_CHECKING
from collections.abc import Iterable

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import (
    NoneType,
    PythonNumber,
    PythonScalar,
    SQLColumns,
    SQLExpression,
    TimeInterval,
)
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, format_type, quote_ident
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.vdataframe._aggregate import vDFAgg, vDCAgg

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFFilter(vDFAgg):
    @save_verticapy_logs
    def at_time(self, ts: str, time: TimeInterval) -> "vDataFrame":
        """
        Filters the vDataFrame  by only keeping the records at the
        input time.

        Parameters
        ----------
        ts: str
            TS (Time Series) vDataColumn used to filter the data.
            The vDataColumn type must be date (date, datetime,
            timestamp...).
        time: TimeInterval
            Input Time. For example, time = '12:00' will filter the
            data when time('ts') is equal to 12:00.

        Returns
        -------
        vDataFrame
            self
        """
        self.filter(f"{self.format_colnames(ts)}::time = '{time}'")
        return self

    @save_verticapy_logs
    def balance(
        self,
        column: str,
        method: Literal["over", "under"] = "under",
        x: float = 0.5,
        order_by: Optional[SQLColumns] = None,
    ) -> "vDataFrame":
        """
        Balances the dataset using the input method.

        \u26A0 Warning : If the data is not sorted, the generated
                         SQL code may differ between attempts.

        Parameters
        ----------
        column: str
            Column used to compute the different categories.
        method: str, optional
            The method with which to sample the data:
                over   : oversampling
                under  : undersampling
        x: float, optional
            The desired ratio between the majority class and minority
            classes.
        order_by: SQLColumns, optional
            vDataColumns used to sort the data.

        Returns
        -------
        vDataFrame
            balanced vDataFrame
        """
        if not 0 <= x <= 1:
            raise ValueError("Parameter 'x' must be between 0 and 1")
        order_by = format_type(order_by, dtype=list)
        column, order_by = self.format_colnames(column, order_by)
        topk = self[column].topk()
        min_cnt = topk["count"][-1]
        min_class = topk["index"][-1]
        max_cnt = topk["count"][0]
        n = len(topk["index"])
        if method == "under":
            vdf = self.search(f"{column} = '{min_class}'")
            for i in range(n - 1):
                cnt = int(max(topk["count"][i] * (1.0 - x), min_cnt))
                vdf = vdf.append(
                    self.search(f"{column} = '{topk['index'][i]}'").sample(n=cnt)
                )
        elif method == "over":
            vdf = self.copy()
            for i in range(1, n):
                cnt_i, cnt = topk["count"][i], 0
                limit = int(max_cnt * x) - cnt_i
                while cnt <= limit:
                    vdf_i = self.search(f"{column} = '{topk['index'][i]}'")
                    if cnt + cnt_i > limit:
                        vdf = vdf.append(vdf_i.sample(n=limit - cnt))
                        break
                    else:
                        vdf = vdf.append(vdf_i)
                    cnt += cnt_i
        else:
            raise ValueError(f"Unrecognized method: '{method}'.")
        vdf.sort(order_by)
        return vdf

    @save_verticapy_logs
    def between(
        self,
        column: str,
        start: Optional[PythonScalar] = None,
        end: Optional[PythonScalar] = None,
        inplace: bool = True,
    ) -> "vDataFrame":
        """
        Filters the vDataFrame by only keeping the records between two
        input elements.

        Parameters
        ----------
        column: str
            TS (Time  Series)  vDataColumn  used to filter the  data.
            The vDataColumn  type  must be date (date,  datetime,
            timestamp...)
        start: PythonScalar, optional
            Input Python Scalar used to filter.
        end: PythonScalar, optional
            Input Python Scalar used to filter.
        inplace: bool, optional
            If  set  to  True, the  filtering  is applied  to  the
            vDataFrame.

        Returns
        -------
        vDataFrame
            self
        """
        if not isinstance(start, NoneType) and not isinstance(end, NoneType):
            condition = f"BETWEEN '{start}' AND '{end}'"
        elif not isinstance(start, NoneType):
            condition = f"> '{start}'"
        elif not isinstance(end, NoneType):
            condition = f"< '{end}'"
        else:
            return self.copy() if inplace else self
        filter_function = self.filter if inplace else self.search
        return filter_function(
            f"{self.format_colnames(column)} {condition}",
        )

    @save_verticapy_logs
    def between_time(
        self,
        ts: str,
        start_time: Optional[TimeInterval] = None,
        end_time: Optional[TimeInterval] = None,
        inplace: bool = True,
    ) -> "vDataFrame":
        """
        Filters the vDataFrame by only keeping the records between two
        input times.

        Parameters
        ----------
        ts: str
            TS   (Time Series) vDataColumn used to filter the  data.
            The  vDataColumn type must be date (date,  datetime,
            timestamp...).
        start_time: TimeInterval
            Input Start Time. For example, time = '12:00' will  filter
            the data when time('ts') is lesser than 12:00.
        end_time: TimeInterval
            Input  End Time. For  example, time = '14:00' will  filter
            the data when time('ts') is greater than 14:00.
        inplace: bool, optional
            If set to True, the filtering is applied to the vDataFrame.

        Returns
        -------
        vDataFrame
            self
        """
        if not isinstance(start_time, NoneType) and not (
            isinstance(end_time, NoneType)
        ):
            condition = f"BETWEEN '{start_time}' AND '{end_time}'"
        elif not isinstance(start_time, NoneType):
            condition = f"> '{start_time}'"
        elif not isinstance(end_time, NoneType):
            condition = f"< '{end_time}'"
        else:
            raise ValueError(
                "One of the parameters 'start_time' or 'end_time' must be defined."
            )
        filter_function = self.filter if inplace else self.search
        return filter_function(
            f"{self.format_colnames(ts)}::time {condition}",
        )

    @save_verticapy_logs
    def drop(self, columns: Optional[SQLColumns] = None) -> "vDataFrame":
        """
        Drops  the input vDataColumns  from the vDataFrame.  Dropping
        vDataColumns means they are not selected in the final SQL code
        generation.
        Be careful when using this method. It can make the vDataFrame
        structure  heavier if other  vDataColumns are  computed
        using the dropped vDataColumns.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names.

        Returns
        -------
        vDataFrame
            self
        """
        columns = format_type(columns, dtype=list)
        columns = self.format_colnames(columns)
        for column in columns:
            self[column].drop()
        return self

    @save_verticapy_logs
    def drop_duplicates(self, columns: Optional[SQLColumns] = None) -> "vDataFrame":
        """
        Filters the duplicates using a partition by the input
        vDataColumns.

        \u26A0 Warning : Dropping  duplicates  will make the  vDataFrame
                         structure heavier. It is recommended that you
                         check the   current   structure   using   the
                         'current_relation'  method and save it using
                         the 'to_db' method, using the parameters
                         'inplace = True' and 'relation_type = table'.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of  the vDataColumns names.  If empty,  all
            vDataColumns are selected.

        Returns
        -------
        vDataFrame
            self
        """
        columns = format_type(columns, dtype=list)
        count = self.duplicated(columns=columns, count=True)
        if count:
            columns = (
                self.get_columns() if not columns else self.format_colnames(columns)
            )
            name = (
                "__verticapy_duplicated_index__"
                + str(random.randint(0, 10000000))
                + "_"
            )
            self.eval(
                name=name,
                expr=f"""ROW_NUMBER() OVER (PARTITION BY {", ".join(columns)})""",
            )
            self.filter(f'"{name}" = 1')
            self._vars["exclude_columns"] += [f'"{name}"']
        elif conf.get_option("print_info"):
            print("No duplicates detected.")
        return self

    @save_verticapy_logs
    def dropna(self, columns: Optional[SQLColumns] = None) -> "vDataFrame":
        """
        Filters the specified vDataColumns in a vDataFrame for
        missing values.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of  the vDataColumns  names. If  empty,  all
            vDataColumns are selected.

        Returns
        -------
        vDataFrame
            self
        """
        columns = format_type(columns, dtype=list)
        columns = self.format_colnames(columns)
        if len(columns) == 0:
            columns = self.get_columns()
        total = self.shape()[0]
        print_info = conf.get_option("print_info")
        for column in columns:
            conf.set_option("print_info", False)
            self[column].dropna()
            conf.set_option("print_info", print_info)
        if conf.get_option("print_info"):
            total -= self.shape()[0]
            if total == 0:
                print("Nothing was filtered.")
            else:
                conj = "s were " if total > 1 else " was "
                print(f"{total} element{conj}filtered.")
        return self

    @save_verticapy_logs
    def filter(
        self, conditions: Union[None, list, str] = None, *args, **kwargs
    ) -> "vDataFrame":
        """
        Filters  the vDataFrame using the  input  expressions.

        Parameters
        ----------
        conditions: SQLExpression, optional
            List of expressions. For example, to keep only the
            records where the vDataColumn 'column' is greater
            than 5 and less than 10, you can write:
            ['"column" > 5', '"column" < 10'].

        Returns
        -------
        vDataFrame
            self
        """
        count = self.shape()[0]
        conj = "s were " if count > 1 else " was "
        if not isinstance(conditions, str) or (args):
            if isinstance(conditions, str) or not isinstance(conditions, Iterable):
                conditions = [conditions]
            elif isinstance(conditions, NoneType):
                conditions = []
            else:
                conditions = list(conditions)
            conditions += list(args)
            for condition in conditions:
                self.filter(str(condition), print_info=False)
            count -= self.shape()[0]
            if count > 0:
                if conf.get_option("print_info"):
                    print(f"{count} element{conj}filtered")
                self._add_to_history(
                    f"[Filter]: {count} element{conj}filtered "
                    f"using the filter '{conditions}'"
                )
            elif conf.get_option("print_info"):
                print("Nothing was filtered.")
        else:
            max_pos = 0
            columns_tmp = copy.deepcopy(self._vars["columns"])
            for column in columns_tmp:
                max_pos = max(max_pos, len(self[column]._transf) - 1)
            new_count = self.shape()[0]
            self._vars["where"] += [(conditions, max_pos)]
            try:
                new_count = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('vDataframe.filter')*/ 
                            COUNT(*) 
                        FROM {self}""",
                    title="Computing the new number of elements.",
                    method="fetchfirstelem",
                    sql_push_ext=self._vars["sql_push_ext"],
                    symbol=self._vars["symbol"],
                )
                count -= new_count
            except QueryError:
                del self._vars["where"][-1]
                if conf.get_option("print_info"):
                    warning_message = (
                        f"The expression '{conditions}' is incorrect.\n"
                        "Nothing was filtered."
                    )
                    warnings.warn(warning_message, Warning)
                return self
            if count > 0:
                self._update_catalog(erase=True)
                self._vars["count"] = new_count
                conj = "s were " if count > 1 else " was "
                if conf.get_option("print_info") and "print_info" not in kwargs:
                    print(f"{count} element{conj}filtered.")
                conditions_clean = clean_query(conditions)
                self._add_to_history(
                    f"[Filter]: {count} element{conj}filtered using "
                    f"the filter '{conditions_clean}'"
                )
            else:
                del self._vars["where"][-1]
                if conf.get_option("print_info") and "print_info" not in kwargs:
                    print("Nothing was filtered.")
        return self

    @save_verticapy_logs
    def first(self, ts: str, offset: str) -> "vDataFrame":
        """
        Filters the vDataFrame by only keeping the first records.

        Parameters
        ----------
        ts: str
            TS  (Time Series)  vDataColumn  used to filter the
            data. The vDataColumn  type  must be date (date,
            datetime, timestamp...)
        offset: str
            Interval offset. For example, to filter and keep only
            the first  6 months of records,  offset should be set
            to '6 months'.

        Returns
        -------
        vDataFrame
            self
        """
        ts = self.format_colnames(ts)
        first_date = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.first')*/ 
                    (MIN({ts}) + '{offset}'::interval)::varchar 
                FROM {self}""",
            title="Getting the vDataFrame first values.",
            method="fetchfirstelem",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        self.filter(f"{ts} <= '{first_date}'")
        return self

    @save_verticapy_logs
    def isin(self, val: dict) -> "vDataFrame":
        """
        Checks whether specific records are in the vDataFrame
        and returns the new vDataFrame of the search.

        Parameters
        ----------
        val: dict
            Dictionary of the different records. Each key of the
            dictionary must represent a vDataColumn. For example,
            to check  if Badr Ouali and Fouad Teban  are in  the
            vDataFrame. You can write the following dict:
            {"name": ["Teban", "Ouali"],
             "surname": ["Fouad", "Badr"]}

        Returns
        -------
        vDataFrame
            The vDataFrame of the search.
        """
        val = self.format_colnames(val)
        n = len(val[list(val.keys())[0]])
        result = []
        for i in range(n):
            tmp_query = []
            for column in val:
                if isinstance(val[column][i], NoneType):
                    tmp_query += [f"{quote_ident(column)} IS NULL"]
                else:
                    val_str = str(val[column][i]).replace("'", "''")
                    tmp_query += [f"{quote_ident(column)} = '{val_str}'"]
            result += [" AND ".join(tmp_query)]
        return self.search(" OR ".join(result))

    @save_verticapy_logs
    def last(self, ts: str, offset: str) -> "vDataFrame":
        """
        Filters the vDataFrame by only keeping the last records.

        Parameters
        ----------
        ts: str
            TS (Time Series)  vDataColumn used to filter the
            data. The vDataColumn type must be date (date,
            datetime, timestamp...)
        offset: str
            Interval  offset.  For example, to filter and  keep
            only the last 6 months of records, offset should be
            set to '6 months'.

        Returns
        -------
        vDataFrame
            self
        """
        ts = self.format_colnames(ts)
        last_date = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.last')*/ 
                    (MAX({ts}) - '{offset}'::interval)::varchar 
                FROM {self}""",
            title="Getting the vDataFrame last values.",
            method="fetchfirstelem",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        self.filter(f"{ts} >= '{last_date}'")
        return self

    @save_verticapy_logs
    def sample(
        self,
        n: Optional[PythonNumber] = None,
        x: Optional[float] = None,
        method: Literal["random", "systematic", "stratified"] = "random",
        by: Optional[SQLColumns] = None,
    ) -> "vDataFrame":
        """
        Downsamples the input vDataFrame.

        \u26A0 Warning : The result might be inconsistent between
                         attempts at SQL code generation if the
                         data is not ordered.

        Parameters
         ----------
         n: PythonNumber, optional
            Approximate  number of elements to consider in  the
            sample.
         x: float, optional
            The sample size. For example, if set to 0.33, it
            downsamples to approximatively 33% of the relation.
        method: str, optional
            The Sample method.
                random     : random sampling.
                systematic : systematic sampling.
                stratified : stratified sampling.
        by: SQLColumns, optional
            vDataColumns used in the partition.

        Returns
        -------
        vDataFrame
            sample vDataFrame
        """
        if x == 1:
            return self.copy()
        assert not isinstance(n, NoneType) or not (isinstance(x, NoneType)), ValueError(
            "One of the parameter 'n' or 'x' must not be empty."
        )
        assert isinstance(n, NoneType) or isinstance(x, NoneType), ValueError(
            "One of the parameter 'n' or 'x' must be empty."
        )
        if not isinstance(n, NoneType):
            x = float(n / self.shape()[0])
            if x >= 1:
                return self.copy()
        if isinstance(method, str):
            method = method.lower()
        if method in ("systematic", "random"):
            order_by = ""
            assert not by, ValueError(
                f"Parameter 'by' must be empty when using '{method}' sampling."
            )
        by = format_type(by, dtype=list)
        by = self.format_colnames(by)
        random_int = random.randint(0, 10000000)
        name = f"__verticapy_random_{random_int}__"
        name2 = f"__verticapy_random_{random_int + 1}__"
        vdf = self.copy()
        assert 0 < x < 1, ValueError("Parameter 'x' must be between 0 and 1")
        if method == "random":
            random_state = conf.get_option("random_state")
            random_seed = random.randint(-10e6, 10e6)
            if isinstance(random_state, int):
                random_seed = random_state
            random_func = f"SEEDED_RANDOM({random_seed})"
            vdf.eval(name, random_func)
            q = vdf[name].quantile(x)
            print_info_init = conf.get_option("print_info")
            conf.set_option("print_info", False)
            vdf.filter(f"{name} <= {q}")
            conf.set_option("print_info", print_info_init)
            vdf._vars["exclude_columns"] += [name]
        elif method in ("stratified", "systematic"):
            assert method != "stratified" or (by), ValueError(
                "Parameter 'by' must include at least one "
                "column when using 'stratified' sampling."
            )
            if method == "stratified":
                order_by = "ORDER BY " + ", ".join(by)
            vdf.eval(name, f"ROW_NUMBER() OVER({order_by})")
            vdf.eval(
                name2,
                f"""MIN({name}) OVER (PARTITION BY CAST({name} * {x} AS Integer) 
                    ORDER BY {name} ROWS BETWEEN UNBOUNDED PRECEDING AND 0 FOLLOWING)""",
            )
            print_info_init = conf.get_option("print_info")
            conf.set_option("print_info", False)
            vdf.filter(f"{name} = {name2}")
            conf.set_option("print_info", print_info_init)
            vdf._vars["exclude_columns"] += [name, name2]
        return vdf

    @save_verticapy_logs
    def search(
        self,
        conditions: SQLExpression = "",
        usecols: Optional[SQLColumns] = None,
        expr: Optional[SQLExpression] = None,
        order_by: Union[None, str, dict, list] = None,
    ) -> "vDataFrame":
        """
        Searches for elements that match the input
        conditions.

        Parameters
        ----------
        conditions: SQLExpression, optional
            Filters  of  the  search.  It can be a list  of
            conditions or an expression.
        usecols: SQLColumns, optional
            vDataColumns   to    select   from  the   final
            vDataFrame relation. If empty, all vDataColumns
            are selected.
        expr: SQLExpression, optional
            List  of  customized  expressions  in  pure SQL.
            For example: 'column1 * column2 AS my_name'.
        order_by: str / dict / list, optional
            List of the vDataColumns used to sort the data,
            using  asc order or a dictionary of all sorting
            methods.  For  example,  to  sort  by  "column1"
            ASC and "column2" DESC, write:
            {"column1": "asc",
             "column2": "desc"}

        Returns
        -------
        vDataFrame
            vDataFrame of the search
        """
        order_by, usecols, expr = format_type(order_by, usecols, expr, dtype=list)
        if isinstance(conditions, Iterable) and not isinstance(conditions, str):
            conditions = " AND ".join([f"({c})" for c in conditions])
        if conditions:
            conditions = f" WHERE {conditions}"
        all_cols = ", ".join(["*"] + expr)
        query = f"SELECT {all_cols} FROM {self}{conditions}"
        result = create_new_vdf(query)
        if usecols:
            result = result.select(usecols)
        return result.sort(order_by)


class vDCFilter(vDCAgg):
    @save_verticapy_logs
    def drop(self, add_history: bool = True) -> "vDataFrame":
        """
        Drops the  vDataColumn from the vDataFrame. Dropping a
        vDataColumn means it is not selected in the final
        generated SQL code.

        Note:  Dropping a vDataColumn  can make the vDataFrame
        "heavier" if it is  used to compute other vDataColumns.

        Parameters
        ----------
        add_history: bool, optional
            If set to True,  the information is stored in
            the vDataFrame history.

        Returns
        -------
        vDataFrame
            self._parent
        """
        try:
            parent = self._parent
            force_columns = copy.deepcopy(self._parent._vars["columns"])
            force_columns.remove(self._alias)
            _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('vDataColumn.drop')*/ * 
                    FROM {self._parent._genSQL(force_columns=force_columns)} 
                    LIMIT 10""",
                print_time_sql=False,
                sql_push_ext=self._parent._vars["sql_push_ext"],
                symbol=self._parent._vars["symbol"],
            )
            self._parent._vars["columns"].remove(self._alias)
            delattr(self._parent, self._alias)
        except QueryError:
            self._parent._vars["exclude_columns"] += [self._alias]
        if add_history:
            self._parent._add_to_history(
                f"[Drop]: vDataColumn {self} was deleted from the vDataFrame."
            )
        return parent

    @save_verticapy_logs
    def drop_outliers(
        self,
        threshold: PythonNumber = 4.0,
        use_threshold: bool = True,
        alpha: PythonNumber = 0.05,
    ) -> "vDataFrame":
        """
        Drops outliers in the vDataColumn.

        Parameters
        ----------
        threshold: PythonNumber, optional
            Uses the  Gaussian distribution  to identify outliers.
            After normalizing the data (Z-Score), if the absolute
            value of the record is greater than the threshold, it
            is considered as an outlier.
        use_threshold: bool, optional
            Uses  the threshold instead of the  'alpha' parameter.
        alpha: PythonNumber, optional
            Number  representing  the outliers threshold.  Values
            less   than   quantile(alpha)   or   greater   than
            quantile(1-alpha) are be dropped.

        Returns
        -------
        vDataFrame
            self._parent
        """
        if use_threshold:
            result = self.aggregate(func=["std", "avg"]).transpose().values
            self._parent.filter(
                f"""
                    ABS({self} - {result["avg"][0]}) 
                  / {result["std"][0]} < {threshold}"""
            )
        else:
            p_alpha, p_1_alpha = (
                self._parent.quantile([alpha, 1 - alpha], [self._alias])
                .transpose()
                .values[self._alias]
            )
            self._parent.filter(f"({self} BETWEEN {p_alpha} AND {p_1_alpha})")
        return self._parent

    @save_verticapy_logs
    def dropna(self) -> "vDataFrame":
        """
        Filters the vDataFrame where the vDataColumn is missing.

        Returns
        -------
        vDataFrame
            self._parent
        """
        self._parent.filter(f"{self} IS NOT NULL")
        return self._parent

    @save_verticapy_logs
    def isin(
        self,
        val: Union[PythonScalar, list],
        *args,
    ) -> "vDataFrame":
        """
        Checks whether specific records are in the vDataColumn and
        returns the new vDataFrame of the search.

        Parameters
        ----------
        val: PythonScalar / list
            List of the different  records. For example, to check if
            Badr and Fouad are in the vDataColumn, you can write the
            following list: ["Fouad", "Badr"]

        Returns
        -------
        vDataFrame
            The vDataFrame of the search.
        """
        if isinstance(val, str) or not isinstance(val, Iterable):
            val = [val]
        val += list(args)
        val = {self._alias: val}
        return self._parent.isin(val)
