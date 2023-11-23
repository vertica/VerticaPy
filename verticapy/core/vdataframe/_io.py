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
import decimal
import pickle
import os
from typing import Literal, Optional, Union, TYPE_CHECKING
from collections.abc import Iterable

import numpy as np
import pandas as pd

import verticapy._config.config as conf
from verticapy._typing import NoneType, SQLColumns, SQLExpression
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type, quote_ident
from verticapy._utils._sql._random import _current_random
from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor
from verticapy.errors import ParsingError

from verticapy.core.tablesample.base import TableSample

from verticapy.core.vdataframe._sys import vDFSystem

if conf.get_import_success("geopandas"):
    from geopandas import GeoDataFrame
    from shapely import wkt

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

pickle.DEFAULT_PROTOCOL = 4


class vDFInOut(vDFSystem):
    def copy(self) -> "vDataFrame":
        """
        Returns a deep copy of the vDataFrame.

        Returns
        -------
        vDataFrame
            The copy of the vDataFrame.
        """
        return copy.deepcopy(self)

    @save_verticapy_logs
    def load(self, offset: int = -1) -> "vDataFrame":
        """
        Loads a previous structure of the vDataFrame.

        Parameters
        ----------
        offset: int, optional
            Offset of the saving. For example, setting to
            -1 loads the last saving.

        Returns
        -------
        vDataFrame
            vDataFrame of the loading.
        """
        save = self._vars["saving"][offset]
        vdf = pickle.loads(save)
        return vdf

    @save_verticapy_logs
    def save(self) -> "vDataFrame":
        """
        Saves the current structure of the vDataFrame.
        This function is useful for loading previous transformations.

        Returns
        -------
        vDataFrame
            self
        """
        vdf = self.copy()
        self._vars["saving"] += [pickle.dumps(vdf)]
        return self

    @save_verticapy_logs
    def to_csv(
        self,
        path: Optional[str] = None,
        sep: str = ",",
        na_rep: Optional[str] = None,
        quotechar: str = '"',
        usecols: Optional[SQLColumns] = None,
        header: bool = True,
        new_header: Optional[list] = None,
        order_by: Union[None, SQLColumns, dict] = None,
        n_files: int = 1,
    ) -> Union[None, str, list[str]]:
        """
        Creates  a CSV  file  or  folder of CSV files of  the  current
        vDataFrame relation.

        Parameters
        ----------
        path: str, optional
            File/Folder system path.  Be  careful:  if a CSV file with
            the same name exists, it will be overwritten.
        sep: str, optional
            Column separator.
        na_rep: str, optional
            Missing values representation.
        quotechar: str, optional
            Char that will enclose the str values.
        usecols: SQLColumns, optional
            vDataColumns to select  from the final vDataFrame relation.
            If empty, all vDataColumns are selected.
        header: bool, optional
            If set to False, no header is written in the CSV file.
        new_header: list, optional
            List of columns used to replace vDataColumns name in the
            CSV.
        order_by: SQLColumns / dict, optional
            List of the vDataColumns used to sort  the data, using asc
            order or a dictionary of all sorting methods. For example,
            to sort by "column1" ASC and "column2" DESC, write:
            {"column1": "asc", "column2": "desc"}
        n_files: int, optional
            Integer  greater than or equal to 1,  the number of CSV files
            to generate.  If n_files is greater than 1, you must also set
            order_by to sort the data,  ideally with a column with unique
            values (e.g. ID).
            Greater values of n_files decrease memory usage, but increase
            execution time.

        Returns
        -------
        str or list
            JSON str or list (n_files>1) if 'path' is not defined;
            otherwise, nothing.
        """
        order_by, usecols, new_header = format_type(
            order_by, usecols, new_header, dtype=list
        )
        if n_files < 1:
            raise ValueError("Parameter 'n_files' must be greater or equal to 1.")
        if (n_files != 1) and not order_by:
            raise ValueError(
                "If you want to store the vDataFrame in many CSV files, "
                "you have to sort your data by using at least one column. "
                "If the column hasn't unique values, the final result can "
                "not be guaranteed."
            )
        columns = self.get_columns() if not usecols else quote_ident(usecols)
        for col in columns:
            if self[col].category() in ("vmap", "complex"):
                raise TypeError(
                    f"Impossible to export virtual column {col} as"
                    " it includes complex data types or vmaps. "
                    "Use 'astype' method to cast them before using "
                    "this function."
                )
        if (new_header) and len(new_header) != len(columns):
            raise ParsingError("The header has an incorrect number of columns")
        total = self.shape()[0]
        current_nb_rows_written, file_id = 0, 0
        limit = int(total / n_files) + 1
        order_by = self._get_sort_syntax(order_by)
        if not order_by:
            order_by = self._get_last_order_by()
        if n_files > 1 and path:
            os.makedirs(path)
        csv_files = []
        while current_nb_rows_written < total:
            if new_header:
                csv_file = sep.join(
                    [
                        quotechar + column.replace('"', "") + quotechar
                        for column in new_header
                    ]
                )
            elif header:
                csv_file = sep.join(
                    [
                        quotechar + column.replace('"', "") + quotechar
                        for column in columns
                    ]
                )
            result = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('vDataframe.to_csv')*/ 
                        {', '.join(columns)} 
                    FROM {self}
                    {order_by} 
                    LIMIT {limit} 
                    OFFSET {current_nb_rows_written}""",
                title="Reading the data.",
                method="fetchall",
                sql_push_ext=self._vars["sql_push_ext"],
                symbol=self._vars["symbol"],
            )
            for row in result:
                tmp_row = []
                for item in row:
                    if isinstance(item, str):
                        tmp_row += [
                            quotechar
                            + item.replace(quotechar, quotechar * 2)
                            + quotechar
                        ]
                    elif isinstance(item, NoneType):
                        tmp_row += ["" if isinstance(na_rep, NoneType) else na_rep]
                    else:
                        tmp_row += [str(item)]
                csv_file += "\n" + sep.join(tmp_row)
            current_nb_rows_written += limit
            file_id += 1
            if n_files == 1 and path:
                with open(path, "w+", encoding="utf-8") as f:
                    f.write(csv_file)
            elif path:
                with open(f"{path}/{file_id}.csv", "w+", encoding="utf-8") as f:
                    f.write(csv_file)
            else:
                csv_files += [csv_file]
        if not path:
            if n_files == 1:
                return csv_files[0]
            else:
                return csv_files

    @save_verticapy_logs
    def to_db(
        self,
        name: str,
        usecols: Optional[SQLColumns] = None,
        relation_type: Literal[
            "view", "temporary", "table", "local", "insert"
        ] = "view",
        inplace: bool = False,
        db_filter: SQLExpression = "",
        nb_split: int = 0,
        order_by: Union[None, SQLColumns, dict] = None,
        segmented_by: Optional[SQLColumns] = None,
    ) -> "vDataFrame":
        """
        Saves the vDataFrame current relation to the Vertica database.

        Parameters
        ----------
        name: str
            Name of the relation.  To save the relation in a specific
            schema, you can write '"my_schema"."my_relation"'.
            Use  double  quotes '"' to  avoid errors due  to  special
            characters.
        usecols: SQLColumns, optional
            vDataColumns to select from the final vDataFrame relation.
            If empty, all vDataColumns are selected.
        relation_type: str, optional
            Type of the relation.
                view      : View
                table     : Table
                temporary : Temporary Table
                local     : Local Temporary Table
                insert    : Inserts into an existing table
        inplace: bool, optional
            If set to True, the vDataFrame is replaced with the new
            relation.
        db_filter: SQLExpression, optional
            Filter used before  creating the relation in the DB. It can
            be a list of conditions or an expression. This parameter is
            useful for creating train and test sets on TS.
        nb_split: int, optional
            If this parameter is greater than 0, it adds a new column
            '_verticapy_split_' to the final relation. This column
            contains values in [0;nb_split - 1] where each category
            represents 1 / nb_split of the entire distribution.
        order_by: SQLColumns / dict, optional
            List of the vDataColumns used to sort  the data, using asc
            order or a dictionary of all sorting methods. For example,
            to sort by "column1" ASC and "column2" DESC, write:
            {"column1": "asc", "column2": "desc"}
        segmented_by: SQLColumns, optional
            This  parameter is only  used when relation_type is 'table'
            or 'temporary'. Otherwise, it is ignored.
            List of the vDataColumns used to segment the data; All the
            columns used will be passed to the HASH function.

        Returns
        -------
        vDataFrame
            self
        """
        relation_type = relation_type.lower()
        usecols, order_by = format_type(usecols, order_by, dtype=list)
        usecols = self.format_colnames(usecols)
        if relation_type in ("local", "temporary"):
            commit = " ON COMMIT PRESERVE ROWS"
        else:
            commit = ""
        order_by = self._get_sort_syntax(order_by)
        if not order_by:
            order_by = self._get_last_order_by()
        if relation_type in ("table", "temporary"):
            segmented_by = self._get_hash_syntax(segmented_by)
        else:
            segmented_by = ""
        if relation_type == "temporary":
            relation_type += " table"
        elif relation_type == "local":
            relation_type += " temporary table"
        isflex = self._vars["isflex"]
        if not usecols:
            usecols = self.get_columns()
        if not usecols and not isflex:
            select = "*"
        elif usecols and not isflex:
            select = ", ".join(quote_ident(usecols))
        else:
            select = []
            for column in usecols:
                ctype, col = self[column].ctype(), quote_ident(column)
                if ctype.startswith("vmap"):
                    column = f"MAPTOSTRING({col}) AS {col}"
                else:
                    column += f"::{ctype}"
                select += [column]
            select = ", ".join(select)
        insert_usecols = ", ".join(quote_ident(usecols))
        random_func = _current_random(nb_split)
        nb_split = f", {random_func} AS _verticapy_split_" if (nb_split > 0) else ""
        if isinstance(db_filter, Iterable) and not isinstance(db_filter, str):
            db_filter = " AND ".join([f"({elem})" for elem in db_filter])
        db_filter = f" WHERE {db_filter}" if (db_filter) else ""
        if relation_type == "insert":
            insert_usecols_str = (
                f" ({insert_usecols})" if not nb_split and select != "*" else ""
            )
            query = f"""
                INSERT INTO {name}{insert_usecols_str} 
                    SELECT 
                        {select}{nb_split} 
                    FROM {self}
                    {db_filter}
                    {order_by}"""
            title = f"Inserting data in {name}."
            history_message = (
                "[Insert]: The vDataFrame was inserted into the " f"table '{name}'."
            )
        else:
            query = f"""
                CREATE 
                    {relation_type.upper()}
                    {name}{commit} 
                AS 
                SELECT 
                    /*+LABEL('vDataframe.to_db')*/ 
                    {select}{nb_split} 
                FROM {self}
                {db_filter}
                {order_by}
                {segmented_by}"""
            title = f"Creating a new {relation_type} to save the vDataFrame."
            history_message = (
                "[Save]: The vDataFrame was saved into a "
                f"{relation_type} named '{name}'."
            )
        _executeSQL(
            query=query,
            title=title,
        )
        if relation_type == "insert":
            _executeSQL(query="COMMIT;", title="Commit.")
        self._add_to_history(history_message)
        if inplace:
            history = self._vars["history"]
            catalog_vars = {}
            for column in usecols:
                catalog_vars[column] = self[column]._catalog
            if relation_type == "local temporary table":
                self.__init__("v_temp_schema." + name)
            else:
                self.__init__(name)
            self._vars["history"] = history
            for column in usecols:
                self[column]._catalog = catalog_vars[column]
        return self

    @save_verticapy_logs
    def to_geopandas(self, geometry: str) -> "GeoDataFrame":
        """
        Converts the vDataFrame to a Geopandas DataFrame.

        \u26A0 Warning : The data will be loaded in memory.

        Parameters
        ----------
        geometry: str
            Geometry object used to create the GeoDataFrame.
            It can also be a Geography object, which will be
            casted to Geometry.

        Returns
        -------
        geopandas.GeoDataFrame
            The geopandas.GeoDataFrame of the current vDataFrame
            relation.
        """
        if not conf.get_import_success("geopandas"):
            raise ImportError(
                "The geopandas module doesn't seem to be installed in your "
                "environment.\nTo be able to use this method, you'll have to "
                "install it.\n[Tips] Run: 'pip3 install geopandas' in your "
                "terminal to install the module."
            )
        columns = self.get_columns(exclude_columns=[geometry])
        columns = ", ".join(columns + [f"ST_AsText({geometry}) AS {geometry}"])
        query = f"""
            SELECT 
                /*+LABEL('vDataframe.to_geopandas')*/ {columns} 
            FROM {self}
            {self._get_last_order_by()}"""
        data = _executeSQL(
            query, title="Getting the vDataFrame values.", method="fetchall"
        )
        column_names = [column[0] for column in current_cursor().description]
        df = pd.DataFrame(data)
        df.columns = column_names
        if len(geometry) > 2 and geometry[0] == geometry[-1] == '"':
            geometry = geometry[1:-1]
        df[geometry] = df[geometry].apply(wkt.loads)
        df = GeoDataFrame(df, geometry=geometry)
        return df

    @save_verticapy_logs
    def to_json(
        self,
        path: Optional[str] = None,
        usecols: Optional[SQLColumns] = None,
        order_by: Union[None, SQLColumns, dict] = None,
        n_files: int = 1,
    ) -> Union[None, str, list[str]]:
        """
        Creates  a JSON file or folder of JSON files of the  current
        vDataFrame relation.

        Parameters
        ----------
        path: str, optional
            File/Folder system path. Be careful: if a JSON file with
            the same name exists, it is overwritten.
        usecols: SQLColumns, optional
            vDataColumns to select from the final vDataFrame relation.
            If empty, all vDataColumns are selected.
        order_by: str / dict / list, optional
            List of the vDataColumns used to sort the data, using asc
            order or dictionary  of all sorting  methods.  For example,
            to   sort   by   "column1"    ASC   and   "column2"   DESC,
            write: {"column1": "asc", "column2": "desc"}
        n_files: int, optional
            Integer  greater than or equal  to 1, the number of CSV files
            to generate.  If n_files is greater than 1, you must also set
            order_by to sort  the data, ideally with a column with unique
            values (e.g. ID).
            Greater values of n_files decrease memory usage, but increase
            execution time.

        Returns
        -------
        str or list
            JSON str or list (n_files>1) if 'path' is not defined;
            otherwise, nothing.
        """
        order_by, usecols = format_type(order_by, usecols, dtype=list)
        if n_files < 1:
            raise ValueError("Parameter 'n_files' must be greater or equal to 1.")
        if (n_files != 1) and not order_by:
            raise ValueError(
                "If you want to store the vDataFrame in many JSON files, you "
                "have to sort your data by using at least one column. If "
                "the column hasn't unique values, the final result can not "
                "be guaranteed."
            )
        columns = self.get_columns() if not usecols else quote_ident(usecols)
        transformations, is_complex_vmap = [], []
        for col in columns:
            if self[col].category() == "complex":
                transformations += [f"TO_JSON({col}) AS {col}"]
                is_complex_vmap += [True]
            elif self[col].category() == "vmap":
                transformations += [f"MAPTOSTRING({col}) AS {col}"]
                is_complex_vmap += [True]
            else:
                transformations += [col]
                is_complex_vmap += [False]
        total = self.shape()[0]
        current_nb_rows_written, file_id = 0, 0
        limit = int(total / n_files) + 1
        order_by = self._get_sort_syntax(order_by)
        if not order_by:
            order_by = self._get_last_order_by()
        if n_files > 1 and path:
            os.makedirs(path)
        if not path:
            json_files = []
        while current_nb_rows_written < total:
            json_file = "[\n"
            result = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('vDataframe.to_json')*/ 
                        {', '.join(transformations)} 
                    FROM {self}
                    {order_by} 
                    LIMIT {limit} 
                    OFFSET {current_nb_rows_written}""",
                title="Reading the data.",
                method="fetchall",
                sql_push_ext=self._vars["sql_push_ext"],
                symbol=self._vars["symbol"],
            )
            for row in result:
                tmp_row = []
                for i, item in enumerate(row):
                    if isinstance(item, (float, int, decimal.Decimal)) or (
                        isinstance(item, (str,)) and is_complex_vmap[i]
                    ):
                        tmp_row += [f"{quote_ident(columns[i])}: {item}"]
                    elif not isinstance(item, NoneType):
                        tmp_row += [f'{quote_ident(columns[i])}: "{item}"']
                json_file += "{" + ", ".join(tmp_row) + "},\n"
            current_nb_rows_written += limit
            file_id += 1
            json_file = json_file[0:-2] + "\n]"
            if n_files == 1 and path:
                with open(path, "w+", encoding="utf-8") as f:
                    f.write(json_file)
            elif path:
                with open(f"{path}/{file_id}.json", "w+", encoding="utf-8") as f:
                    f.write(json_file)
            else:
                json_files += [json_file]
        if not path:
            if n_files == 1:
                return json_files[0]
            else:
                return json_files

    @save_verticapy_logs
    def to_list(self) -> list:
        """
        Converts the vDataFrame to a Python list.

        \u26A0 Warning : The data will be loaded in memory.

        Returns
        -------
        List
            The list of the current vDataFrame relation.
        """
        res = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.to_list')*/ * 
                FROM {self}
                {self._get_last_order_by()}""",
            title="Getting the vDataFrame values.",
            method="fetchall",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        final_result = []
        for row in res:
            final_result += [
                [
                    float(item) if isinstance(item, decimal.Decimal) else item
                    for item in row
                ]
            ]
        return final_result

    @save_verticapy_logs
    def to_numpy(self) -> np.ndarray:
        """
        Converts the vDataFrame to a Numpy array.

        \u26A0 Warning : The data will be loaded in memory.

        Returns
        -------
        numpy.array
            The numpy array of the current vDataFrame relation.
        """
        return np.array(self.to_list())

    @save_verticapy_logs
    def to_pandas(self) -> pd.DataFrame:
        """
        Converts the vDataFrame to a pandas DataFrame.

        \u26A0 Warning : The data will be loaded in memory.

        Returns
        -------
        pandas.DataFrame
            The pandas.DataFrame of the current vDataFrame relation.
        """
        data = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.to_pandas')*/ * 
                FROM {self}{self._get_last_order_by()}""",
            title="Getting the vDataFrame values.",
            method="fetchall",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        column_names = [column[0] for column in current_cursor().description]
        df = pd.DataFrame(data)
        df.columns = column_names
        return df

    @save_verticapy_logs
    def to_parquet(
        self,
        directory: str,
        compression: Literal[
            "snappy", "gzip", "brotli", "zstd", "uncompressed"
        ] = "snappy",
        rowGroupSizeMB: int = 512,
        fileSizeMB: int = 10000,
        fileMode: str = "660",
        dirMode: str = "755",
        int96AsTimestamp: bool = True,
        by: Optional[SQLColumns] = None,
        order_by: Union[None, SQLColumns, dict] = None,
    ) -> TableSample:
        """
        Exports  a  table, columns from a  table, or query results  to
        Parquet  files.  You  can  partition  data  instead of, or  in
        addition to, exporting the column data, which enables partition
        pruning and improves query performance.

        Parameters
        ----------
        directory: str
            The  destination  directory  for  the output file(s).  The
            directory must not already exist, and the current user must
            have write permissions on it.
            The destination can be one of the following file systems:
                HDFS File System
                S3 Object Store
                Google Cloud Storage (GCS) Object Store
                Azure Blob Storage Object Store
                Linux file system (either an NFS mount or local storage
                on each node)
        compression: str, optional
            Column compression type, one the following:
                Snappy (default)
                gzip
                Brotli
                zstd
                Uncompressed
        rowGroupSizeMB: int, optional
            The  uncompressed size,  in MB, of exported  row  groups, an
            integer value in the range [1, fileSizeMB]. If fileSizeMB is
            0, the uncompressed size is unlimited.
            Row groups in the exported files are smaller than this value
            because Parquet files are compressed on write.
            For best performance when exporting to HDFS, set this
            rowGroupSizeMB to be smaller than the HDFS block size.
        fileSizeMB: int, optional
            The maximum file size of a single output file. This fileSizeMB
            is a hint/ballpark and not a hard limit.
            A value of 0 indicates that the size of a single output file is
            unlimited. This parameter affects the size of individual output
            files, not the total output size.
            For smaller values, Vertica divides the output into more files;
            all data is still exported.
        fileMode: int, optional
            HDFS only: the permission to apply to all exported files.  You
            can specify the value in octal (such as 755) or symbolic (such
            as rwxr-xr-x) modes.
            The value must be a string even when using octal mode.
            Valid octal values are in the range [0,1777]. For details, see
            HDFS Permissions in the Apache Hadoop documentation.
            If the destination is not HDFS, this parameter has no effect.
        dirMode: int, optional
            HDFS only:  the permission to apply to all exported  directories.
            Values follow the same rules as those for fileMode. Additionally,
            you must give the Vertica HDFS user full
            permissions: at least rwx------ (symbolic) or 700 (octal).
            If the destination is not HDFS, this parameter has no effect.
        int96AsTimestamp: bool, optional
            Boolean, specifies whether to export timestamps as int96 physical
            type (True) or int64 physical type (False).
        by: SQLColumns, optional
            vDataColumns used in the partition.
        order_by: str / dict / list, optional
            If specified as a list: the list of vDataColumns useed to sort the
            data in ascending order.
            If specified as a dictionary:  a dictionary of all sorting methods.
            For example, to sort by "column1" ASC and "column2" DESC:
            {"column1": "asc", "column2": "desc"}

        Returns
        -------
        TableSample
            An object containing the number of rows exported.
        """
        order_by, by = format_type(order_by, by, dtype=list)
        if rowGroupSizeMB <= 0:
            raise ValueError("Parameter 'rowGroupSizeMB' must be greater than 0.")
        if fileSizeMB <= 0:
            raise ValueError("Parameter 'fileSizeMB' must be greater than 0.")
        by = self.format_colnames(by)
        partition = ""
        if by:
            partition = f"PARTITION BY {', '.join(by)}"
        result = TableSample.read_sql(
            query=f"""
                EXPORT TO PARQUET(directory = '{directory}',
                                  compression = '{compression}',
                                  rowGroupSizeMB = {rowGroupSizeMB},
                                  fileSizeMB = {fileSizeMB},
                                  fileMode = '{fileMode}',
                                  dirMode = '{dirMode}',
                                  int96AsTimestamp = {str(int96AsTimestamp).lower()}) 
                          OVER({partition}{self._get_sort_syntax(order_by)}) 
                       AS SELECT * FROM {self};""",
            title="Exporting data to Parquet format.",
            sql_push_ext=self._vars["sql_push_ext"],
            symbol=self._vars["symbol"],
        )
        return result

    @save_verticapy_logs
    def to_pickle(self, name: str) -> "vDataFrame":
        """
        Saves the vDataFrame to a Python pickle file.

        Parameters
        ----------
        name: str
            Name of the file.
            Be careful: if a file with the same name
            exists, it is overwritten.

        Returns
        -------
        vDataFrame
            self
        """
        pickle.dump(self, open(name, "wb"))
        return self

    @save_verticapy_logs
    def to_shp(
        self,
        name: str,
        path: str,
        usecols: Optional[SQLColumns] = None,
        overwrite: bool = True,
        shape: Literal[
            "Point",
            "Polygon",
            "Linestring",
            "Multipoint",
            "Multipolygon",
            "Multilinestring",
        ] = "Polygon",
    ) -> "vDataFrame":
        """
        Creates a SHP file of the current vDataFrame relation.
        For the moment, files will be exported in the Vertica
        server.

        Parameters
        ----------
        name: str
            Name of the SHP file.
        path: str
            Absolute path where the SHP file is created.
        usecols: list, optional
            vDataColumns  to select from the final  vDataFrame
            relation.  If  empty,  all  vDataColumns  are
            selected.
        overwrite: bool, optional
            If set to True,  the  function overwrites the
            index (if an index exists).
        shape: str, optional
            Must be one of the following spatial classes:
                Point, Polygon, Linestring, Multipoint,
                Multipolygon, Multilinestring.
            Polygons and Multipolygons always have a clockwise
            orientation.

        Returns
        -------
        vDataFrame
            self
        """
        usecols = format_type(usecols, dtype=list)
        query = f"""
            SELECT 
                /*+LABEL('vDataframe.to_shp')*/ 
                STV_SetExportShapefileDirectory(
                USING PARAMETERS path = '{path}');"""
        _executeSQL(query=query, title="Setting SHP Export directory.")
        columns = self.get_columns() if not usecols else quote_ident(usecols)
        columns = ", ".join(columns)
        query = f"""
            SELECT 
                /*+LABEL('vDataframe.to_shp')*/ 
                STV_Export2Shapefile({columns} 
                USING PARAMETERS shapefile = '{name}.shp',
                                 overwrite = {overwrite}, 
                                 shape = '{shape}') 
                OVER() 
            FROM {self};"""
        _executeSQL(query=query, title="Exporting the SHP.")
        return self
