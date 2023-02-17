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
from typing import Literal, Union
from collections.abc import Iterable
import copy, decimal, pickle, os
import numpy as np

import pandas as pd

from verticapy._config.config import current_random, GEOPANDAS_ON
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._sql._execute import _executeSQL
from verticapy._utils._sql._format import quote_ident
from verticapy.connect import current_cursor
from verticapy.errors import ParameterError, ParsingError

from verticapy.sql.read import to_tablesample

if GEOPANDAS_ON:
    from geopandas import GeoDataFrame
    from shapely import wkt

pickle.DEFAULT_PROTOCOL = 4


class vDFIO:
    def copy(self):
        """
    Returns a deep copy of the vDataFrame.

    Returns
    -------
    vDataFrame
        The copy of the vDataFrame.
        """
        return copy.deepcopy(self)

    @save_verticapy_logs
    def load(self, offset: int = -1):
        """
    Loads a previous structure of the vDataFrame. 

    Parameters
    ----------
    offset: int, optional
        offset of the saving. Example: -1 to load the last saving.

    Returns
    -------
    vDataFrame
        vDataFrame of the loading.

    See Also
    --------
    vDataFrame.save : Saves the current vDataFrame structure.
        """
        save = self._VERTICAPY_VARIABLES_["saving"][offset]
        vdf = pickle.loads(save)
        return vdf

    @save_verticapy_logs
    def save(self):
        """
    Saves the current structure of the vDataFrame. 
    This function is useful for loading previous transformations.

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.load : Loads a saving.
        """
        vdf = self.copy()
        self._VERTICAPY_VARIABLES_["saving"] += [pickle.dumps(vdf)]
        return self

    @save_verticapy_logs
    def to_csv(
        self,
        path: str = "",
        sep: str = ",",
        na_rep: str = "",
        quotechar: str = '"',
        usecols: Union[str, list] = [],
        header: bool = True,
        new_header: list = [],
        order_by: Union[str, list, dict] = [],
        n_files: int = 1,
    ):
        """
    Creates a CSV file or folder of CSV files of the current vDataFrame 
    relation.

    Parameters
    ----------
    path: str, optional
        File/Folder system path. Be careful: if a CSV file with the same name 
        exists, it will over-write it.
    sep: str, optional
        Column separator.
    na_rep: str, optional
        Missing values representation.
    quotechar: str, optional
        Char which will enclose the str values.
    usecols: str / list, optional
        vDataColumns to select from the final vDataFrame relation. If empty, all
        vDataColumns will be selected.
    header: bool, optional
        If set to False, no header will be written in the CSV file.
    new_header: list, optional
        List of columns to use to replace vDataColumns name in the CSV.
    order_by: str / dict / list, optional
        List of the vDataColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    n_files: int, optional
        Integer greater than or equal to 1, the number of CSV files to generate.
        If n_files is greater than 1, you must also set order_by to sort the data,
        ideally with a column with unique values (e.g. ID).
        Greater values of n_files decrease memory usage, but increase execution 
        time.

    Returns
    -------
    str or list
        JSON str or list (n_files>1) if 'path' is not defined; otherwise, nothing

    See Also
    --------
    vDataFrame.to_db   : Saves the vDataFrame current relation to the Vertica database.
    vDataFrame.to_json : Creates a JSON file of the current vDataFrame relation.
        """
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(usecols, str):
            usecols = [usecols]
        if n_files < 1:
            raise ParameterError("Parameter 'n_files' must be greater or equal to 1.")
        if (n_files != 1) and not (order_by):
            raise ParameterError(
                "If you want to store the vDataFrame in many CSV files, "
                "you have to sort your data by using at least one column. "
                "If the column hasn't unique values, the final result can "
                "not be guaranteed."
            )
        columns = (
            self.get_columns()
            if not (usecols)
            else [quote_ident(column) for column in usecols]
        )
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
        order_by = self.__get_sort_syntax__(order_by)
        if not (order_by):
            order_by = self.__get_last_order_by__()
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
                    FROM {self.__genSQL__()}
                    {order_by} 
                    LIMIT {limit} 
                    OFFSET {current_nb_rows_written}""",
                title="Reading the data.",
                method="fetchall",
                sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self._VERTICAPY_VARIABLES_["symbol"],
            )
            for row in result:
                tmp_row = []
                for item in row:
                    if isinstance(item, str):
                        tmp_row += [quotechar + item + quotechar]
                    elif item == None:
                        tmp_row += [na_rep]
                    else:
                        tmp_row += [str(item)]
                csv_file += "\n" + sep.join(tmp_row)
            current_nb_rows_written += limit
            file_id += 1
            if n_files == 1 and path:
                file = open(path, "w+")
                file.write(csv_file)
                file.close()
            elif path:
                file = open(f"{path}/{file_id}.csv", "w+")
                file.write(csv_file)
                file.close()
            else:
                csv_files += [csv_file]
        if not (path):
            if n_files == 1:
                return csv_files[0]
            else:
                return csv_files

    @save_verticapy_logs
    def to_db(
        self,
        name: str,
        usecols: Union[str, list] = [],
        relation_type: Literal[
            "view", "temporary", "table", "local", "insert"
        ] = "view",
        inplace: bool = False,
        db_filter: Union[str, list] = "",
        nb_split: int = 0,
    ):
        """
    Saves the vDataFrame current relation to the Vertica database.

    Parameters
    ----------
    name: str
        Name of the relation. To save the relation in a specific schema you can
        write '"my_schema"."my_relation"'. Use double quotes '"' to avoid errors
        due to special characters.
    usecols: str / list, optional
        vDataColumns to select from the final vDataFrame relation. If empty, all
        vDataColumns will be selected.
    relation_type: str, optional
        Type of the relation.
            view      : View
            table     : Table
            temporary : Temporary Table
            local     : Local Temporary Table
            insert    : Inserts into an existing table
    inplace: bool, optional
        If set to True, the vDataFrame will be replaced using the new relation.
    db_filter: str / list, optional
        Filter used before creating the relation in the DB. It can be a list of
        conditions or an expression. This parameter is very useful to create train 
        and test sets on TS.
    nb_split: int, optional
        If this parameter is greater than 0, it will add to the final relation a
        new column '_verticapy_split_' which will contain values in 
        [0;nb_split - 1] where each category will represent 1 / nb_split
        of the entire distribution. 

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.to_csv : Creates a csv file of the current vDataFrame relation.
        """
        if isinstance(usecols, str):
            usecols = [usecols]
        relation_type = relation_type.lower()
        usecols = self.format_colnames(usecols)
        commit = (
            " ON COMMIT PRESERVE ROWS"
            if (relation_type in ("local", "temporary"))
            else ""
        )
        if relation_type == "temporary":
            relation_type += " table"
        elif relation_type == "local":
            relation_type += " temporary table"
        isflex = self._VERTICAPY_VARIABLES_["isflex"]
        if not (usecols):
            usecols = self.get_columns()
        if not (usecols) and not (isflex):
            select = "*"
        elif usecols and not (isflex):
            select = ", ".join([quote_ident(column) for column in usecols])
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
        insert_usecols = ", ".join([quote_ident(column) for column in usecols])
        random_func = current_random(nb_split)
        nb_split = f", {random_func} AS _verticapy_split_" if (nb_split > 0) else ""
        if isinstance(db_filter, Iterable) and not (isinstance(db_filter, str)):
            db_filter = " AND ".join([f"({elem})" for elem in db_filter])
        db_filter = f" WHERE {db_filter}" if (db_filter) else ""
        if relation_type == "insert":
            insert_usecols_str = (
                f" ({insert_usecols})" if not (nb_split) and select != "*" else ""
            )
            query = f"""
                INSERT INTO {name}{insert_usecols_str} 
                    SELECT 
                        {select}{nb_split} 
                    FROM {self.__genSQL__()}
                    {db_filter}
                    {self.__get_last_order_by__()}"""
        else:
            query = f"""
                CREATE 
                    {relation_type.upper()}
                    {name}{commit} 
                AS 
                SELECT 
                    /*+LABEL('vDataframe.to_db')*/ 
                    {select}{nb_split} 
                FROM {self.__genSQL__()}
                {db_filter}
                {self.__get_last_order_by__()}"""
        _executeSQL(
            query=query,
            title=f"Creating a new {relation_type} to save the vDataFrame.",
        )
        if relation_type == "insert":
            _executeSQL(query="COMMIT;", title="Commit.")
        self.__add_to_history__(
            "[Save]: The vDataFrame was saved into a "
            f"{relation_type} named '{name}'."
        )
        if inplace:
            history, saving = (
                self._VERTICAPY_VARIABLES_["history"],
                self._VERTICAPY_VARIABLES_["saving"],
            )
            catalog_vars = {}
            for column in usecols:
                catalog_vars[column] = self[column].catalog
            if relation_type == "local temporary table":
                self.__init__("v_temp_schema." + name)
            else:
                self.__init__(name)
            self._VERTICAPY_VARIABLES_["history"] = history
            for column in usecols:
                self[column].catalog = catalog_vars[column]
        return self

    @save_verticapy_logs
    def to_geopandas(self, geometry: str):
        """
    Converts the vDataFrame to a Geopandas DataFrame.

    \u26A0 Warning : The data will be loaded in memory.

    Parameters
    ----------
    geometry: str
        Geometry object used to create the GeoDataFrame.
        It can also be a Geography object but it will be casted to Geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        The geopandas.GeoDataFrame of the current vDataFrame relation.
        """
        if not (GEOPANDAS_ON):
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
            FROM {self.__genSQL__()}
            {self.__get_last_order_by__()}"""
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
        path: str = "",
        usecols: Union[str, list] = [],
        order_by: Union[str, list, dict] = [],
        n_files: int = 1,
    ):
        """
    Creates a JSON file or folder of JSON files of the current vDataFrame 
    relation.

    Parameters
    ----------
    path: str, optional
        File/Folder system path. Be careful: if a JSON file with the same name 
        exists, it will over-write it.
    usecols: str / list, optional
        vDataColumns to select from the final vDataFrame relation. If empty, all
        vDataColumns will be selected.
    order_by: str / dict / list, optional
        List of the vDataColumns to use to sort the data using asc order or
        dictionary of all sorting methods. For example, to sort by "column1"
        ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
    n_files: int, optional
        Integer greater than or equal to 1, the number of CSV files to generate.
        If n_files is greater than 1, you must also set order_by to sort the data,
        ideally with a column with unique values (e.g. ID).
        Greater values of n_files decrease memory usage, but increase execution time.

    Returns
    -------
    str or list
        JSON str or list (n_files>1) if 'path' is not defined; otherwise, nothing

    See Also
    --------
    vDataFrame.to_csv : Creates a CSV file of the current vDataFrame relation.
    vDataFrame.to_db  : Saves the vDataFrame current relation to the Vertica database.
        """
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(usecols, str):
            usecols = [usecols]
        if n_files < 1:
            raise ParameterError("Parameter 'n_files' must be greater or equal to 1.")
        if (n_files != 1) and not (order_by):
            raise ParameterError(
                "If you want to store the vDataFrame in many JSON files, you "
                "have to sort your data by using at least one column. If "
                "the column hasn't unique values, the final result can not "
                "be guaranteed."
            )
        columns = (
            self.get_columns()
            if not (usecols)
            else [quote_ident(column) for column in usecols]
        )
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
        order_by = self.__get_sort_syntax__(order_by)
        if not (order_by):
            order_by = self.__get_last_order_by__()
        if n_files > 1 and path:
            os.makedirs(path)
        if not (path):
            json_files = []
        while current_nb_rows_written < total:
            json_file = "[\n"
            result = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('vDataframe.to_json')*/ 
                        {', '.join(transformations)} 
                    FROM {self.__genSQL__()}
                    {order_by} 
                    LIMIT {limit} 
                    OFFSET {current_nb_rows_written}""",
                title="Reading the data.",
                method="fetchall",
                sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
                symbol=self._VERTICAPY_VARIABLES_["symbol"],
            )
            for row in result:
                tmp_row = []
                for i, item in enumerate(row):
                    if isinstance(item, (float, int, decimal.Decimal)) or (
                        isinstance(item, (str,)) and is_complex_vmap[i]
                    ):
                        tmp_row += [f"{quote_ident(columns[i])}: {item}"]
                    elif item != None:
                        tmp_row += [f'{quote_ident(columns[i])}: "{item}"']
                json_file += "{" + ", ".join(tmp_row) + "},\n"
            current_nb_rows_written += limit
            file_id += 1
            json_file = json_file[0:-2] + "\n]"
            if n_files == 1 and path:
                file = open(path, "w+")
                file.write(json_file)
                file.close()
            elif path:
                file = open(f"{path}/{file_id}.json", "w+")
                file.write(json_file)
                file.close()
            else:
                json_files += [json_file]
        if not (path):
            if n_files == 1:
                return json_files[0]
            else:
                return json_files

    @save_verticapy_logs
    def to_list(self):
        """
    Converts the vDataFrame to a Python list.

    \u26A0 Warning : The data will be loaded in memory.

    Returns
    -------
    List
        The list of the current vDataFrame relation.
        """
        result = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('vDataframe.to_list')*/ * 
                FROM {self.__genSQL__()}
                {self.__get_last_order_by__()}""",
            title="Getting the vDataFrame values.",
            method="fetchall",
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
        )
        final_result = []
        for elem in result:
            final_result += [
                [
                    float(item) if isinstance(item, decimal.Decimal) else item
                    for item in elem
                ]
            ]
        return final_result

    @save_verticapy_logs
    def to_numpy(self):
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
    def to_pandas(self):
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
                FROM {self.__genSQL__()}{self.__get_last_order_by__()}""",
            title="Getting the vDataFrame values.",
            method="fetchall",
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
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
        by: Union[str, list] = [],
        order_by: Union[str, list, dict] = [],
    ):
        """
    Exports a table, columns from a table, or query results to Parquet files.
    You can partition data instead of or in addition to exporting the column data, 
    which enables partition pruning and improves query performance. 

    Parameters
    ----------
    directory: str
        The destination directory for the output file(s). The directory must not 
        already exist, and the current user must have write permissions on it. 
        The destination can be one of the following file systems: 
            HDFS File System
            S3 Object Store
            Google Cloud Storage (GCS) Object Store
            Azure Blob Storage Object Store
            Linux file system (either an NFS mount or local storage on each node)
    compression: str, optional
        Column compression type, one the following:        
            Snappy (default)
            gzip
            Brotli
            zstd
            Uncompressed
    rowGroupSizeMB: int, optional
        The uncompressed size, in MB, of exported row groups, an integer value in the range
        [1, fileSizeMB]. If fileSizeMB is 0, the uncompressed size is unlimited.
        Row groups in the exported files are smaller than this value because Parquet 
        files are compressed on write. 
        For best performance when exporting to HDFS, set this rowGroupSizeMB to be 
        smaller than the HDFS block size.
    fileSizeMB: int, optional
        The maximum file size of a single output file. This fileSizeMB is a hint/ballpark 
        and not a hard limit. 
        A value of 0 indicates that the size of a single output file is unlimited.  
        This parameter affects the size of individual output files, not the total output size. 
        For smaller values, Vertica divides the output into more files; all data is still exported.
    fileMode: int, optional
        HDFS only: the permission to apply to all exported files. You can specify 
        the value in octal (such as 755) or symbolic (such as rwxr-xr-x) modes. 
        The value must be a string even when using octal mode.
        Valid octal values are in the range [0,1777]. For details, see HDFS Permissions in the 
        Apache Hadoop documentation.
        If the destination is not HDFS, this parameter has no effect.
    dirMode: int, optional
        HDFS only: the permission to apply to all exported directories. Values follow 
        the same rules as those for fileMode. Additionally, you must give the Vertica HDFS user full 
        permissions: at least rwx------ (symbolic) or 700 (octal).
        If the destination is not HDFS, this parameter has no effect.
    int96AsTimestamp: bool, optional
        Boolean, specifies whether to export timestamps as int96 physical type (True) or int64 
        physical type (False).
    by: str / list, optional
        vDataColumns used in the partition.
    order_by: str / dict / list, optional
        If specified as a list: the list of vDataColumns useed to sort the data in ascending order.
        If specified as a dictionary: a dictionary of all sorting methods.
        For example, to sort by "column1" ASC and "column2" DESC: {"column1": "asc", "column2": "desc"}

    Returns
    -------
    tablesample
        An object containing the number of rows exported. For details, 
        see utilities.tablesample.

    See Also
    --------
    vDataFrame.to_csv : Creates a CSV file of the current vDataFrame relation.
    vDataFrame.to_db  : Saves the current relation's vDataFrame to the Vertica database.
    vDataFrame.to_json: Creates a JSON file of the current vDataFrame relation.
        """
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(by, str):
            by = [by]
        if rowGroupSizeMB <= 0:
            raise ParameterError("Parameter 'rowGroupSizeMB' must be greater than 0.")
        if fileSizeMB <= 0:
            raise ParameterError("Parameter 'fileSizeMB' must be greater than 0.")
        by = self.format_colnames(by)
        partition = ""
        if by:
            partition = f"PARTITION BY {', '.join(by)}"
        result = to_tablesample(
            query=f"""
                EXPORT TO PARQUET(directory = '{directory}',
                                  compression = '{compression}',
                                  rowGroupSizeMB = {rowGroupSizeMB},
                                  fileSizeMB = {fileSizeMB},
                                  fileMode = '{fileMode}',
                                  dirMode = '{dirMode}',
                                  int96AsTimestamp = {str(int96AsTimestamp).lower()}) 
                          OVER({partition}{self.__get_sort_syntax__(order_by)}) 
                       AS SELECT * FROM {self.__genSQL__()};""",
            title="Exporting data to Parquet format.",
            sql_push_ext=self._VERTICAPY_VARIABLES_["sql_push_ext"],
            symbol=self._VERTICAPY_VARIABLES_["symbol"],
        )
        return result

    @save_verticapy_logs
    def to_pickle(self, name: str):
        """
    Saves the vDataFrame to a Python pickle file.

    Parameters
    ----------
    name: str
        Name of the file. Be careful: if a file with the same name exists, it 
        will over-write it.

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
        usecols: Union[str, list] = [],
        overwrite: bool = True,
        shape: Literal[
            "Point",
            "Polygon",
            "Linestring",
            "Multipoint",
            "Multipolygon",
            "Multilinestring",
        ] = "Polygon",
    ):
        """
    Creates a SHP file of the current vDataFrame relation. For the moment, 
    files will be exported in the Vertica server.

    Parameters
    ----------
    name: str
        Name of the SHP file.
    path: str
        Absolute path where the SHP file will be created.
    usecols: list, optional
        vDataColumns to select from the final vDataFrame relation. If empty, all
        vDataColumns will be selected.
    overwrite: bool, optional
        If set to True, the function will overwrite the index if an index exists.
    shape: str, optional
        Must be one of the following spatial classes: 
            Point, Polygon, Linestring, Multipoint, Multipolygon, Multilinestring. 
        Polygons and Multipolygons always have a clockwise orientation.

    Returns
    -------
    vDataFrame
        self
        """
        if isinstance(usecols, str):
            usecols = [usecols]
        query = f"""
            SELECT 
                /*+LABEL('vDataframe.to_shp')*/ 
                STV_SetExportShapefileDirectory(
                USING PARAMETERS path = '{path}');"""
        _executeSQL(query=query, title="Setting SHP Export directory.")
        columns = (
            self.get_columns()
            if not (usecols)
            else [quote_ident(column) for column in usecols]
        )
        columns = ", ".join(columns)
        query = f"""
            SELECT 
                /*+LABEL('vDataframe.to_shp')*/ 
                STV_Export2Shapefile({columns} 
                USING PARAMETERS shapefile = '{name}.shp',
                                 overwrite = {overwrite}, 
                                 shape = '{shape}') 
                OVER() 
            FROM {self.__genSQL__()};"""
        _executeSQL(query=query, title="Exporting the SHP.")
        return self
