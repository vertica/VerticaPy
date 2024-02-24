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
import csv
import logging
import os
from typing import Optional

import pandas as pd

import verticapy._config.config as conf
from verticapy._typing import NoneType
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_schema_table, format_type, quote_ident
from verticapy._utils._sql._sys import _executeSQL


from verticapy.core.parsers.csv import read_csv
from verticapy.core.vdataframe.base import vDataFrame


@save_verticapy_logs
def read_pandas(
    df: pd.DataFrame,
    name: Optional[str] = None,
    schema: Optional[str] = None,
    dtype: Optional[dict] = None,
    parse_nrows: int = 10000,
    temp_path: Optional[str] = None,
    insert: bool = False,
    abort_on_error: bool = False,
) -> vDataFrame:
    """
    Ingests a ``pandas.DataFrame`` into
    the Vertica database by creating
    a CSV file and then using flex
    tables to load the data.

    Parameters
    ----------
    df: pandas.DataFrame
        The ``pandas.DataFrame`` to
        ingest.
    name: str, optional
        Name of the new relation or
        the relation in which to
        insert the data.
        If unspecified, a temporary
        local table is created. This
        temporary table is dropped at
        the end of the local session.
    schema: str, optional
        Schema of the new relation.
        If empty, a temporary schema
        is used. To modify the temporary
        schema, use the :py:func:`~set_option`
        function.
    dtype: dict, optional
        Dictionary of input types.
        Providing a dictionary can
        increase ingestion speed and
        precision. If specified,
        rather than parsing the
        intermediate CSV and guessing
        the input types, VerticaPy
        uses the specified input
        types instead.
    parse_nrows: int, optional
        If this parameter is greater
        than zero, VerticaPy creates
        and ingests a temporary file
        containing ``parse_nrows``
        number of rows to determine
        the input data types before
        ingesting the intermediate
        CSV file containing the rest
        of the data. This method of
        data type identification is
        less accurate, but is much
        faster for large datasets.
    temp_path: str, optional
        The path to which to write
        the intermediate CSV file.
        This is useful in cases
        where the user does not
        have write permissions
        on the current directory.
    insert: bool, optional
        If set to ``True``, the
        data are ingested into the
        input relation. The column
        names of your table and the
        ``pandas.DataFrame`` must
        match.
    abort_on_error: bool, optional
        If set to ``True``, any parser
        error that would reject a row
        will cause the copy statement
        to fail and rollback.

    Returns
    -------
    vDataFrame
        :py:class:`~vDataFrame`
        of the new relation.

    Examples
    --------

    In this example, we will first create
    a ``pandas.DataFrame`` using
    ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.to_pandas`
    and ingest it into Vertica database.

    We import :py:mod:`verticapy`:

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

    We will use the Titanic dataset.

    .. code-block:: python

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

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd

        data = vpd.load_titanic()

    Let's convert the :py:class:`~vDataFrame`
    to a ``pandas.DataFrame``.

    .. code-block:: python

        pandas_df = data.to_pandas()
        display(pandas_df)

    .. ipython:: python
        :suppress:

        pandas_df = data.to_pandas()
        res = pandas_df
        html_file = open("figures/core_parsers_pandas_1.html", "w")
        html_file.write(res.to_html(max_rows = 6, justify = "center"))
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_parsers_pandas_1.html

    Now, we will ingest the
    ``pandas.DataFrame``
    into the Vertica database.

    .. code-block:: python

        from verticapy.core.parsers import read_pandas

        read_pandas(
            df = pandas_df,
            name = "titanic_pandas",
            schema = "public",
        )

    .. ipython:: python
        :suppress:
        :okexcept:

        from verticapy.core.parsers import read_pandas
        res = read_pandas(
            df = pandas_df,
            name = "titanic_pandas",
            schema = "public",
        )
        html_file = open("figures/core_parsers_pandas_2.html", "w")
        html_file.write(res._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_parsers_pandas_2.html

    Let's specify data types using
    "dtypes" parameter.

    .. code-block:: python

        read_pandas(
            df = pandas_df,
            name = "titanic_pandas_dtypes",
            schema = "public",
            dtype = {
                "pclass": "Integer",
                "survived": "Integer",
                "name": "Varchar(164)",
                "sex": "Varchar(20)",
                "age": "Numeric(6,3)",
                "sibsp": "Integer",
                "parch": "Integer",
                "ticket": "Varchar(36)",
                "fare": "Numeric(10,5)",
                "cabin": "Varchar(30)",
                "embarked": "Varchar(20)",
                "boat": "Varchar(100)",
                "body": "Integer",
                "home.dest": "Varchar(100)",
            },
        )

    .. ipython:: python
        :suppress:
        :okexcept:

        res = read_pandas(
            df = pandas_df,
            name = "titanic_pandas_dtypes",
            schema = "public",
            dtype = {
                "pclass": "Integer",
                "survived": "Integer",
                "name": "Varchar(164)",
                "sex": "Varchar(20)",
                "age": "Numeric(6,3)",
                "sibsp": "Integer",
                "parch": "Integer",
                "ticket": "Varchar(36)",
                "fare": "Numeric(10,5)",
                "cabin": "Varchar(30)",
                "embarked": "Varchar(20)",
                "boat": "Varchar(100)",
                "body": "Integer",
                "home.dest": "Varchar(100)",
            },
        )
        html_file = open("figures/core_parsers_pandas_3.html", "w")
        html_file.write(res._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_parsers_pandas_3.html

    .. important::

        A limited number of rows, determined by the
        ``parse_nrows`` parameter, is ingested. If
        your dataset is large and you want to ingest
        the entire dataset, increase its value.

    .. note::

        During the ingestion process, an intermediate
        CSV file is created. You can retrieve its
        location by using the temp_path parameter.

    .. note::

        If you want to ingest into an existing table,
        set the insert parameter to ``True``.

    .. seealso::

        | :py:func:`~verticapy.read_avro` :
            Ingests a AVRO file into the Vertica DB.
        | :py:func:`~verticapy.read_csv` :
            Ingests a CSV file into the Vertica DB.
        | :py:func:`~verticapy.read_file` :
            Ingests an input file into the Vertica DB.
        | :py:func:`~verticapy.read_json` :
            Ingests a JSON file into the Vertica DB.
    """
    dtype = format_type(dtype, dtype=dict)
    if not schema:
        schema = conf.get_option("temp_schema")
    if insert and not name:
        raise ValueError(
            "Parameter 'name' can not be empty when "
            "parameter 'insert' is set to True."
        )
    if not name:
        tmp_name = gen_tmp_name(name="df")[1:-1]
    else:
        tmp_name = ""
    sep = "/" if ((temp_path) and temp_path[-1] != "/") else ""
    path = f"{temp_path}{sep}{name}.csv"
    clear = False
    try:
        # Adding the quotes to STR pandas columns in order
        # to simplify the ingestion.
        # Not putting them can lead to wrong data ingestion.
        str_cols, null_columns = [], []
        for c in df.columns:
            if isinstance(df[c].first_valid_index(), NoneType):
                null_columns += [c]
            elif df[c].dtype == object and isinstance(
                df[c].loc[df[c].first_valid_index()], str
            ):
                str_cols += [c]
        if len(df.columns) == len(null_columns):
            names = ", ".join([f"NULL AS {quote_ident(col)}" for col in df.columns])
            q = " UNION ALL ".join([f"(SELECT {names})" for i in range(len(df))])
            return vDataFrame(q)
        if len(str_cols) > 0 or len(null_columns) > 0:
            tmp_df = df.copy()
            for c in str_cols:
                tmp_df[c] = '"' + tmp_df[c].str.replace('"', '""') + '"'
            for c in null_columns:
                tmp_df[c] = '""'
            clear = True
        else:
            tmp_df = df

        tmp_df.to_csv(
            path,
            index=False,
            quoting=csv.QUOTE_NONE,
            escapechar="\027",
            sep="\001",
            lineterminator="\002",
        )

        if len(str_cols) > 0 or len(null_columns) > 0:
            # to_csv is adding an undesired special character
            # we remove it
            logging.debug(f"Replacing undesired characters in" f" csv file {path}")
            with open(path, "r", encoding="utf-8") as f:
                filedata = f.read()
            filedata = filedata.replace(",", ",").replace('""', "")
            with open(path, "w", encoding="utf-8") as f:
                f.write(filedata)
        if insert:
            input_relation = format_schema_table(schema, name)
            tmp_df_columns_str = ", ".join(
                ['"' + col.replace('"', '""') + '"' for col in tmp_df.columns]
            )
            abort_str = "ABORT ON ERROR" if abort_on_error else ""
            sql_string = f"""
                    COPY {input_relation}
                    ({tmp_df_columns_str}) 
                    FROM LOCAL '{path}' 
                    DELIMITER '\001' 
                    NULL ''
                    ENCLOSED BY '\"' 
                    ESCAPE AS '\\' 
                    SKIP 1
                    RECORD TERMINATOR '\002'
                    {abort_str};"""
            logging.debug(f"Copy statement is: {sql_string}")
            _executeSQL(
                query=sql_string,
                title="Inserting the pandas.DataFrame.",
            )
            vdf = vDataFrame(name, schema=schema)
        elif tmp_name:
            vdf = read_csv(
                path,
                table_name=tmp_name,
                dtype=dtype,
                temporary_local_table=True,
                parse_nrows=parse_nrows,
                sep="\001",
                record_terminator="\002",
                escape="\027",
            )
        else:
            vdf = read_csv(
                path,
                table_name=name,
                dtype=dtype,
                schema=schema,
                temporary_local_table=False,
                parse_nrows=parse_nrows,
                sep="\001",
                record_terminator="\002",
                escape="\027",
            )
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        if clear:
            del tmp_df
    return vdf
