"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
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

# Character set used for the intermediate CSV that read_pandas writes and
# COPY reads back. These are control characters so that ordinary text -
# including commas, tabs and double quotes - is never special and never
# needs escaping by us.
#
# The enclosure is deliberately NOT '"'. Vertica's ENCLOSED BY has no notion
# of a doubled quote as an escaped quote, and pandas' to_csv (under
# QUOTE_NONE) will not emit ESCAPE_AS before a quote we added ourselves, so a
# value containing '"' could be neither escaped nor doubled safely - the
# quotes were silently stripped. Enclosing with a control character instead
# makes '"' ordinary data.
# See https://docs.vertica.com/25.3.x/en/data-load/data-formats/delimited-data/
DELIMITER = "\001"
RECORD_TERMINATOR = "\002"
ESCAPE_AS = "\027"

# Candidate enclosure characters, tried in order. COPY accepts any ASCII value
# in E'\001'-E'\177' as ENCLOSED BY, but the character must not occur in the
# data: Vertica requires an enclosure character appearing inside an enclosed
# field to be written as ESCAPE_AS + enclosure (verified against 25.3 - a raw
# or a doubled one makes COPY reject the entire row), and to_csv cannot be
# made to emit that sequence. Under QUOTE_NONE it only ever writes ESCAPE_AS
# ahead of ESCAPE_AS, the delimiter or the record terminator, so no
# pre-escaping survives the write. Choosing a character the data does not
# contain removes the need to escape it at all.
ENCLOSURE_CANDIDATES = (
    "\026",
    "\025",
    "\024",
    "\023",
    "\022",
    "\021",
    "\020",
    "\017",
    "\016",
    "\005",
    "\004",
    "\003",
)


def _pick_enclosure(df: pd.DataFrame, str_cols: list) -> Optional[str]:
    """
    Returns the first ``ENCLOSURE_CANDIDATES`` entry that appears neither in
    any string value nor in any column name, so that enclosed fields need no
    escaping of the enclosure character, or ``None`` if the data contains
    every candidate.
    """
    column_names = "".join(str(col) for col in df.columns)
    for candidate in ENCLOSURE_CANDIDATES:
        if candidate in column_names:
            continue
        if any(
            df[c].str.contains(candidate, regex=False, na=False).any()
            for c in str_cols
        ):
            continue
        return candidate
    return None


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
        Because no type is guessed,
        the fields of the intermediate
        CSV file do not have to be
        enclosed. Specifying ``dtype``
        is therefore the way to ingest
        string columns that contain the
        control characters VerticaPy
        would otherwise use to enclose
        them.
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
            elif (
                # pandas >= 3.0 infers a dedicated string dtype for text
                # columns instead of 'object', so both must be accepted.
                df[c].dtype == object
                or isinstance(df[c].dtype, pd.StringDtype)
            ) and isinstance(df[c].loc[df[c].first_valid_index()], str):
                str_cols += [c]
        if len(df.columns) == len(null_columns):
            names = ", ".join([f"NULL AS {quote_ident(col)}" for col in df.columns])
            q = " UNION ALL ".join([f"(SELECT {names})" for i in range(len(df))])
            if q == "":
                if len(df.columns) > 0:
                    joins = ", ".join(
                        [f"NULL::VARCHAR(64000) AS {col}" for col in df.columns]
                    )
                    q = f"""SELECT  {joins} LIMIT 0"""
                else:
                    raise ValueError(
                        "There are no columns or values. Invalid DataFrame."
                    )
            return vDataFrame(q)
        enclosed_by = _pick_enclosure(df, str_cols)
        if isinstance(enclosed_by, NoneType) and not (insert or dtype):
            # Enclosing the fields is only needed so that the flex table used
            # to guess the column types does not retype a string column - a
            # column of '007' would otherwise be read as INTEGER. Ingesting
            # into an existing relation, or supplying 'dtype', skips that
            # guess, and the data then loads correctly with no enclosure at
            # all: to_csv escapes the delimiter, the record terminator and
            # the escape character, which is all COPY needs.
            raise ValueError(
                "The string columns of the 'pandas.DataFrame' contain every "
                "control character that could be used to enclose the fields "
                "of the intermediate CSV file, so the column types can not "
                "be guessed safely. Specify the column types with the "
                "'dtype' parameter - the enclosing is then unnecessary and "
                "the data is ingested as is - or set 'insert' to True to "
                "load into an existing relation."
            )
        enclose = not isinstance(enclosed_by, NoneType)
        if (enclose and len(str_cols) > 0) or len(null_columns) > 0:
            tmp_df = df.copy()
            if enclose:
                for c in str_cols:
                    # The value itself needs no escaping: 'enclosed_by' was
                    # chosen so that it does not occur in the data, and
                    # to_csv escapes DELIMITER, RECORD_TERMINATOR and
                    # ESCAPE_AS itself. The no-op .str.slice() keeps the
                    # historical handling of a non-string value in an object
                    # column - it becomes NA, where a plain concatenation
                    # would raise TypeError.
                    tmp_df[c] = enclosed_by + tmp_df[c].str.slice() + enclosed_by
            for c in null_columns:
                # An empty, unenclosed field is what COPY's NULL '' matches.
                tmp_df[c] = ""
            clear = True
        else:
            tmp_df = df

        tmp_df.to_csv(
            path,
            index=False,
            quoting=csv.QUOTE_NONE,
            # quotechar=None is required: the enclosures added above are meant
            # to be read back by COPY's ENCLOSED BY. pandas >= 3.0 escapes the
            # quotechar even under QUOTE_NONE, which would turn them into
            # literal data instead. pandas 2.x produces identical output here.
            quotechar=None,
            escapechar=ESCAPE_AS,
            sep=DELIMITER,
            lineterminator=RECORD_TERMINATOR,
        )

        if insert:
            input_relation = format_schema_table(schema, name)
            tmp_df_columns_str = ", ".join(
                ['"' + col.replace('"', '""') + '"' for col in tmp_df.columns]
            )
            abort_str = "ABORT ON ERROR" if abort_on_error else ""
            enclosed_by_str = f"ENCLOSED BY '{enclosed_by}'" if enclose else ""
            sql_string = f"""
                    COPY {input_relation}
                    ({tmp_df_columns_str})
                    FROM LOCAL '{path}'
                    DELIMITER '{DELIMITER}'
                    NULL ''
                    {enclosed_by_str}
                    ESCAPE AS '{ESCAPE_AS}'
                    SKIP 1
                    RECORD TERMINATOR '{RECORD_TERMINATOR}'
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
                sep=DELIMITER,
                record_terminator=RECORD_TERMINATOR,
                quotechar=enclosed_by,
                escape=ESCAPE_AS,
            )
        else:
            vdf = read_csv(
                path,
                table_name=name,
                dtype=dtype,
                schema=schema,
                temporary_local_table=False,
                parse_nrows=parse_nrows,
                sep=DELIMITER,
                record_terminator=RECORD_TERMINATOR,
                quotechar=enclosed_by,
                escape=ESCAPE_AS,
            )
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        if clear:
            del tmp_df
    return vdf
