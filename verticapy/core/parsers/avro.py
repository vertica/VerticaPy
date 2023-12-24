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
from typing import Optional

from verticapy._utils._sql._collect import save_verticapy_logs

from verticapy.core.parsers.json import read_json
from verticapy.core.vdataframe.base import vDataFrame


@save_verticapy_logs
def read_avro(
    path: str,
    schema: Optional[str] = None,
    table_name: Optional[str] = None,
    usecols: Optional[list] = None,
    new_name: Optional[dict] = None,
    insert: bool = False,
    reject_on_materialized_type_error: bool = False,
    flatten_maps: bool = True,
    flatten_arrays: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    gen_tmp_table_name: bool = True,
    ingest_local: bool = True,
    genSQL: bool = False,
    materialize: bool = True,
    use_complex_dt: bool = False,
) -> vDataFrame:
    """
    Ingests an AVRO file
    using flex tables.

    Parameters
    ----------
    path: str
        Absolute path where the AVRO file is located.
    schema: str, optional
        Schema  where the AVRO file will be ingested.
    table_name: str, optional
        Final relation name.
    usecols: list, optional
        ``list``  of the  AVRO parameters to ingest.  The
        other ones will be ignored.  If empty, all the
        AVRO parameters will be ingested.
    new_name: dict, optional
        Dictionary of the new column
        names. If the AVRO file is
        nested, it is recommended to
        change the final names because
        special characters will be
        included in the new column names.
        For example,
        ``{"param": {"age": 3, "name": Badr}, "date": 1993-03-11}``
        will create 3 columns: "param.age",
        "param.name" and "date".  You can
        rename these columns using the
        ``new_name`` parameter with the
        following ``dictionary``:
        ``{"param.age": "age", "param.name": "name"}``
    insert: bool, optional
        If set to ``True``, the  data will be ingested to the
        input relation.  The AVRO  parameters must be the
        same  as the input  relation otherwise they  will
        not be ingested. If set to ``True``, ``table_name`` cannot
        be empty.
    reject_on_materialized_type_error: bool, optional
        ``boolean``, whether to reject
        a data row that contains a
        materialized column value that
        cannot be coerced into a
        compatible data type. If the
        value is ``False`` and the
        type cannot be coerced, the
        parser sets the value in that
        column to ``None``.  If the
        column is a strongly-typed
        complex type, as opposed to a
        flexible complex type, then a
        type mismatch anywhere in the
        complex type causes the entire
        column to be treated as a mismatch.
        The parser does not partially
        load complex types.
    flatten_maps: bool, optional
        ``boolean``, whether to
        flatten sub-maps within
        the AVRO data, separating
        map levels  with a period
        (.). This value affects
        all data in the load,
        including nested maps.
    flatten_arrays: bool, optional
        ``boolean``,  whether to
        convert lists to sub-maps
        with ``integer`` keys.
        When lists are flattened,
        key names are concatenated
        in the same way as maps.
        ``lists`` are not flattened
        by default. This value affects
        all data  in the load,
        including nested ``lists``.
    temporary_table: bool, optional
        If set to ``True``, a
        temporary table will be
        created.
    temporary_local_table: bool, optional
        If set to ``True``, a
        temporary  local table
        will be created. The
        parameter ``schema``
        must be empty, otherwise
        this parameter is ignored.
    gen_tmp_table_name: bool, optional
        Sets the name of the temporary
        table. This  parameter is only
        used when the parameter
        ``temporary_local_table`` is
        set to ``True`` and if the
        parameters ``table_name`` and
        ``schema`` are unspecified.
    ingest_local: bool, optional
        If set to ``True``, the file
        will be ingested from the
        local machine.
    genSQL: bool, optional
        If set to ``True``, the SQL
        code for creating the final
        table is generated but not
        executed. This is a good way
        to change the  final relation
        types or to customize the
        data ingestion.
    materialize: bool, optional
        If set to ``True``, the flex
        table is materialized into a
        table. Otherwise, it will
        remain a flex table. Flex
        tables simplify the data
        ingestion but have worse
        performace compared to
        regular tables.
    use_complex_dt: bool, optional
        ``boolean``, whether the input
        data file has complex structure.
        If set to ``True``, most of the
        other parameters are ignored.

    Returns
    -------
    vDataFrame
        The :py:class:`~vDataFrame`
        of the relation.

    Examples
    --------
    In this example, we will first download
    an *AVRO* file and then ingest it
    into Vertica database.

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

    Let's download the AVRO file.

    .. ipython:: python
        :okexcept:

        import requests
        url = "https://github.com/vertica/VerticaPy/raw/master/verticapy/tests/utilities/variants.avro"
        r = requests.get(url)
        open('variants.avro', 'wb').write(r.content)

    Let's ingest the AVRO file
    into the Vertica database.

    .. code-block:: python

        from verticapy.core.parsers.avro import read_avro

        read_avro(
            path = "variants.avro",
            table_name = "variants",
            schema = "public",
        )

    .. ipython:: python
        :suppress:
        :okexcept:

        from verticapy.core.parsers.avro import read_avro

        res = read_avro(
            path = "variants.avro",
            table_name = "variants",
            schema = "public",
        )
        html_file = open("figures/core_parsers_avro1.html", "w")
        html_file.write(res._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_parsers_avro1.html

    Let's ingest only two
    columns.

    .. code-block:: python

        read_avro(
            path = "variants.avro",
            table_name = "variants_usecols",
            schema = "public",
            usecols  = [
                "type",
                "sv",
            ],
        )

    .. ipython:: python
        :suppress:
        :okexcept:

        res = read_avro(
            path = "variants.avro",
            table_name = "variants_usecols",
            schema = "public",
            usecols  = [
                "type",
                "sv",
            ],
        )
        html_file = open("figures/core_parsers_avro2.html", "w")
        html_file.write(res._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_parsers_avro2.html

    .. note::

        You can ingest multiple AVRO
        files into the Vertica database
        by using the following syntax.

        .. code-block:: python

            read_avro(
                path = "*.avro",
                table_name = "variants_multi_files",
                schema = "public",
            )

    .. ipython:: python
        :suppress:
        :okexcept:

        #Cleanup block - drop / remove objects created for this example
        from verticapy import drop
        drop(name = "public.variants")
        drop(name = "public.variants_usecols")
        import os
        os.remove("variants.avro")

    .. seealso::

        | :py:func:`~verticapy.read_csv` :
            Ingests a CSV file into the Vertica DB.
        | :py:func:`~verticapy.read_file` :
            Ingests an input file into the Vertica DB.
        | :py:func:`~verticapy.read_json` :
            Ingests a JSON file into the Vertica DB.
        | :py:func:`~verticapy.read_pandas` :
            Ingests the ``pandas.DataFrame``
            into the Vertica DB.
    """
    return read_json(
        **locals(),
        is_avro=True,
    )
