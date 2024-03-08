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
import datetime
import logging
import os
from pathlib import Path
import tarfile
from typing import Set, List, Mapping

import pandas as pd


from verticapy._utils._sql._sys import _executeSQL
from verticapy.core.vdataframe import vDataFrame
from verticapy.core.parsers.pandas import read_pandas

from verticapy.performance.vertica.collection.collection_tables import (
    AllTableTypes,
    BundleVersion,
    CollectionTable,
    ExportMetadata,
    getAllCollectionTables,
)


class ProfileExportError(Exception):
    """
    ProfileExportError is an exception that can be raised by the ProfileExport class.

    Example
    ------------

    First, let's import the ``ProfileExport`` object and ProfileExportError

    .. code-block:: python

        from verticapy.performance.vertica.collection.profile_export import ProfileExport

    Now we can create a new instance of ``ProfileExport``

    .. code-block:: python

        exporter = ProfileExport(target_schema="missing_tables_schema",
                                key="example123",
                                filename="example_001.tar")

    In our scenario, the schema ``missing_tables_schema`` lacks a replica of the
    ``dc_requests_issued`` system table.

    Next we will try to export the tables. We expect export to fail because we
    lack a system table.

    .. code-block:: python

        try:
            exporter.export()
        except ProfileExportError as err:
            print(f"Observed a ProfileExport error")

    The output will be:

    .. code-block::

        Observed a ProfileExport error

    """

    pass


class ProfileExport:
    """
    The profile ``ProfileExport`` class provides backend methods for
    creating an export bundle of parquet files. ``ProfileExport``
    produces an export bundle of parquet files from a set of replica
    tables created by an instance of the
    :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler` class.

    The export bundle is a tarball. Inside the tarball there are:
        * ``profile_meta.json``, a file with some information about
          the other files in the tarball
        * Several ``.parquet`` files. There is one ``.parquet`` for
          each system table that
          py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
          uses to analyze query performance.
          * For example, there is a file called ``dc_requests_issued.parquet``.

    .. note::
        Many high-level export use cases can be addressed with the method
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler.export_profile()`.
        In particular, ``export_profile()`` is suitable for many cases that do not require
        fine-grained access to the behavior of the export process.

    Examples
    --------

    First, let's import the
    :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
    object and the ``ProfileExport`` object.

    .. code-block:: python

        from verticapy.performance.vertica import QueryProfiler
        from verticapy.performance.vertica.collection.profile_export import ProfileExport

    Now we can profile a query and create a set of system table replicas
    by calling the ``QueryProfiler`` constructor:

    .. code-block:: python

        qprof = QueryProfiler(
            "select transaction_id, statement_id, request, request_duration"
            " from query_requests where start_timestamp > now() - interval'1 hour'"
            " order by request_duration desc limit 10;",
            target_schema="replica_001",
            key_id="example123"
        )

    The parameter ``target_schema`` tells the QueryProfiler to create a
    set of replica tables. The parameter ``key_id`` specifies a suffix for all
    of the replica tables associated with this profile. The replica tables are
    a snapshot of the system tables. The replica tables are filtered to contain
    only the information relevant to the query that we have profiled.

    Now we can use ``ProfileExport`` to produce an export bundle. We need to
    specify which replica tables to export by providing the ``target_schema``
    and ``key_id`` parameters. We choose to name our export bundle
    ``"query_requests_example_001.tar"``.

    .. code-block:: python

        exporter = ProfileExport(target_schema="replica_001",
                                 key="example123",
                                 filename="query_requests_example_001.tar")
        exporter.export()


    After producing an export bundle, we can examine the file contents using
    any tool that read tar-format files. For instance, we can use the tarfile
    library to print the names of all files in the tarball

    .. code-block:: python

        tfile = tarfile.open("query_requests_example_001.tar")
        for f in tfile.getnames():
            print(f"Tarball contains path: {f}")

    The output will be:

    .. code-block::

        Tarball contains path: dc_explain_plans.parquet,
        Tarball contains path: dc_query_executions.parquet
        ...

    """

    def __init__(
        self,
        target_schema: str,
        key: str,
        filename: Path,
    ) -> None:
        """
        Initializes a ``ProfileExport`` object by assigning values
        to instance members. Initialization does not query the
        database or file system.

        .. note::
            Many high-level export use cases can be addressed with the method
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler.export_profile()`.
            In particular, ``export_profile()`` is suitable for many cases that do not require
            fine-grained access to the behavior of the export process.


        Parameters
        -----------------
        target_schema: str
            The schema containing replica system tables created by
            an instance of the
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler` class.
        key: str
            The suffix of the replica system tables in ``target_schema``. key acts
            as a namespace for tables within the schema.
        filename: Path
            The name of the export bundle that ``ProfileExport`` will create.

        Examples
        ---------------

        First, let's import the ``ProfileExport`` object.

        .. code-block:: python

            from verticapy.performance.vertica.collection.profile_export import ProfileExport

        Now we can create a new instance of ``ProfileExport``

        .. code-block:: python

            exporter = ProfileExport(target_schema="replica_001",
                                 key="example123",
                                 filename="query_requests_example_001.tar")

        We can see the values were set by the constructor

        .. code-block:: python

            print(f"Target schema is {exporter.target_schema}\n"
                  f"Key is {exporter.key}\n"
                  f"filename is {exporter.filename}")

        The output will be:

        .. code-block::

            Target schema is replica_001
            Key is example123
            filename is query_requests_example_001.tar

        """
        if not isinstance(target_schema, str):
            raise TypeError(
                f"Expected target_schema to have type str but found type {type(target_schema)}"
            )
        self.target_schema = target_schema
        self.key = key
        self.filename = filename if isinstance(filename, Path) else Path(filename)
        self.logger = logging.getLogger("ProfileExport")

        # Usually the version of export will be "LATEST"
        self.bundle_version = BundleVersion.LATEST

        self.tmp_path = os.getcwd()

    @property
    def tmp_path(self) -> Path:
        """
        Returns the  value of tmp_path.

        ``ProfileExport`` creates temporary directories in ``tmp_path``
        when packing bundles.

        Examples
        ---------------

        First, let's import the ``ProfileExport`` object.

        .. code-block:: python

            from verticapy.performance.vertica.collection.profile_export import ProfileExport

        Now we can create a new instance of ``ProfileExport``

        .. code-block:: python

            exporter = ProfileExport(target_schema="replica_001",
                                 key="example123",
                                 filename="query_requests_example_001.tar")

        Now we get the value of tmp_path:

        .. code-block:: python

            print(f"Temp path = {exporter.tmp_path}")

        The output will be the current working directory. For this example,
        let's assume the current working directory is ``/home/u1``.

        .. code-block::

            Temp path = /home/u1

        """
        return self._tmp_path

    @tmp_path.setter
    def tmp_path(self, val: os.PathLike) -> None:
        """
        Sets the value of tmp_path and returns None

        ``ProfileExport`` creates temporary directories in ``tmp_path``
        when packing bundles.

         Examples
        ---------------

        First, let's import the ``ProfileExport`` object.

        .. code-block:: python

            from verticapy.performance.vertica.collection.profile_export import ProfileExport

        Now we can create a new instance of ``ProfileExport``

        .. code-block:: python

            exporter = ProfileExport(target_schema="replica_001",
                                 key="example123",
                                 filename="query_requests_example_001.tar")

        We can set the value of tmp_path as follows:

        .. code-block:: python

            exporter.tmp_path = "/tmp"

        Now we get the value of tmp_path:

        .. code-block:: python

            print(f"Temp path = {exporter.tmp_path}")

        The output will be:

        .. code-block::

            Temp path = /tmp

        """
        if not isinstance(val, str) and not isinstance(val, os.PathLike):
            raise TypeError(
                f"Cannot set tmp_dir to value of type {type(val)}. Must be type string or PathLike."
            )
        self._tmp_path = val

    def export(self) -> None:
        """
        ``export()`` produces an export bundle of parquet files from a set of replica
        tables created by an instance of the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler` class.

        The export bundle is a tarball. Inside the tarball there are:
            * ``profile_meta.json``, a file with some information about
            the other files in the tarball
            * Several ``.parquet`` files. There is one ``.parquet`` for
            each system table that
            py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
            uses to analyze query performance.
            * For example, there is a file called ``dc_requests_issued.parquet``.

        .. note::
            Many high-level export use cases can be addressed with the method
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler.export_profile()`.
            In particular, ``export_profile()`` is suitable for many cases that do not require
            fine-grained access to the behavior of the export process.

        Examples
        --------

        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object and the ``ProfileExport`` object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler
            from verticapy.performance.vertica.collection.profile_export import ProfileExport

        Now we can profile a query and create a set of system table replicas
        by calling the ``QueryProfiler`` constructor:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;",
                target_schema="replica_001",
                key_id="example123"
            )

        Now we can use ``ProfileExport`` to produce an export bundle. We need to
        specify which replica tables to export by providing the ``target_schema``
        and ``key_id`` parameters. We choose to name our export bundle
        ``"query_requests_example_001.tar"``.

        .. code-block:: python

            exporter = ProfileExport(target_schema="replica_001",
                                    key="example123",
                                    filename="query_requests_example_001.tar")

        We use the ``export()`` function to run the export process as follows:

        .. code-block:: python

            exporter.export()

        The ``export()`` function will confirm that all expected tables are
        present in ``target_schema`` with suffix ``key_id``. Then it will export
        each table to a pandas dataframe, and that dataframe will be saved to a
        file.

        After producing an export bundle, we can examine the file contents using
        any tool that read tar-format files. For instance, we can use the tarfile
        library to print the names of all files in the tarball

        .. code-block:: python

            tfile = tarfile.open("query_requests_example_001.tar")
            for f in tfile.getnames():
                print(f"Tarball contains path: {f}")

        The output will be:

        .. code-block::

            Tarball contains path: dc_explain_plans.parquet,
            Tarball contains path: dc_query_executions.parquet
            ...
        """
        self._tables_exist_or_raise()
        export_metadata = self._save_tables()
        self._bundle_tables(export_metadata)

    def _tables_exist_or_raise(self) -> None:
        """
        Queries the database looking for an existing
        set of tables and compares it to the expected set of tables.
        If an expected tables does not exist, the method
        will raise a ``ProfileExportError``.

        .. note::
           ``_tables_exist_or_raise`` is an internal function
           for the ``ProfileExport`` class. It should not be used
           by external callers.

        Parameters
        ----------------
        None

        Returns
        ------------------
        None

        """
        tables_in_schema = self._get_set_of_tables_in_schema()

        missing_tables = []

        all_tables = getAllCollectionTables(
            self.target_schema, self.key, self.bundle_version
        )
        for ctable in all_tables.values():
            search_name = ctable.get_import_name()
            if search_name not in tables_in_schema:
                missing_tables.append(search_name)
        if len(missing_tables) == 0:
            return

        # At least one table is missing
        # Throw an execption
        raise ProfileExportError(
            f"Missing {len(missing_tables)} tables"
            f" in schema {self.target_schema}."
            f" Missing: [ {','.join(missing_tables)} ]"
        )

    def _get_set_of_tables_in_schema(self) -> Set[str]:
        """
        Queries the database looking to produce an set of table
        names in a schema.

        .. note::
           ``_get_set_of_tables_in_schema`` is an internal function
           for the ``ProfileExport`` class. It should not be used
           by external callers.

        Parameters
        ----------------
        None

        Returns
        ------------------
        A ``set`` containing table names. The table names do not
        fully-qualified, that is, they do not contain the database
        name and schema name.

        """
        result = _executeSQL(
            f"""SELECT table_name FROM v_catalog.tables 
                    WHERE 
                        table_schema = '{self.target_schema}'
                        and table_name ilike '%_{self.key}';
                    """,
            method="fetchall",
        )
        existing_tables = set()
        for row in result:
            existing_tables.add(row[0])
        return existing_tables

    def _save_tables(self) -> ExportMetadata:
        """
        Exports the replica tables to parquet files, retrieves the
        table metadata and packages it into export metadata.
        Calls the ``export_table`` method on  each ``CollectionTable``
        object. Writes the export metadata to disk.

        .. note::
           ``_save_tables`` is an internal function
           for the ``ProfileExport`` class. It should not be used
           by external callers.

        Parameters
        ----------------
        None

        Returns
        ------------------
        A ``ExportMetadata`` instance containing export information.

        """
        all_tables = getAllCollectionTables(
            self.target_schema, self.key, self.bundle_version
        )
        tmp_path = self.filename.parent
        all_table_metadata = []
        for ctable in all_tables.values():
            exported_meta = ctable.export_table(tmp_path)
            all_table_metadata.append(exported_meta)

        metadata_file = tmp_path / "profile_metadata.json"
        export_metadata = ExportMetadata(
            metadata_file, self.bundle_version, all_table_metadata
        )

        export_metadata.write_to_file()

        return export_metadata

    def _bundle_tables(self, export_metadata: ExportMetadata) -> None:
        """
        Packages the parquet files and metadata file into a tarball.
        Then removes the parquet files and metadata files.

        .. note::
           ``_bundle_tables`` is an internal function
           for the ``ProfileExport`` class. It should not be used
           by external callers.

        Parameters
        ----------------
        export_metadata: ExportMetadata
            The ExportMetadata object produced by internal function
            ``_save_tables``

        Returns
        ------------------
        None

        """

        self.tarfile_obj = tarfile.open(self.filename, "w")
        for t in export_metadata.tables:
            self.tarfile_obj.add(t.file_name, arcname=t.file_name.name)
        self.tarfile_obj.add(
            export_metadata.file_name, arcname=export_metadata.file_name.name
        )
        self.tarfile_obj.close()
        for t in export_metadata.tables:
            os.remove(t.file_name)

        # Everything went ok, clean it up
        os.remove(export_metadata.file_name)
