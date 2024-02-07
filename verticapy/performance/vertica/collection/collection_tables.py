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
from abc import abstractmethod
from enum import Enum
from typing import Mapping


class AllTableTypes(Enum):
    """
    Enumeration (``Enum``) of all table types understood by profile collection.

    It is best to match table schema (col types) by comparing to this enumeration.
    Tables can have the same schema and different names.
    """

    COLLECTION_EVENTS = "collection_events"
    COLLECTION_INFO = "collection_info"
    DC_EXPLAIN_PLANS = "dc_explain_plans"
    DC_QUERY_EXECUTIONS = "dc_query_executions"
    DC_REQUESTS_ISSUED = "dc_requests_issued"
    EXECUTION_ENGINE_PROFILES = "execution_engine_profiles"
    EXPORT_EVENTS = "export_events"
    HOST_RESOURCES = "host_resources"
    QUERY_CONSUMPTION = "query_consumption"
    QUERY_PLAN_PROFILES = "query_plan_profiles"
    QUERY_PROFILES = "query_profiles"
    RESOURCE_POOL_STATUS = "resource_pool_status"


class CollectionTable:
    """
    ``CollectionTable`` is the abstract parent class for tables created by query profile export.

    Child classes are expected to implement abstract methods
        - ``get_create_table_sql()``
        - ``get_create_projection_sql()``
        - ``has_copy_staging()``
        - ``copy_from_local_file()``
    """

    def __init__(self, table_type: AllTableTypes, table_schema: str, key: str) -> None:
        self.table_type = table_type
        self.name = self.table_type.value
        self.schema = table_schema
        self.key = key
        self.import_prefix = "qprof_"
        self.import_suffix = f"_{key}"
        self.staging_suffix = "_staging"

    def get_import_name_fq(self) -> str:
        return self._get_import_name(fully_qualified=True)

    def get_import_name(self) -> str:
        return self._get_import_name(fully_qualified=False)

    def get_super_proj_name_fq(self) -> str:
        return f"{self._get_import_name(fully_qualified=True)}_super"

    def _get_import_name(self, fully_qualified) -> str:
        return (
            f"{self.schema}.{self.import_prefix}{self.name}{self.import_suffix}"
            if fully_qualified
            else f"{self.import_prefix}{self.name}{self.import_suffix}"
        )

    def get_staging_name(self) -> str:
        return f"{self.schema}.{self.import_prefix}{self.name}{self.staging_suffix}{self.import_suffix}"

    def get_drop_table_sql(self) -> str:
        # Drop table can be in the base class because it is
        # DROP TABLE IF EXISTS {self.get_import_name_fq}
        raise NotImplementedError(f"Drop table not implemented yet")

    # Recall: abstract methods won't raise by default
    @abstractmethod
    def get_create_table_sql(self) -> str:
        raise NotImplementedError(
            f"get_create_table_sql is not implemented in the base class CollectionTable."
            f" Current table name = {self.name} schema {self.schema}"
        )

    @abstractmethod
    def get_create_projection_sql(self) -> str:
        raise NotImplementedError(
            f"get_create_projection_sql is not implemented in the base class CollectionTable"
            f" Current table name = {self.name} schema {self.schema}"
        )

    @abstractmethod
    def has_copy_staging(self) -> str:
        raise NotImplementedError(
            f"has_copy_staging is not implemented in the base class CollectionTable"
            f" Current table name = {self.name} schema {self.schema}"
        )

    @abstractmethod
    def copy_from_local_file(self, filename: str) -> str:
        # This method will generate and run the copy statement
        # It will copy to a staging table when necessary
        raise NotImplementedError(
            f"copy_from_local_file is not implemented in base class CollectionTable"
            f"Current table name = {self.name} and schema {self.schema}"
        )


def getAllCollectionTables(
    target_schema: str, key: str
) -> Mapping[str, CollectionTable]:
    """
    Produces a map with one of each kind of collection table. The key
    to the map is the table name.

    Returns
    ----------
    A ``dictionary`` of ``CollectionTables``.

    The ``dictionary`` contains one collection table instance for each type in
    ``AllTableTypes``. The key is the string value of the ``AllTableTypes`` enum.

    Parameters
    ----------
    target_schema: str
        The schema for all tables.

    key: str
        A suffix for all table names to help uniquely identify the table.
    """
    result = {}
    for name in AllTableTypes:
        c = collectionTableFactory(name, target_schema, key)
        result[name.name] = c

    return result


def collectionTableFactory(
    table_type: AllTableTypes, target_schema: str, key: str
) -> CollectionTable:
    """
    ``collectionTableFactory`` implements the constructor pattern. It
    instantiates the appropriate subclass of ``CollectionTable`` for
    the parameter ``table_type``.

    Returns
    ----------
    An instance of a CollectionTable for a given table_type.

    Parameters
    ------------
    table_type: AllTableTypes
        The type of table collectionTableFactory should create.

    target_schema: str
        The schema for the table.

    key: str
        The suffix for the table name.
    """
    if table_type == AllTableTypes.COLLECTION_EVENTS:
        return CollectionEventsTable(target_schema, key)
    if table_type == AllTableTypes.COLLECTION_INFO:
        return CollectionInfoTable(target_schema, key)
    if table_type == AllTableTypes.DC_EXPLAIN_PLANS:
        return DCExplainPlansTable(target_schema, key)
    if table_type == AllTableTypes.DC_QUERY_EXECUTIONS:
        return DCQueryExecutionsTable(target_schema, key)
    if table_type == AllTableTypes.DC_REQUESTS_ISSUED:
        return DCRequestsIssuedTable(target_schema, key)
    if table_type == AllTableTypes.EXECUTION_ENGINE_PROFILES:
        return ExecutionEngineProfilesTable(target_schema, key)
    if table_type == AllTableTypes.EXPORT_EVENTS:
        return ExportEventsTable(target_schema, key)
    if table_type == AllTableTypes.HOST_RESOURCES:
        return HostResourcesTable(target_schema, key)
    if table_type == AllTableTypes.QUERY_CONSUMPTION:
        return QueryConsumptionTable(target_schema, key)
    if table_type == AllTableTypes.QUERY_PLAN_PROFILES:
        return QueryPlanProfilesTable(target_schema, key)
    if table_type == AllTableTypes.QUERY_PROFILES:
        return QueryProfilesTable(target_schema, key)
    if table_type == AllTableTypes.RESOURCE_POOL_STATUS:
        return ResourcePoolStatusTable(target_schema, key)

    raise ValueError(f"Unrecognized table type {table_type}.")


############## collection_events ######################
class CollectionEventsTable(CollectionTable):
    """
    ``CollectionEventsTable`` stores data produced during the ProfileCollection process.
    It is not a vertica system table, but rather a diagnostics table produced
    while profiling queries.
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.COLLECTION_EVENTS, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            transaction_id int,
            statement_id int,
            table_name varchar(256),
            operation varchar(128),
            row_count int
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name} 
        /*+basename({import_name}),createtype(L)*/
        (
            transaction_id,
            statement_id,
            table_name,
            operation,
            row_count
        )
        AS
        SELECT {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.table_name,
                {import_name}.operation,
                {import_name}.row_count
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
                {import_name}.statement_id
        SEGMENTED BY hash({import_name}.transaction_id, {import_name}.statement_id) ALL NODES;
        """


############## collection_info ######################
class CollectionInfoTable(CollectionTable):
    """
    ``CollectionInfoTable`` stores information about the profile collection,
    such as the vertica version and a some comments about the queries being profiled.
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.COLLECTION_INFO, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            transaction_id int,
            statement_id int,
            user_query_label varchar(256),
            user_query_comment varchar(512),
            project_name varchar(128),
            customer_name varchar(128),
            -- Note that this should have 
            -- DEFAULT version() during collection
            version varchar(512)
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name} 
        /*+basename({import_name}),createtype(L)*/
        (
            transaction_id encoding rle,
            statement_id encoding rle,
            user_query_label encoding rle,
            user_query_comment encoding rle,
            project_name encoding rle,
            customer_name encoding rle,
            version encoding rle
        )
        AS
        SELECT {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.user_query_label,
                {import_name}.user_query_comment,
                {import_name}.project_name,
                {import_name}.customer_name,
                {import_name}.version
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.user_query_label,
                {import_name}.user_query_comment,
                {import_name}.project_name,
                {import_name}.customer_name
        SEGMENTED BY hash({import_name}.transaction_id, 
                        {import_name}.statement_id, 
                        {import_name}.user_query_label) 
        ALL NODES;
        """


########### dc_explain_plans ######################
class DCExplainPlansTable(CollectionTable):
    """
    ``DCExplainPlansTable`` stores data from the system table
    dc_explain_plans
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.DC_EXPLAIN_PLANS, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            "time" timestamptz,
            node_name varchar(128),
            session_id varchar(128),
            user_id int,
            user_name varchar(128),
            transaction_id int,
            statement_id int,
            request_id int,
            path_id int,
            path_line_index int,
            path_line varchar(64000),
            query_name varchar(128)
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name} 
        /*+basename({import_name}),createtype(A)*/
        (
            "time",
            node_name,
            session_id,
            user_id,
            user_name,
            transaction_id,
            statement_id,
            request_id,
            path_id,
            path_line_index,
            path_line,
            query_name
        )
        AS
        SELECT {import_name}."time",
                {import_name}.node_name,
                {import_name}.session_id,
                {import_name}.user_id,
                {import_name}.user_name,
                {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.request_id,
                {import_name}.path_id,
                {import_name}.path_line_index,
                {import_name}.path_line,
                {import_name}.query_name
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.node_name,
                {import_name}."time",
                {import_name}.session_id,
                {import_name}.user_id,
                {import_name}.user_name,
                {import_name}.request_id
        SEGMENTED BY hash({import_name}."time", 
                {import_name}.user_id,
                {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.request_id,
                {import_name}.path_id,
                {import_name}.path_line_index, 
                {import_name}.node_name) 
        ALL NODES;
        """


################ dc_query_executions ###################
class DCQueryExecutionsTable(CollectionTable):
    """
    ``DCQueryExecutionsTable`` stores data from the system table
    dc_query_executions
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.DC_QUERY_EXECUTIONS, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            "time" timestamptz,
            node_name varchar(128),
            session_id varchar(128),
            user_id int,
            user_name varchar(128),
            transaction_id int,
            statement_id int,
            request_id int,
            execution_step varchar(128),
            completion_time timestamptz,
            query_name varchar(128)
        );

        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name}
        /*+basename({import_name}),createtype(A)*/
        (
            "time",
            node_name,
            session_id,
            user_id,
            user_name,
            transaction_id,
            statement_id,
            request_id,
            execution_step,
            completion_time,
            query_name
        )
        AS
        SELECT {import_name}."time",
            {import_name}.node_name,
            {import_name}.session_id,
            {import_name}.user_id,
            {import_name}.user_name,
            {import_name}.transaction_id,
            {import_name}.statement_id,
            {import_name}.request_id,
            {import_name}.execution_step,
            {import_name}.completion_time,
            {import_name}.query_name
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
            {import_name}.statement_id,
            {import_name}.node_name,
            {import_name}."time",
            {import_name}.request_id,
            {import_name}.session_id,
            {import_name}.user_id,
            {import_name}.user_name
        SEGMENTED BY hash({import_name}."time",
            {import_name}.user_id,
            {import_name}.transaction_id,
            {import_name}.statement_id, 
            {import_name}.request_id,
            {import_name}.completion_time, 
            {import_name}.node_name,
            {import_name}.session_id)
        ALL NODES;
        """


################ dc_requests_issued ###################
class DCRequestsIssuedTable(CollectionTable):
    """
    ``DCRequestsIssuedTable`` stores data from the system table
    dc_requests_issued
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.DC_REQUESTS_ISSUED, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            "time" timestamptz,
            node_name varchar(128),
            session_id varchar(128),
            user_id int,
            user_name varchar(128),
            transaction_id int,
            statement_id int,
            request_id int,
            request_type varchar(128),
            label varchar(128),
            client_label varchar(64000),
            search_path varchar(64000),
            query_start_epoch int,
            request varchar(64000),
            is_retry boolean,
            digest int,
            query_name varchar(128)
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name}
        /*+basename({import_name}),createtype(A)*/
        (
            "time",
            node_name,
            session_id,
            user_id,
            user_name,
            transaction_id,
            statement_id,
            request_id,
            request_type,
            label,
            client_label,
            search_path,
            query_start_epoch,
            request,
            is_retry,
            digest,
            query_name
        )
        AS
        SELECT {import_name}."time",
            {import_name}.node_name,
            {import_name}.session_id,
            {import_name}.user_id,
            {import_name}.user_name,
            {import_name}.transaction_id,
            {import_name}.statement_id,
            {import_name}.request_id,
            {import_name}.request_type,
            {import_name}.label,
            {import_name}.client_label,
            {import_name}.search_path,
            {import_name}.query_start_epoch,
            {import_name}.request,
            {import_name}.is_retry,
            {import_name}.digest,
            {import_name}.query_name
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
            {import_name}.statement_id,
            {import_name}.node_name,
            {import_name}.label,
            {import_name}.request_id
        SEGMENTED BY hash({import_name}."time", 
            {import_name}.user_id, 
            {import_name}.transaction_id, 
            {import_name}.statement_id, 
            {import_name}.request_id, 
            {import_name}.query_start_epoch, 
            {import_name}.is_retry, 
            {import_name}.digest) 
        ALL NODES;
        """


################ execution_engine_profiles ###################
class ExecutionEngineProfilesTable(CollectionTable):
    """
    ``ExecutionEngineProfilesTable`` stores data from the system view
    execution_engine_profiles
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.EXECUTION_ENGINE_PROFILES, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            node_name varchar(128),
            user_id int,
            user_name varchar(128),
            session_id varchar(128),
            transaction_id int,
            statement_id int,
            plan_id int,
            operator_name varchar(128),
            operator_id int,
            baseplan_id int,
            path_id int,
            localplan_id int,
            activity_id int,
            resource_id int,
            counter_name varchar(128),
            counter_tag varchar(128),
            counter_value int,
            is_executing boolean,
            query_name varchar(128)
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name}
        /*+basename({import_name}),createtype(A)*/
        (
            node_name,
            user_id,
            user_name,
            session_id,
            transaction_id,
            statement_id,
            plan_id,
            operator_name,
            operator_id,
            baseplan_id,
            path_id,
            localplan_id,
            activity_id,
            resource_id,
            counter_name,
            counter_tag,
            counter_value,
            is_executing,
            query_name
        )
        AS
        SELECT {import_name}.node_name,
            {import_name}.user_id,
            {import_name}.user_name,
            {import_name}.session_id,
            {import_name}.transaction_id,
            {import_name}.statement_id,
            {import_name}.plan_id,
            {import_name}.operator_name,
            {import_name}.operator_id,
            {import_name}.baseplan_id,
            {import_name}.path_id,
            {import_name}.localplan_id,
            {import_name}.activity_id,
            {import_name}.resource_id,
            {import_name}.counter_name,
            {import_name}.counter_tag,
            {import_name}.counter_value,
            {import_name}.is_executing,
            {import_name}.query_name
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
          {import_name}.statement_id,
          {import_name}.node_name,
          {import_name}.plan_id,
          {import_name}.path_id,
          {import_name}.operator_id
        SEGMENTED BY hash({import_name}.user_id,
            {import_name}.transaction_id,
            {import_name}.statement_id, 
            {import_name}.plan_id, 
            {import_name}.operator_id, 
            {import_name}.baseplan_id, 
            {import_name}.path_id, 
            {import_name}.localplan_id) 
        ALL NODES;
        """


################ export_events ###################
class ExportEventsTable(CollectionTable):
    """
    ``ExportEventsTable`` stores data produced during the ProfileCollection process.
    It is not a vertica system table, but rather a diagnostics table produced
    when exporting profile information.
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.EXPORT_EVENTS, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            table_name varchar(256),
            operation varchar(128),
            row_count int
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name} 
        /*+basename({import_name}),createtype(L)*/
        (
            table_name,
            operation,
            row_count
        )
        AS
        SELECT {import_name}.table_name,
                {import_name}.operation,
                {import_name}.row_count
        FROM {import_name_fq}
        ORDER BY {import_name}.table_name,
                {import_name}.operation,
                {import_name}.row_count
        SEGMENTED BY hash({import_name}.row_count, 
                {import_name}.operation, 
                {import_name}.table_name) 
        ALL NODES;
        """


################ host_resources ###################
class HostResourcesTable(CollectionTable):
    """
    ``HostResourcesTable`` stores a snapshot of data from the host_resources
    system table. It adds two columns to the host_resources table:
    transaction_id and statement_id.
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.HOST_RESOURCES, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            host_name varchar(128),
            open_files_limit int,
            threads_limit int,
            core_file_limit_max_size_bytes int,
            processor_count int,
            processor_core_count int,
            processor_description varchar(8192),
            opened_file_count int,
            opened_socket_count int,
            opened_nonfile_nonsocket_count int,
            total_memory_bytes int,
            total_memory_free_bytes int,
            total_buffer_memory_bytes int,
            total_memory_cache_bytes int,
            total_swap_memory_bytes int,
            total_swap_memory_free_bytes int,
            disk_space_free_mb int,
            disk_space_used_mb int,
            disk_space_total_mb int,
            system_open_files int,
            system_max_files int,
            transaction_id int,
            statement_id int,
            query_name varchar(128)
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name} 
        /*+basename({import_name}),createtype(A)*/
        (
            host_name,
            open_files_limit,
            threads_limit,
            core_file_limit_max_size_bytes,
            processor_count,
            processor_core_count,
            processor_description,
            opened_file_count,
            opened_socket_count,
            opened_nonfile_nonsocket_count,
            total_memory_bytes,
            total_memory_free_bytes,
            total_buffer_memory_bytes,
            total_memory_cache_bytes,
            total_swap_memory_bytes,
            total_swap_memory_free_bytes,
            disk_space_free_mb,
            disk_space_used_mb,
            disk_space_total_mb,
            system_open_files,
            system_max_files,
            transaction_id,
            statement_id,
            query_name
        )
        AS
        SELECT {import_name}.host_name,
                {import_name}.open_files_limit,
                {import_name}.threads_limit,
                {import_name}.core_file_limit_max_size_bytes,
                {import_name}.processor_count,
                {import_name}.processor_core_count,
                {import_name}.processor_description,
                {import_name}.opened_file_count,
                {import_name}.opened_socket_count,
                {import_name}.opened_nonfile_nonsocket_count,
                {import_name}.total_memory_bytes,
                {import_name}.total_memory_free_bytes,
                {import_name}.total_buffer_memory_bytes,
                {import_name}.total_memory_cache_bytes,
                {import_name}.total_swap_memory_bytes,
                {import_name}.total_swap_memory_free_bytes,
                {import_name}.disk_space_free_mb,
                {import_name}.disk_space_used_mb,
                {import_name}.disk_space_total_mb,
                {import_name}.system_open_files,
                {import_name}.system_max_files,
                {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.query_name
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.host_name
        SEGMENTED BY hash({import_name}.open_files_limit,
                {import_name}.threads_limit, 
                {import_name}.core_file_limit_max_size_bytes, 
                {import_name}.processor_count, 
                {import_name}.processor_core_count, 
                {import_name}.opened_file_count, 
                {import_name}.opened_socket_count, 
                {import_name}.opened_nonfile_nonsocket_count) 
        ALL NODES;
        """


################ query_consumption ###################
class QueryConsumptionTable(CollectionTable):
    """
    ``QueryComsumptionTable`` stores data from the system view
    query_consumption.
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.QUERY_CONSUMPTION, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            start_time timestamptz,
            end_time timestamptz,
            session_id varchar(128),
            user_id int,
            user_name varchar(128),
            transaction_id int,
            statement_id int,
            cpu_cycles_us int,
            network_bytes_received int,
            network_bytes_sent int,
            data_bytes_read int,
            data_bytes_written int,
            data_bytes_loaded int,
            bytes_spilled int,
            input_rows int,
            input_rows_processed int,
            peak_memory_kb int,
            thread_count int,
            duration_ms int,
            resource_pool varchar(128),
            output_rows int,
            request_type varchar(128),
            label varchar(128),
            is_retry boolean,
            success boolean,
            query_name varchar(128)
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name} 
        /*+basename({import_name}),createtype(A)*/
        (
            start_time,
            end_time,
            session_id,
            user_id,
            user_name,
            transaction_id,
            statement_id,
            cpu_cycles_us,
            network_bytes_received,
            network_bytes_sent,
            data_bytes_read,
            data_bytes_written,
            data_bytes_loaded,
            bytes_spilled,
            input_rows,
            input_rows_processed,
            peak_memory_kb,
            thread_count,
            duration_ms,
            resource_pool,
            output_rows,
            request_type,
            label,
            is_retry,
            success,
            query_name
        )
        AS 
         SELECT {import_name}.start_time,
            {import_name}.end_time,
            {import_name}.session_id,
            {import_name}.user_id,
            {import_name}.user_name,
            {import_name}.transaction_id,
            {import_name}.statement_id,
            {import_name}.cpu_cycles_us,
            {import_name}.network_bytes_received,
            {import_name}.network_bytes_sent,
            {import_name}.data_bytes_read,
            {import_name}.data_bytes_written,
            {import_name}.data_bytes_loaded,
            {import_name}.bytes_spilled,
            {import_name}.input_rows,
            {import_name}.input_rows_processed,
            {import_name}.peak_memory_kb,
            {import_name}.thread_count,
            {import_name}.duration_ms,
            {import_name}.resource_pool,
            {import_name}.output_rows,
            {import_name}.request_type,
            {import_name}.label,
            {import_name}.is_retry,
            {import_name}.success,
            {import_name}.query_name
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.label,
                {import_name}.start_time,
                {import_name}.end_time
        SEGMENTED BY hash({import_name}.start_time, 
                {import_name}.end_time, 
                {import_name}.user_id, 
                {import_name}.transaction_id, 
                {import_name}.statement_id,
                {import_name}.cpu_cycles_us,
                {import_name}.network_bytes_received, 
                {import_name}.network_bytes_sent)
        ALL NODES;
        """


################ query_plan_profiles ###################
class QueryPlanProfilesTable(CollectionTable):
    """
    ``QueryPlanProfilesTable`` stores data from the system table
    query_plan_profiles.
    """

    def __init__(self, table_schema: str, key: str) -> None:
        # Note: This table requires a staging table
        super().__init__(AllTableTypes.QUERY_PLAN_PROFILES, table_schema, key)

    def get_create_table_sql(self) -> str:
        # running_time is an interval. it needs special treatment
        # because parquet doesn't support intervals
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            transaction_id int,
            statement_id int,
            path_id int,
            path_line_index int,
            path_is_started boolean,
            path_is_completed boolean,
            is_executing boolean,
             /* Now running_time is a interval, which is what qprof queries expect*/
            running_time interval,
            memory_allocated_bytes int,
            read_from_disk_bytes int,
            received_bytes int,
            sent_bytes int,
            path_line varchar(64000),
            query_name varchar(128)
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name} 
        /*+basename({import_name}),createtype(A)*/
        (
            transaction_id,
            statement_id,
            path_id,
            path_line_index,
            path_is_started,
            path_is_completed,
            is_executing,
            running_time,
            memory_allocated_bytes,
            read_from_disk_bytes,
            received_bytes,
            sent_bytes,
            path_line,
            query_name
        )
        AS
        SELECT {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.path_id,
                {import_name}.path_line_index,
                {import_name}.path_is_started,
                {import_name}.path_is_completed,
                {import_name}.is_executing,
                {import_name}.running_time,
                {import_name}.memory_allocated_bytes,
                {import_name}.read_from_disk_bytes,
                {import_name}.received_bytes,
                {import_name}.sent_bytes,
                {import_name}.path_line,
                {import_name}.query_name
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.path_id,
                {import_name}.path_line_index,
                {import_name}.path_is_started,
                {import_name}.path_is_completed,
                {import_name}.is_executing,
                {import_name}.running_time
        SEGMENTED BY hash({import_name}.transaction_id, 
                {import_name}.statement_id, 
                {import_name}.path_id, 
                {import_name}.path_line_index,
                {import_name}.path_is_started,
                {import_name}.path_is_completed,
                {import_name}.is_executing,
                {import_name}.running_time) 
        ALL NODES;
        """


################ query_profiles ###################
class QueryProfilesTable(CollectionTable):
    """
    ``QueryProfilesTable`` stores data from the system table
    query_profiles.
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.QUERY_PROFILES, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.get_import_name_fq()}
        (
            session_id varchar(128),
            transaction_id int,
            statement_id int,
            identifier varchar(128),
            node_name varchar(128),
            query varchar(64000),
            query_search_path varchar(64000),
            schema_name varchar(128),
            table_name varchar(128),
            query_duration_us numeric(36,6),
            query_start_epoch int,
            query_start varchar(63),
            query_type varchar(128),
            error_code int,
            user_name varchar(128),
            processed_row_count int,
            reserved_extra_memory_b int,
            is_executing boolean,
            query_name varchar(128)
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name} /*+basename({import_name}),createtype(A)*/
        (
            session_id,
            transaction_id,
            statement_id,
            identifier,
            node_name,
            query,
            query_search_path,
            schema_name,
            table_name,
            query_duration_us,
            query_start_epoch,
            query_start,
            query_type,
            error_code,
            user_name,
            processed_row_count,
            reserved_extra_memory_b,
            is_executing,
            query_name
        )
        AS  
        SELECT {import_name}.session_id,
            {import_name}.transaction_id,
            {import_name}.statement_id,
            {import_name}.identifier,
            {import_name}.node_name,
            {import_name}.query,
            {import_name}.query_search_path,
            {import_name}.schema_name,
            {import_name}.table_name,
            {import_name}.query_duration_us,
            {import_name}.query_start_epoch,
            {import_name}.query_start,
            {import_name}.query_type,
            {import_name}.error_code,
            {import_name}.user_name,
            {import_name}.processed_row_count,
            {import_name}.reserved_extra_memory_b,
            {import_name}.is_executing,
            {import_name}.query_name
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.node_name,
                {import_name}.session_id,
                {import_name}.identifier,
                {import_name}.query,
                {import_name}.query_search_path,
                {import_name}.schema_name
        SEGMENTED BY hash({import_name}.transaction_id, 
                {import_name}.statement_id, 
                {import_name}.query_start_epoch,
                {import_name}.error_code,
                {import_name}.processed_row_count,
                {import_name}.reserved_extra_memory_b,
                {import_name}.is_executing,
                {import_name}.query_duration_us)
        ALL NODES;

        """


################ resource_pool_status ###################
class ResourcePoolStatusTable(CollectionTable):
    """
    ``ResourcePoolStatusTable`` stores a snapshot of data from the system table resource_pool_status.
    It adds two additional columns to the table to label the status: transaction_id and statement_id.
    """

    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__(AllTableTypes.RESOURCE_POOL_STATUS, table_schema, key)

    def get_create_table_sql(self) -> str:
        return f"""
        /* Create the real table */
        CREATE TABLE IF NOT EXISTS  {self.get_import_name_fq()}
        (
            node_name varchar(128),
            pool_oid int,
            pool_name varchar(128),
            is_internal boolean,
            memory_size_kb int,
            memory_size_actual_kb int,
            memory_inuse_kb int,
            general_memory_borrowed_kb int,
            queueing_threshold_kb int,
            max_memory_size_kb int,
            max_query_memory_size_kb int,
            running_query_count int,
            planned_concurrency int,
            max_concurrency int,
            is_standalone boolean,
            -- queue timeout is really an interval
            queue_timeout interval,
            queue_timeout_in_seconds int,
            execution_parallelism varchar(128),
            priority int,
            runtime_priority varchar(128),
            runtime_priority_threshold int,
            runtimecap_in_seconds int,
            single_initiator varchar(128),
            query_budget_kb int,
            cpu_affinity_set varchar(256),
            cpu_affinity_mask varchar(1024),
            cpu_affinity_mode varchar(128),
            transaction_id int,
            statement_id int,
            query_name varchar(128)
        );
        """

    def get_create_projection_sql(self) -> str:
        import_name = self.get_import_name()
        fq_proj_name = self.get_super_proj_name_fq()
        import_name_fq = self.get_import_name_fq()
        return f"""
        CREATE PROJECTION IF NOT EXISTS {fq_proj_name} 
        /*+basename({import_name}),createtype(A)*/
        (
            node_name,
            pool_oid,
            pool_name,
            is_internal,
            memory_size_kb,
            memory_size_actual_kb,
            memory_inuse_kb,
            general_memory_borrowed_kb,
            queueing_threshold_kb,
            max_memory_size_kb,
            max_query_memory_size_kb,
            running_query_count,
            planned_concurrency,
            max_concurrency,
            is_standalone,
            queue_timeout,
            queue_timeout_in_seconds,
            execution_parallelism,
            priority,
            runtime_priority,
            runtime_priority_threshold,
            runtimecap_in_seconds,
            single_initiator,
            query_budget_kb,
            cpu_affinity_set,
            cpu_affinity_mask,
            cpu_affinity_mode,
            transaction_id,
            statement_id,
            query_name
        )
        AS
        SELECT {import_name}.node_name,
                {import_name}.pool_oid,
                {import_name}.pool_name,
                {import_name}.is_internal,
                {import_name}.memory_size_kb,
                {import_name}.memory_size_actual_kb,
                {import_name}.memory_inuse_kb,
                {import_name}.general_memory_borrowed_kb,
                {import_name}.queueing_threshold_kb,
                {import_name}.max_memory_size_kb,
                {import_name}.max_query_memory_size_kb,
                {import_name}.running_query_count,
                {import_name}.planned_concurrency,
                {import_name}.max_concurrency,
                {import_name}.is_standalone,
                {import_name}.queue_timeout,
                {import_name}.queue_timeout_in_seconds,
                {import_name}.execution_parallelism,
                {import_name}.priority,
                {import_name}.runtime_priority,
                {import_name}.runtime_priority_threshold,
                {import_name}.runtimecap_in_seconds,
                {import_name}.single_initiator,
                {import_name}.query_budget_kb,
                {import_name}.cpu_affinity_set,
                {import_name}.cpu_affinity_mask,
                {import_name}.cpu_affinity_mode,
                {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.query_name
        FROM {import_name_fq}
        ORDER BY {import_name}.transaction_id,
                {import_name}.statement_id,
                {import_name}.node_name,
                {import_name}.pool_name,
                {import_name}.query_budget_kb
        SEGMENTED BY hash({import_name}.pool_oid, 
                    {import_name}.is_internal, 
                    {import_name}.memory_size_kb, 
                    {import_name}.memory_size_actual_kb,
                    {import_name}.memory_inuse_kb,
                    {import_name}.general_memory_borrowed_kb,
                    {import_name}.queueing_threshold_kb,
                    {import_name}.max_memory_size_kb) 
        ALL NODES;
        """
