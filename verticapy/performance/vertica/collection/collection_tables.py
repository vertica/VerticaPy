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

class CollectionTable:
    def __init__(self, table_name:str, table_schema:str, key:str) -> None:
        self.name = table_name
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
        return (f"{self.schema}.{self.import_prefix}{self.name}{self.import_suffix}"
                if fully_qualified else
                f"{self.import_prefix}{self.name}{self.import_suffix}")
    
    def get_staging_name(self) -> str:
        return f"{self.schema}.{self.import_prefix}{self.name}{self.staging_suffix}{self.import_suffix}"

    # Recall: abstract methods won't raise by default
    @abstractmethod
    def get_create_table_sql(self) -> str:
        raise NotImplementedError(f"get_create_table_sql is not implemented in the base class CollectionTable."
                                  f" Current table name = {self.name} schema {self.schema}")

    @abstractmethod
    def get_create_projection_sql(self) -> str:
        raise NotImplementedError(f"get_create_projection_sql is not implemented in the base class CollectionTable"
                                  f" Current table name = {self.name} schema {self.schema}")


    
class AllTableNames(Enum):
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


def getAllCollectionTables(target_schema:str, key:str) -> Mapping[str,CollectionTable]:
    result = {}
    for name in AllTableNames:
        name_str = name.value
        c = collectionTableFactory(name_str, target_schema, key)
        result[name.name] = c

    return result

def collectionTableFactory(table_name:str, target_schema:str, key:str) -> CollectionTable:
    if table_name == AllTableNames.COLLECTION_EVENTS.value:
        return CollectionEventsTable(target_schema, key)
    if table_name == AllTableNames.COLLECTION_INFO.value:
        return CollectionInfoTable(target_schema, key)
    if table_name == AllTableNames.DC_EXPLAIN_PLANS.value:
        return DCExplainPlansTable(target_schema, key)
    if table_name == AllTableNames.DC_QUERY_EXECUTIONS.value:
        return DCQueryExecutionsTable(target_schema, key)
    if table_name == AllTableNames.DC_REQUESTS_ISSUED.value:
        return DCRequestsIssuedTable(target_schema, key)
    if table_name == AllTableNames.EXECUTION_ENGINE_PROFILES.value:
        return DCExecutionEngineProfilesTable(target_schema, key)
    

    # TODO: eventually this will be an error    
    return CollectionTable(table_name, target_schema, key)


############## collection_events ######################
class CollectionEventsTable(CollectionTable):
    def __init__(self, table_schema:str, key:str) -> None:
        super().__init__("collection_events", table_schema, key)

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
    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__("collection_info", table_schema, key)

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
    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__("dc_explain_plans", table_schema, key)

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
    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__("dc_query_executions", table_schema, key)

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
    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__("dc_requests_issued", table_schema, key)

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
class DCExecutionEngineProfilesTable(CollectionTable):
    def __init__(self, table_schema: str, key: str) -> None:
        super().__init__("execution_engine_profiles", table_schema, key)

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