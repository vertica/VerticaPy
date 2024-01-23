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
        raise NotImplementedError("get_create_table_sql is not implemented in the base class CollectionTable")

    @abstractmethod
    def get_create_projection_sql(self) -> str:
        raise NotImplementedError("get_create_projection_sql is not implemented in the base class CollectionTable")


    
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

    # TODO: eventually, this will probably be     
    return CollectionTable(table_name, target_schema, key)

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
    