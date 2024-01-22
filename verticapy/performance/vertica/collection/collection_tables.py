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

from enum import Enum
from typing import Mapping

class CollectionTable:
    def __init__(self, table_name:str, tables_schema:str, key:str) -> None:
        self.name = table_name
        self.schema = tables_schema
        self.key = key
        self.import_prefix = "qprof_"
        self.import_suffix = f"_{key}"
        self.staging_suffix = "_staging"

    def get_import_name_fq(self) -> str:
        return self._get_import_name(fully_qualified=True)
    
    def get_import_name(self) -> str:
        return self._get_import_name(fully_qualified=False)
    
    def _get_import_name(self, fully_qualified) -> str:
        return (f"{self.schema}.{self.import_prefix}{self.name}{self.import_suffix}"
                if fully_qualified else
                f"{self.import_prefix}{self.name}{self.import_suffix}")
    
    def get_staging_name(self) -> str:
        return f"{self.schema}.{self.import_prefix}{self.name}{self.staging_suffix}{self.import_suffix}"

    
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
        c = CollectionTable(name_str, target_schema, key)
        result[name.name] = c

    return result
