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
from verticapy.io.sql.create import create_schema, create_table, create_verticapy_schema
from verticapy.io.sql.drop import drop
from verticapy.io.sql.read import readSQL, to_tablesample, vDataFrameSQL
from verticapy.io.sql.insert import insert_into, insert_verticapy_schema
