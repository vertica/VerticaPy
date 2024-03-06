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
import pytest

import verticapy.sql.sys as sys

from verticapy.pipeline._helper import setup


@pytest.fixture(scope="session", autouse=True)
def pipeline_setup():
    setup()


def pipeline_exists(pipeline_name: str):
    """
    helper function to test if a pipeline
    is properly created.
    """
    return (
            sys.does_view_exist(f"{pipeline_name}_TRAIN_VIEW", "public")
            and sys.does_view_exist(f"{pipeline_name}_TEST_VIEW", "public")
            and sys.does_table_exist(f"{pipeline_name}_METRIC_TABLE", "public")
           )

def pipeline_not_exists(pipeline_name: str):
    """
    helper function to test if a pipeline
    no longer exists.
    """
    return (
            not sys.does_view_exist(f"{pipeline_name}_TRAIN_VIEW", "public")
            and not sys.does_view_exist(f"{pipeline_name}_TEST_VIEW", "public")
            and not sys.does_table_exist(f"{pipeline_name}_METRIC_TABLE", "public")
           )
