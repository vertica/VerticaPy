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
from verticapy.tests_new.pipeline.conftest import build_pipeline, pipeline_exists


def test_parser():
    build_pipeline("test_pipeline")
    build_pipeline("test_pipeline")  # Purposely test duplicates
    build_pipeline("test_pipeline_2")

    assert pipeline_exists("test_pipeline", check_metric=True)
    assert pipeline_exists("test_pipeline_2", check_metric=True)
