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
from verticapy.core.vdataframe._multiprocessing import (
    aggregate_parallel_block,
    describe_parallel_block,
)


class TestMultiprocessing:
    """
    test class for Multiprocessing functions test
    """

    def test_aggregate_parallel_block(self, titanic_vd_fun):
        """
        test aggregate_parallel_block function
        """
        _res = aggregate_parallel_block(
            titanic_vd_fun, "max", titanic_vd_fun.get_columns(), 5, 1
        )
        res = dict(zip(_res.values["index"], _res.values["max"]))
        assert res['"age"'] == titanic_vd_fun["age"].max()

    def test_describe_parallel_block(self, titanic_vd_fun):
        """
        test describe_parallel_block function
        """
        res = describe_parallel_block(
            titanic_vd_fun, "numerical", ["age"], False, 10, 1
        )

        assert res.shape() == titanic_vd_fun.describe().shape()
