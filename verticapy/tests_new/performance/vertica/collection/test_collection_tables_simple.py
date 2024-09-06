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
import logging

import pytest

from verticapy.performance.vertica.collection.collection_tables import ALL_TABLES_V1


class TestCollectionTables:
    """
    Tests for the CollectionTable and its subclasses
    """

    def test_no_duplicates(self):
        """
        Ensure that there are no duplicates in ALL_TABLES_V*
        """

        observed_tables = set()
        for t in ALL_TABLES_V1:
            assert t not in observed_tables
            observed_tables.add(t)
