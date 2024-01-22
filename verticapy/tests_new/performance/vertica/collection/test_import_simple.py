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

import verticapy as vp
from verticapy._utils._sql._sys import _executeSQL

from verticapy.performance.vertica.collection.profile_import import ProfileImport, ProfileImportError
from verticapy.datasets import load_amazon
from verticapy.core.vdataframe import vDataFrame


class TestQueryProfilerSimple:
    """
    Test Base Class.
    """

    def test_empty_schema(self, schema_loader):
        pi = ProfileImport(target_schema=schema_loader,
                           key='no_such_key',
                           filename='no_such_file.tar',
                           skip_create_table=True)
        with pytest.raises(ProfileImportError, 
                           match=f"Missing [0-9]+ tables in schema {schema_loader}"):
            pi.check_schema()