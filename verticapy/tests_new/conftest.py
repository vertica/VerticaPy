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
import pytest
import numpy as np

from verticapy.core.vdataframe.base import vDataFrame

@pytest.fixture(scope="module")
def pred_cl_dataset():
    labels = np.array(["a", "b", "c"])
    y_t = np.array(["a", "a", "b", "c", "c", "a", "b", "c", "a", "b", "c", "a", "b", "a", "b", "c", "a", "b", "c"])
    y_s = np.array(["a", "b", "b", "b", "c", "a", "b", "a", "a", "c", "a", "a", "b", "a", "b", "c", "b", "b", "a"])
    input_relation = np.column_stack((y_t, y_s))
    vdf = vDataFrame(input_relation, usecols=["y_t", "y_s"])
    yield (vdf, y_t, y_s, labels)
