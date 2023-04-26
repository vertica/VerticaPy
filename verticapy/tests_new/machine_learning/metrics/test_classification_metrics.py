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

import sklearn.metrics as skl_metrics

import verticapy.machine_learning.metrics.classification as vpy_metrics


class TestClassificationMetrics:
    def test_f1_score(self, pred_cl_dataset):
        vdf, y_t, y_s, labels = pred_cl_dataset
        kwargs = {"average": "weighted", "labels": labels}
        vpy_res = vpy_metrics.f1_score("y_t", "y_s", vdf, **kwargs)
        skl_res = skl_metrics.f1_score(y_t, y_s, **kwargs)
        assert vpy_res == pytest.approx(skl_res)
