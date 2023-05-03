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
import verticapy.machine_learning.metrics.regression as vpy_metrics
import pandas as pd
import numpy as np
from verticapy.core.vdataframe.base import vDataFrame


# @pytest.mark.parametrize('compute_method, dataset, func_args, expected',
#                          [
#                              ('binary', 'pred_cl_dataset_binary', {'pos_label': 'b'}, ''),
#                              ('micro', 'pred_cl_dataset_multi', {'labels': 'labels'}, ''),
#                              ('macro', 'pred_cl_dataset_multi', {'labels': 'labels'}, ''),
#                              ('weighted', 'pred_cl_dataset_multi', {'labels': 'labels'}, ''),
#                              # ('scores', 'pred_cl_dataset_multi', ''),
#                              # (None, 'pred_cl_dataset_multi', ''),
#                              # pytest.param('invalid', 'pred_cl_dataset_binary', {'pos_label': 'b'}, '', marks=pytest.mark.xfail)
#                          ])
@pytest.mark.parametrize('vpy_regression_metrics, skl_regression_metrics, is_skl_metrics',
                         [
                             ('explained_variance', 'explained_variance_score', 'y')
                         ])
class TestRegressionMetrics:
    def test_master_regression_metrics(self, vpy_regression_metrics, skl_regression_metrics, is_skl_metrics, request):

        # vdf, y_true, y_pred, y_prob, labels = request.getfixturevalue(dataset)
        y_true = np.arange(50)
        y_pred = y_true + 1
        input_relation = np.column_stack((y_true, y_pred))
        vdf = vDataFrame(input_relation, usecols=["y_true", "y_pred"])

        # if 'labels' in func_args: func_args['labels'] = labels

        # verticapy logic
        if vpy_regression_metrics:
            vpy_res = getattr(vpy_metrics, vpy_regression_metrics)("y_true", "y_pred", vdf)

        # sklearn logic
        if is_skl_metrics == 'y':
            if skl_regression_metrics:
                skl_res = getattr(skl_metrics, skl_regression_metrics)(y_true, y_pred)

        print(f'vertica: {vpy_res}, sklearn: {skl_res}')
        assert vpy_res == pytest.approx(skl_res)
