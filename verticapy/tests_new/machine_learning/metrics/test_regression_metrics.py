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
from verticapy.learn.linear_model import LinearRegression
import statsmodels.api as sm
from verticapy import drop


@pytest.fixture(scope="module")
def vpy_lr_model_pred(winequality_vpy):
    vpy_lr_model = LinearRegression("vpy_lr_model", )
    vpy_lr_model.drop()
    vpy_lr_model.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"], "quality", )
    vpy_lr_pred_vdf = vpy_lr_model.predict(winequality_vpy, name='quality_pred')
    yield vpy_lr_model, vpy_lr_pred_vdf
    vpy_lr_model.drop()
    drop(name="vpy_lr_pred", )


@pytest.mark.parametrize('sql_relation_type, expected',
                         [
                             ('table', ''),
                             ('view', ''),
                             ('temporary', ''),
                             # ('custom', ''),
                             # pytest.param('invalid', '', marks=pytest.mark.xfail)
                         ])
@pytest.mark.parametrize('vpy_regression_metrics, skl_regression_metrics, is_skl_metrics',
                         [
                             # ('aic_score', 'aic', 'n'),  # fail
                             # ('bic_score', 'bic', 'n'),  # fail
                             ('explained_variance', 'explained_variance_score', 'y'),
                             # ('max_error', 'max_error', 'y'),
                             # ('mean_absolute_error', 'mean_absolute_error', 'y'),
                             # ('mean_squared_error', 'mean_squared_error', 'y'),
                             # ('mean_squared_log_error', 'mean_squared_log_error', 'y'),  # fail
                             # ('median_absolute_error', 'median_absolute_error', 'y'),  # fail
                             # ('quantile_error', 'quantile_error', 'y'),  # error
                             # ('r2_score', 'r2_score', 'y'),
                             # ('anova_table', 'anova_table', 'y'),  # need to implement
                             # ('regression_report', 'regression_report', 'y'),  # need to implement

                         ])
class TestRegressionMetrics:
    def test_master_regression_metrics(self, sql_relation_type, expected, vpy_lr_model_pred, vpy_regression_metrics,
                                       skl_regression_metrics, is_skl_metrics, request):

        vpy_lr_model, vpy_lr_pred_vdf = vpy_lr_model_pred

        # converts to pandas dataframe for non vertica framework
        vpy_lr_pred_pdf = vpy_lr_pred_vdf.to_pandas()
        vpy_lr_pred_pdf['citric_acid'] = vpy_lr_pred_pdf['citric_acid'].astype(float)
        vpy_lr_pred_pdf['residual_sugar'] = vpy_lr_pred_pdf['residual_sugar'].astype(float)

        # vdf, y_true, y_pred, y_prob, labels = request.getfixturevalue(dataset)
        y_true = np.arange(50)
        y_pred = y_true + 1
        # input_relation = np.column_stack((y_true, y_pred))
        # vdf = vDataFrame(input_relation, usecols=["y_true", "y_pred"])

        # verticapy logic
        if vpy_regression_metrics:
            # verticapy dataframe to vertica db
            drop(name="public.vpy_lr_pred", )
            vpy_lr_pred_vdf.to_db(name='vpy_lr_pred', relation_type=f"{sql_relation_type}")
            vpy_res = getattr(vpy_metrics, vpy_regression_metrics)("quality", "quality_pred", 'vpy_lr_pred')

        # sklearn logic
        if is_skl_metrics == 'y':
            if skl_regression_metrics:
                skl_res = getattr(skl_metrics, skl_regression_metrics)(vpy_lr_pred_pdf['quality'],
                                                                       vpy_lr_pred_pdf['quality_pred'])
        else:
            x = vpy_lr_pred_pdf[["citric_acid", "residual_sugar", "alcohol"]]
            y = vpy_lr_pred_pdf['quality']
            # add constant to predictor variables
            x = sm.add_constant(x)

            # fit regression model
            model = sm.OLS(y, x).fit()

            # view AIC of model
            print(model.aic)
            print(model.bic)
            skl_res = model.aic

        print(f'vertica: {vpy_res}, sklearn: {skl_res}')
        assert vpy_res == pytest.approx(skl_res)
