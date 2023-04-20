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


@pytest.mark.parametrize('compute_method, expected',
                         [
                             ('micro', ''),
                             ('macro', ''),
                             ('weighted', ''),
                             ('scores', ''),
                             (None, ''),
                             pytest.param('invalid', '', marks=pytest.mark.xfail)
                         ])
class TestClassificationMetrics:

    # def test_confusion_matrix(self, pred_cl_dataset_binary, pred_cl_dataset_multi):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.confusion_matrix("y_t", "y_s", vdf, 'b')
    #     skl_res = skl_metrics.confusion_matrix(y_t, y_s, labels=['a', 'b'])
    #     assert vpy_res == pytest.approx(skl_res)

    # @pytest.mark.parametrize('compute_method, expected',
    #                          [
    #                              ('micro', ''),
    #                              ('macro', ''),
    #                              ('weighted', ''),
    #                              (None, ''),
    #                              pytest.param('invalid', '', marks=pytest.mark.xfail)
    #                          ])
    # def test_accuracy_score(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.accuracy_score("y_t", "y_s", vdf, 'b')
    #     kwargs = {"average": compute_method, "labels": labels}

    # def test_balanced_accuracy(self):
    #     pass

    # def test_critical_success_index(self):
    #     pass

    # def test_diagnostic_odds_ratio(self):
    #     pass
    #
    # @pytest.mark.skip(reason="skipping to test other functions")
    # @pytest.mark.parametrize('compute_method, expected',
    #                          [
    #                              ('micro', ''),
    #                              ('macro', ''),
    #                              ('weighted', ''),
    #                              ('scores', ''),
    #                              (None, ''),
    #                              pytest.param('invalid', '', marks=pytest.mark.xfail)
    #                          ])
    # def test_f1_score(self, pred_cl_dataset_multi, compute_method, expected):
    #     vdf, y_t, y_s, labels = pred_cl_dataset_multi
    #     kwargs = {"average": compute_method, "labels": labels}
    #     vpy_res = vpy_metrics.f1_score("y_t", "y_s", vdf, **kwargs)
    #     skl_res = skl_metrics.f1_score(y_t, y_s, **kwargs)
    #     assert vpy_res == pytest.approx(skl_res)

    def test_false_negative_rate(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
        # binary class
        vdf, y_t, y_s, labels = pred_cl_dataset_binary
        kwargs = {"average": compute_method, "labels": labels}
        vpy_res = vpy_metrics.false_negative_rate("y_t", "y_s", vdf, kwargs)
        tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
        skl_recall = tp/(tp+fn)
        skl_false_negative_rate = 1 - skl_recall
        print(skl_false_negative_rate)
        print('111111111111111111111')
        print(vpy_res)

        # multi class
        # vdf, y_t, y_s, labels = pred_cl_dataset_multi
        # kwargs = {"average": compute_method, "labels": labels}
        # vpy_res = vpy_metrics.false_negative_rate("y_t", "y_s", vdf, kwargs)
        # skl_res = skl_metrics.confusion_matrix(y_t, y_s).ravel()
        # print(skl_res)
        # print('111111111111111111111')
        # print(vpy_res)

    # def test_false_positive_rate(self):
    #     pass
    #
    # def test_false_discovery_rate(self):
    #     pass
    #
    # def test_false_omission_rate(self):
    #     pass
    #
    # def test_fowlkes_mallows_index(self):
    #     pass
    #
    # def test_informedness(self):
    #     pass
    #
    # def test_markedness(self):
    #     pass
    #
    # def test_matthews_corrcoef(self):
    #     pass
    #
    # def test_negative_predictive_score(self):
    #     pass
    #
    # def test_negative_likelihood_ratio(self):
    #     pass
    #
    # def test_positive_likelihood_ratio(self):
    #     pass
    #
    # def test_precision_score(self):
    #     pass
    #
    # def test_prevalence_threshold(self):
    #     pass
    #
    # def test_recall_score(self):
    #     pass
    #
    # def test_specificity_score(self):
    #     pass
    #
    # def test_best_cutoff(self):
    #     pass
    #
    # def test_roc_auc(self):
    #     pass
    #
    # def test_prc_auc(self):
    #     pass
    #
    # def test_log_loss(self):
    #     pass
    #
    # def test_classification_report(self):
    #     pass
