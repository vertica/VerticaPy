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
import numpy as np
import random
from verticapy.core.vdataframe.base import vDataFrame


@pytest.mark.parametrize('compute_method, expected',
                         [
                             ('micro', ''),
                             # ('macro', ''),
                             # ('weighted', ''),
                             # ('scores', ''),
                             # (None, ''),
                             # pytest.param('invalid', '', marks=pytest.mark.xfail)
                         ])
class TestClassificationMetrics:

    @staticmethod
    def skl_metrics_cal(pred_cl_dataset_binary):
        skl_metrics_dict = {}
        vdf, y_t, y_s, labels = pred_cl_dataset_binary
        tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()

        skl_metrics_dict['skl_tpr'] = tp / (tp + fn)
        skl_metrics_dict['skl_tnr'] = tn / (tn + fp)
        skl_metrics_dict['skl_fnr'] = fn / (fn + tp)
        skl_metrics_dict['skl_fpr'] = fp / (fp + tn)
        skl_metrics_dict['skl_specificity'] = tn / (tn + fp)
        skl_metrics_dict['tn'] = tn
        skl_metrics_dict['fp'] = fp
        skl_metrics_dict['fn'] = fn
        skl_metrics_dict['tp'] = tp

        return skl_metrics_dict

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

    def test_balanced_accuracy(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
        # binary class
        vdf, y_t, y_s, labels = pred_cl_dataset_binary
        # generating probability - float
        # y_p = [random.uniform(0, 1) for x in range(len(y_t))]
        # input_relation = np.column_stack((y_t, y_p))
        # vdf_p = vDataFrame(input_relation, usecols=["y_t", "y_p"])
        vpy_res = vpy_metrics.balanced_accuracy("y_t", "y_p", vdf_p, average=compute_method, labels=labels, pos_label='b')

        skl_res = skl_metrics.log_loss(y_t, y_p, labels=labels)

        print(f'vertica: {vpy_res}, sklearn: {skl_res}')

        assert vpy_res == pytest.approx(skl_res)

    # def test_critical_success_index(self, pred_cl_dataset_multi, compute_method, expected):
    #     pass

    # def test_diagnostic_odds_ratio(self, pred_cl_dataset_multi, compute_method, expected):
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

    # def test_false_negative_rate(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     kwargs = {"average": compute_method, "labels": labels}
    #     vpy_res = vpy_metrics.false_negative_rate("y_t", "y_s", vdf, kwargs, pos_label='b')
    #     tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     skl_res = fn/(tp+fn)
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # multi class
    # vdf, y_t, y_s, labels = pred_cl_dataset_multi
    # vpy_res = vpy_metrics.false_negative_rate("y_t", "y_s", vdf, average=compute_method, labels=labels)
    # print(vpy_res)
    # tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    # skl_recall = fn/(tp+fn)
    # skl_res = 1 - skl_recall
    # print(skl_metrics.confusion_matrix(y_t, y_s, labels=labels))

    # print(vpy_metrics.confusion_matrix("y_t", "y_s", vdf, labels=labels))

    # assert vpy_res == pytest.approx(skl_res)

    # def test_false_positive_rate(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     kwargs = {"average": compute_method, "labels": labels}
    #     vpy_res = vpy_metrics.false_positive_rate("y_t", "y_s", vdf, kwargs, pos_label='b')
    #     tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     skl_res = fp / (fp + tn)
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_false_discovery_rate(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     kwargs = {"average": compute_method, "labels": labels}
    #     vpy_res = vpy_metrics.false_discovery_rate("y_t", "y_s", vdf, kwargs, pos_label='b')
    #     tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     skl_res = fp / (fp + tp)
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_false_omission_rate(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     kwargs = {"average": compute_method, "labels": labels}
    #     vpy_res = vpy_metrics.false_omission_rate("y_t", "y_s", vdf, kwargs, pos_label='b')

    #     tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     skl_res = fn / (fn + tn)
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')

    #     assert vpy_res == pytest.approx(skl_res)

    # def test_fowlkes_mallows_index(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     kwargs = {"average": compute_method, "labels": labels}
    #     vpy_res = vpy_metrics.fowlkes_mallows_index("y_t", "y_s", vdf, kwargs, pos_label='b')
    #
    #     # skl_res = skl_metrics.fowlkes_mallows_score(y_t, y_s)
    #     tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     skl_res = tp / np.sqrt((tp + fp) * (tp + fn))
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_informedness(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     kwargs = {"average": compute_method, "labels": labels}
    #     vpy_res = vpy_metrics.informedness("y_t", "y_s", vdf, kwargs, pos_label='b')
    #
    #     # skl_res = skl_metrics.fowlkes_mallows_score(y_t, y_s)
    #     tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     skl_res = (tp/(tp+fn)) + (tn/(tn+fp)) - 1  # Recall + Inverse Recall – 1
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_markedness(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     kwargs = {"average": compute_method, "labels": labels}
    #     vpy_res = vpy_metrics.markedness("y_t", "y_s", vdf, kwargs, pos_label='b')
    #
    #     # skl_res = skl_metrics.fowlkes_mallows_score(y_t, y_s)
    #     tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     skl_res = (tp / (tp + fp)) + (tn / (tn + fn)) - 1  # Precision + Inverse Precision – 1
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_matthews_corrcoef(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     kwargs = {"average": compute_method, "labels": labels}
    #     vpy_res = vpy_metrics.matthews_corrcoef("y_t", "y_s", vdf, kwargs, pos_label='b')
    #
    #     skl_res = skl_metrics.matthews_corrcoef(y_t, y_s)
    #     # tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     # skl_res = ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_negative_predictive_score(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    # binary class
    # vdf, y_t, y_s, labels = pred_cl_dataset_binary
    # kwargs = {"average": compute_method, "labels": labels}
    # vpy_res = vpy_metrics.negative_predictive_score("y_t", "y_s", vdf, kwargs, pos_label='b')

    # tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    # skl_res = tn/(tn+fn)
    # print(f'vertica: {vpy_res}, sklearn: {skl_res}')

    # assert vpy_res == pytest.approx(skl_res)

    # def test_negative_likelihood_ratio(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.negative_likelihood_ratio("y_t", "y_s", vdf, average=compute_method, labels=labels,
    #                                                     pos_label='b')
    #
    #     tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     skl_res = (fn/(fn+tp)) / (tn/(tn+fp))
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_positive_likelihood_ratio(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.positive_likelihood_ratio("y_t", "y_s", vdf, average=compute_method, labels=labels,
    #                                                     pos_label='b')
    #
    #     tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     skl_res = (tp / (tp + fn)) / (fp / (fp + tn))
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_precision_score(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.precision_score("y_t", "y_s", vdf, average=compute_method, labels=labels, pos_label='b')
    #
    #     skl_res = skl_metrics.precision_score(y_t, y_s, average='binary', labels=labels, pos_label='b')
    #     # skl_res = (tp / (tp + fn)) / (fp / (fp + tn))
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_prevalence_threshold(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.prevalence_threshold("y_t", "y_s", vdf, average=compute_method, labels=labels,
    #                                                pos_label='b')
    #
    #     # tn, fp, fn, tp = skl_metrics.confusion_matrix(y_t, y_s).ravel()
    #     # skl_res = np.sqrt((fp/(fp + tn))) / (np.sqrt((fp/(fp+tn))) + np.sqrt(tp/(tp+fn)))
    #     skl_tpr, skl_tnr, skl_fnr, skl_fpr = self.data_setup(pred_cl_dataset_binary)
    #     skl_res = np.sqrt(skl_fpr)/(np.sqrt(skl_fpr)+np.sqrt(skl_tpr))
    #
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)
    #
    # def test_recall_score(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.recall_score("y_t", "y_s", vdf, average=compute_method, labels=labels,
    #                                        pos_label='b')
    #
    #     skl_res = skl_metrics.recall_score(y_t, y_s, average='binary', labels=labels, pos_label='b')
    #
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_specificity_score(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.specificity_score("y_t", "y_s", vdf, average=compute_method, labels=labels,
    #                                             pos_label='b')
    #
    #     skl_metrics_dict = self.skl_metrics_cal(pred_cl_dataset_binary)
    #     skl_res = skl_metrics_dict['skl_specificity']
    #
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_best_cutoff(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.best_cutoff("y_t", "y_s", vdf, average=compute_method, labels=labels,
    #                                             pos_label='b')
    #
    #     skl_metrics_dict = self.skl_metrics_cal(pred_cl_dataset_binary)
    #     skl_res = skl_metrics_dict['skl_specificity']
    #
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_roc_auc(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     # generating probability - float
    #     y_p = [random.uniform(0, 1) for x in range(len(y_t))]
    #     input_relation = np.column_stack((y_t, y_p))
    #     vdf_p = vDataFrame(input_relation, usecols=["y_t", "y_p"])
    #     vpy_res = vpy_metrics.roc_auc("y_t", "y_p", vdf_p, average=compute_method, labels=labels, pos_label='b')
    #
    #     skl_res = skl_metrics.roc_auc_score(y_t, y_p, labels=labels)
    #
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_prc_auc(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     # generating probability - float
    #     y_p = [random.uniform(0, 1) for x in range(len(y_t))]
    #     # print(sorted(y_p))
    #     input_relation = np.column_stack((y_t, y_p))
    #     vdf_p = vDataFrame(input_relation, usecols=["y_t", "y_p"])
    #     vpy_res = vpy_metrics.prc_auc("y_t", "y_p", vdf_p, average=compute_method, labels=labels, pos_label='b')
    #     print(vpy_res)
    #
    #     precision, recall, thresholds = skl_metrics.precision_recall_curve(y_t, y_p, pos_label='b')
    #     print(skl_metrics.auc(sorted(precision), sorted(recall)))

        # print(f'vertica: {vpy_res}, sklearn: {skl_res}')

        # assert vpy_res == pytest.approx(skl_res)

    # def test_log_loss(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     # generating probability - float
    #     y_p = [random.uniform(0, 1) for x in range(len(y_t))]
    #     input_relation = np.column_stack((y_t, y_p))
    #     vdf_p = vDataFrame(input_relation, usecols=["y_t", "y_p"])
    #     vpy_res = vpy_metrics.log_loss("y_t", "y_p", vdf_p, average=compute_method, labels=labels, pos_label='b')
    #
    #     skl_res = skl_metrics.log_loss(y_t, y_p, labels=labels)
    #
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)

    # def test_classification_report(self, pred_cl_dataset_binary, pred_cl_dataset_multi, compute_method, expected):
    #     # binary class
    #     vdf, y_t, y_s, labels = pred_cl_dataset_binary
    #     vpy_res = vpy_metrics.classification_report("y_t", "y_s", vdf, average=compute_method, labels=labels,
    #                                                 pos_label='b')
    #
    #     skl_metrics_dict = self.skl_metrics_cal(pred_cl_dataset_binary)
    #     skl_res = skl_metrics_dict['skl_specificity']
    #
    #     print(f'vertica: {vpy_res}, sklearn: {skl_res}')
    #
    #     assert vpy_res == pytest.approx(skl_res)
