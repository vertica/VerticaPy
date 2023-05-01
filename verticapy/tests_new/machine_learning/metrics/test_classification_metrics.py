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
import pandas as pd


@pytest.mark.parametrize('compute_method, dataset, func_args, expected',
                         [
                             ('binary', 'pred_cl_dataset_binary', {'pos_label': 'b'}, ''),
                             ('micro', 'pred_cl_dataset_multi', {'labels': 'labels'}, ''),
                             ('macro', 'pred_cl_dataset_multi', {'labels': 'labels'}, ''),
                             ('weighted', 'pred_cl_dataset_multi', {'labels': 'labels'}, ''),
                             # ('scores', 'pred_cl_dataset_multi', ''),
                             # (None, 'pred_cl_dataset_multi', ''),
                             # pytest.param('invalid', 'pred_cl_dataset_multi', '', marks=pytest.mark.xfail)
                         ])
@pytest.mark.parametrize('is_skl_metrics, classification_metrics',
                         [
                             ('y', 'confusion_matrix'),
                             # ('y', 'accuracy_score'),  # fail
                             # ('y', 'balanced_accuracy_score'),  # fail
                             # ('n', 'critical_success_index'),
                             # ('n', 'diagnostic_odds_ratio'),  # fail
                             # ('y', 'f1_score'),
                             # ('n', 'false_negative_rate'),
                             # ('n', 'false_positive_rate'),
                             # ('n', 'false_discovery_rate'),
                             # ('n', 'false_omission_rate'),
                             # ('n', 'fowlkes_mallows_index'),
                             # ('n', 'informedness'),
                             # ('n', 'markedness'),
                             # ('y', 'matthews_corrcoef'),  # fail
                             # ('n', 'negative_predictive_score'),
                             # ('n', 'negative_likelihood_ratio'),
                             # ('n', 'positive_likelihood_ratio'),
                             # ('y', 'precision_score'),
                             # ('n', 'prevalence_threshold'),  # fail
                             # ('y', 'recall_score'),
                             # ('n', 'specificity_score'),
                             # ('n', 'best_cutoff'),  # need to implement
                             # ('y', 'roc_auc_score'),  # error for multi class
                             # ('n', 'prc_auc'),  # need to implement
                             # ('y', 'log_loss'),  # fail
                             # ('y', 'classification_report') # error
                         ])
class TestClassificationMetrics:

    @staticmethod
    def skl_metrics_cal(y_true, y_pred, average='binary', classification_metric=''):
        skl_metrics_dict = {}
        support = []

        if average == 'binary':
            tn, fp, fn, tp = skl_metrics.confusion_matrix(y_true, y_pred).ravel()

            skl_metrics_dict['skl_tpr'] = tp / (tp + fn)
            skl_metrics_dict['skl_tnr'] = tn / (tn + fp)
            skl_metrics_dict['skl_fnr'] = fn / (fn + tp)
            skl_metrics_dict['skl_fpr'] = fp / (fp + tn)
            skl_metrics_dict['skl_sensitivity'] = skl_metrics_dict['skl_tpr']
            skl_metrics_dict['skl_specificity_score'] = tn / (tn + fp)
            skl_metrics_dict['skl_critical_success_index'] = tp / (tp + fp + fn)
            skl_metrics_dict['skl_diagnostic_odds_ratio'] = (tp * tn) / (fp + fn)
            skl_metrics_dict['skl_false_negative_rate'] = fn / (fn + tp)
            skl_metrics_dict['skl_false_positive_rate'] = fp / (fp + tn)
            skl_metrics_dict['skl_false_discovery_rate'] = fp / (fp + tp)
            skl_metrics_dict['skl_false_omission_rate'] = fn / (fn + tn)
            skl_metrics_dict['skl_fowlkes_mallows_index'] = tp / np.sqrt((tp + fp)*(tp + fn))
            skl_metrics_dict['skl_informedness'] = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
            skl_metrics_dict['skl_markedness'] = (tp / (tp + fp)) + (tn / (tn + fn)) - 1
            skl_metrics_dict['skl_negative_predictive_score'] = tn/(tn + fn)
            skl_metrics_dict['skl_negative_likelihood_ratio'] = (fn / (fn + tp)) / (tn / (tn + fp))
            skl_metrics_dict['skl_positive_likelihood_ratio'] = (tp / (tp + fn)) / (fp / (fp + tn))
            skl_metrics_dict['skl_prevalence_threshold'] = (np.sqrt(skl_metrics_dict['skl_sensitivity']*(-skl_metrics_dict['skl_specificity_score'] + 1)) + skl_metrics_dict['skl_specificity_score']-1) / (skl_metrics_dict['skl_sensitivity'] + skl_metrics_dict['skl_specificity_score'] - 1)

        else:
            cnf_matrix = skl_metrics.confusion_matrix(y_true, y_pred)
            # print(cnf_matrix)
            num_classes = cnf_matrix.shape[0]

            tp = np.diag(cnf_matrix)
            fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            tn = cnf_matrix.sum() - (fp + fn + tp)

            tpr = tp / (tp + fn)
            sensitivity = tpr
            specificity_score = tn / (tn + fp)

            for class_row in cnf_matrix:
                support.append(sum(class_row))

            if average == 'micro':
                tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

                skl_metrics_dict['skl_tpr'] = tp / (tp + fn)
                skl_metrics_dict['skl_tnr'] = tn / (tn + fp)
                skl_metrics_dict['skl_fnr'] = fn / (fn + tp)
                skl_metrics_dict['skl_fpr'] = fp / (fp + tn)
                skl_metrics_dict['skl_sensitivity'] = skl_metrics_dict['skl_tpr']
                skl_metrics_dict['skl_specificity_score'] = tn / (tn + fp)
                skl_metrics_dict['skl_critical_success_index'] = tp / (tp + fp + fn)
                skl_metrics_dict['skl_diagnostic_odds_ratio'] = (tp * tn) / (fp + fn)
                skl_metrics_dict['skl_false_negative_rate'] = fn / (fn + tp)
                skl_metrics_dict['skl_false_positive_rate'] = fp / (fp + tn)
                skl_metrics_dict['skl_false_discovery_rate'] = fp / (fp + tp)
                skl_metrics_dict['skl_false_omission_rate'] = fn / (fn + tn)
                skl_metrics_dict['skl_fowlkes_mallows_index'] = tp / np.sqrt((tp + fp) * (tp + fn))
                skl_metrics_dict['skl_informedness'] = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
                skl_metrics_dict['skl_markedness'] = (tp / (tp + fp)) + (tn / (tn + fn)) - 1
                skl_metrics_dict['skl_negative_predictive_score'] = tn / (tn + fn)
                skl_metrics_dict['skl_negative_likelihood_ratio'] = (fn / (fn + tp)) / (tn / (tn + fp))
                skl_metrics_dict['skl_positive_likelihood_ratio'] = (tp / (tp + fn)) / (fp / (fp + tn))
                skl_metrics_dict['skl_prevalence_threshold'] = (np.sqrt(skl_metrics_dict['skl_sensitivity'] * (-skl_metrics_dict['skl_specificity_score'] + 1)) + skl_metrics_dict['skl_specificity_score'] - 1) / (skl_metrics_dict['skl_sensitivity'] + skl_metrics_dict['skl_specificity_score'] - 1)

            elif average in ['macro', 'weighted']:
                support_proportion = sum(support)
                # print(skl_metrics.classification_report(y_true, y_pred))

                # print(tp, fp, fn, tn)
                for num_class in range(num_classes):
                    skl_metrics_dict['skl_per_class_critical_success_index'] = skl_metrics_dict.get('skl_per_class_critical_success_index', [])
                    skl_metrics_dict['skl_per_class_critical_success_index'].append(tp[num_class] / (tp[num_class] + fp[num_class] + fn[num_class]))

                    skl_metrics_dict['skl_per_class_diagnostic_odds_ratio'] = skl_metrics_dict.get('skl_per_class_diagnostic_odds_ratio', [])
                    skl_metrics_dict['skl_per_class_diagnostic_odds_ratio'].append((tp[num_class] * tn[num_class]) / (fp[num_class] + fn[num_class]))

                    skl_metrics_dict['skl_per_class_false_negative_rate'] = skl_metrics_dict.get('skl_per_class_false_negative_rate', [])
                    skl_metrics_dict['skl_per_class_false_negative_rate'].append(fn[num_class] / (fn[num_class] + tp[num_class]))

                    skl_metrics_dict['skl_per_class_false_positive_rate'] = skl_metrics_dict.get('skl_per_class_false_positive_rate', [])
                    skl_metrics_dict['skl_per_class_false_positive_rate'].append(fp[num_class] / (fp[num_class] + tn[num_class]))

                    skl_metrics_dict['skl_per_class_false_discovery_rate'] = skl_metrics_dict.get('skl_per_class_false_discovery_rate', [])
                    skl_metrics_dict['skl_per_class_false_discovery_rate'].append(fp[num_class] / (fp[num_class] + tp[num_class]))

                    skl_metrics_dict['skl_per_class_false_omission_rate'] = skl_metrics_dict.get('skl_per_class_false_omission_rate', [])
                    skl_metrics_dict['skl_per_class_false_omission_rate'].append(fn[num_class] / (fn[num_class] + tn[num_class]))

                    skl_metrics_dict['skl_per_class_fowlkes_mallows_index'] = skl_metrics_dict.get('skl_per_class_fowlkes_mallows_index', [])
                    skl_metrics_dict['skl_per_class_fowlkes_mallows_index'].append(tp[num_class] / np.sqrt((tp[num_class] + fp[num_class])*(tp[num_class] + fn[num_class])))

                    skl_metrics_dict['skl_per_class_informedness'] = skl_metrics_dict.get('skl_per_class_informedness', [])
                    skl_metrics_dict['skl_per_class_informedness'].append((tp[num_class] / (tp[num_class] + fn[num_class])) + (tn[num_class] / (tn[num_class] + fp[num_class])) - 1 )

                    skl_metrics_dict['skl_per_class_markedness'] = skl_metrics_dict.get('skl_per_class_markedness', [])
                    skl_metrics_dict['skl_per_class_markedness'].append((tp[num_class] / (tp[num_class] + fp[num_class])) + (tn[num_class] / (tn[num_class] + fn[num_class])) - 1)

                    skl_metrics_dict['skl_per_class_negative_predictive_score'] = skl_metrics_dict.get('skl_per_class_negative_predictive_score', [])
                    skl_metrics_dict['skl_per_class_negative_predictive_score'].append(tn[num_class]/(tn[num_class] + fn[num_class]))

                    skl_metrics_dict['skl_per_class_negative_likelihood_ratio'] = skl_metrics_dict.get('skl_per_class_negative_likelihood_ratio', [])
                    skl_metrics_dict['skl_per_class_negative_likelihood_ratio'].append((fn[num_class] / (fn[num_class] + tp[num_class])) / (tn[num_class] / (tn[num_class] + fp[num_class])))

                    skl_metrics_dict['skl_per_class_positive_likelihood_ratio'] = skl_metrics_dict.get('skl_per_class_positive_likelihood_ratio', [])
                    skl_metrics_dict['skl_per_class_positive_likelihood_ratio'].append((tp[num_class] / (tp[num_class] + fn[num_class])) / (fp[num_class] / (fp[num_class] + tn[num_class])))

                    skl_metrics_dict['skl_per_class_prevalence_threshold'] = skl_metrics_dict.get('skl_per_class_prevalence_threshold', [])
                    skl_metrics_dict['skl_per_class_prevalence_threshold'].append((np.sqrt(sensitivity[num_class]*(-specificity_score[num_class] + 1)) + specificity_score[num_class]-1) / (sensitivity[num_class] + specificity_score[num_class] - 1))

                    skl_metrics_dict['skl_per_class_specificity_score'] = skl_metrics_dict.get('skl_per_class_specificity_score', [])
                    skl_metrics_dict['skl_per_class_specificity_score'].append(specificity_score[num_class])

                if average == 'macro':
                    skl_metrics_dict[f'skl_{classification_metric}'] = sum(skl_metrics_dict[f'skl_per_class_{classification_metric}']) / num_classes
                elif average == 'weighted':
                    skl_metrics_dict[f'skl_{classification_metric}'] = 0

                    for per_class_support, per_class_csi in zip(support, skl_metrics_dict[f'skl_per_class_{classification_metric}']):
                        skl_metrics_dict[f'skl_{classification_metric}'] += per_class_csi*(per_class_support/support_proportion)

        return skl_metrics_dict

    def test_master_classification_metrics(self, compute_method, dataset, func_args, expected, is_skl_metrics,
                                           classification_metrics, request):

        vdf, y_true, y_pred, y_prob, labels = request.getfixturevalue(dataset)

        if 'labels' in func_args: func_args['labels'] = labels

        # verticapy logic
        if classification_metrics in ['confusion_matrix']:
            vpy_res = getattr(vpy_metrics, classification_metrics)("y_true", "y_pred", vdf, **func_args)
        elif classification_metrics in ['roc_auc_score', 'log_loss']:
            vpy_res = getattr(vpy_metrics, classification_metrics)("y_true", "y_prob", vdf, average=compute_method, **func_args)
        elif classification_metrics in ['classification_report']:
            vpy_res = getattr(vpy_metrics, classification_metrics)("y_true", ["y_prob", "y_pred"], vdf, labels=['a', 'b'])
            # print(vpy_res)
        else:
            print(compute_method, func_args)
            vpy_res = getattr(vpy_metrics, classification_metrics)("y_true", "y_pred", vdf, average=compute_method, **func_args)

        # sklearn logic
        if is_skl_metrics == 'y':
            if classification_metrics in ['accuracy_score', 'matthews_corrcoef', 'balanced_accuracy_score']:
                skl_res = getattr(skl_metrics, classification_metrics)(y_true, y_pred)
            elif classification_metrics in ['precision_score', 'recall_score']:
                skl_res = getattr(skl_metrics, classification_metrics)(y_true, y_pred, average=compute_method, **func_args)
            elif classification_metrics in ['f1_score']:
                func_args['labels'] = 'b' if 'labels' in func_args else func_args['pos_label']
                skl_res = getattr(skl_metrics, classification_metrics)(y_true, y_pred, average=compute_method, pos_label='b')
            elif classification_metrics in ['roc_auc_score', 'log_loss']:
                skl_res = getattr(skl_metrics, classification_metrics)(y_true, y_prob, labels=labels)
            elif classification_metrics in ['classification_report']:
                report = getattr(skl_metrics, classification_metrics)(y_true, y_prob, labels=labels)
                skl_res = pd.DataFrame(report).transpose()
                print(skl_res)
            else:
                skl_res = getattr(skl_metrics, classification_metrics)(y_true, y_pred, labels=labels)
        else:
            skl_metrics_dict = self.skl_metrics_cal(y_true, y_pred, average=compute_method, classification_metric=classification_metrics)
            skl_res = skl_metrics_dict[f'skl_{classification_metrics}']

        print(f'vertica: {vpy_res}, sklearn: {skl_res}')
        assert vpy_res == pytest.approx(skl_res)
