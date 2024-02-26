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
import itertools
import pytest

from verticapy.pipeline._validate import testing

from verticapy._utils._sql._sys import _executeSQL

from verticapy import drop
from verticapy.datasets import load_winequality 
from verticapy.pipeline._train import training
import verticapy.sql.sys as sys

from verticapy.machine_learning.vertica.linear_model import LinearRegression

class TestValidate:
    """
    test class for Transform tests
    """
    
    """
    Analytic Functions test
    - 
    """
    @pytest.mark.parametrize(
        "test",
        [
            {'metric': {
                    'name': 'aic',
                    'y_true': 'quality',
                    'y_score': 'prediction',
                    }
            },
            {'metric': {
                    'name': 'bic',
                    'y_true': 'quality',
                    'y_score': 'prediction',
                    }
            },
            {'metric': {
                    'name': 'max_error',
                    'y_true': 'quality',
                    'y_score': 'prediction',
                    }
            },
            {'metric': {
                    'name': 'mean_absolute_error',
                    'y_true': 'quality',
                    'y_score': 'prediction',
                    }
            },
            {'metric': {
                    'name': 'mean_squared_error',
                    'y_true': 'quality',
                    'y_score': 'prediction',
                    }
            },
            {'metric': {
                    'name': 'mean_squared_log_error',
                    'y_true': 'quality',
                    'y_score': 'prediction',
                    }
            },
            {'metric': {
                    'name': 'r2',
                    'y_true': 'quality',
                    'y_score': 'prediction',
                    }
            },
        ],
    )
    def test_regression(
        self,
        test,
    ):

        pipeline_name = 'test_pipeline'
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")
        
        # Model Setup
        table = load_winequality()
        kwargs = {'method': 
                    {'name': 'LinearRegression',
                     'target': 'quality',
                     'params': {
                        'tol': 1e-6,
                        'max_iter': 100,
                        'solver': 'newton',
                        'fit_intercept': True
                     }
                    }
                }

        cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar']
        

        # Part 1: Train a Model
        meta_sql, model, model_sql = training(kwargs, table, 'test_pipeline', cols)
        
        # Part 2: Run the Metrics
        res = testing(test, model, pipeline_name, cols)
        
        assert True
        assert model

        assert sys.does_view_exist("test_pipeline_TRAIN_VIEW", 'public')
        assert sys.does_view_exist("test_pipeline_TEST_VIEW", 'public')

        # drop pipeline
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")
        drop('public.winequality')
