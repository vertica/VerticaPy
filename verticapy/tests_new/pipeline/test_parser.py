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
import pytest
from collections import defaultdict

from verticapy._utils._sql._sys import _executeSQL

from verticapy.pipeline import parser

import verticapy.sql.sys as sys


def build_pipeline(pipeline_name: str):
    steps = {'schema': 'public', 
             'pipeline': pipeline_name, 
             'table': 'public.winequality', 
             'steps': {
                    'transform': {
                        'col1': {
                            'sql': 'fixed_acidity'
                        }, 
                        'col2': {
                            'sql': 'volatile_acidity', 
                        }, 
                        'col3': {
                            'sql': 'citric_acid', 
                        }, 
                    }, 
                    'train': {
                        'train_test_split': {
                            'test_size': 0.34
                        }, 
                        'method': {
                            'name': 'LinearRegression',  
                            'target': 'quality', 
                        }, 
                        'schedule': '* * * * *'
                    },
                    'test': {
                        'metric1': {
                            'name': 'r2',
                            'y_true': 'quality',
                            'y_score': 'prediction'
                        }
                    }
                }
            }
    parser.parse_yaml(steps)


def test_parser():
    build_pipeline('test_pipeline')
    build_pipeline('test_pipeline') # Purposely test duplicates
    build_pipeline('test_pipeline_2')

    assert sys.does_view_exist("test_pipeline_TRAIN_VIEW", 'public')
    assert sys.does_view_exist("test_pipeline_TEST_VIEW", 'public')
    assert sys.does_table_exist("test_pipeline_METRIC_TABLE", 'public')
    
    assert sys.does_view_exist("test_pipeline_2_TRAIN_VIEW", 'public')
    assert sys.does_view_exist("test_pipeline_2_TEST_VIEW", 'public')
    assert sys.does_table_exist("test_pipeline_2_METRIC_TABLE", 'public')
