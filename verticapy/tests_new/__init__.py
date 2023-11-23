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
from collections import namedtuple

AggregateFun = namedtuple("AggregateFun", ["vpy", "py"])
functions = {
    "aad": [
        "vpy_data.aad()",
        "np.absolute(py_data - py_data.mean(numeric_only=True)).mean(numeric_only=True)",
    ],
    "count": ["vpy_data.count()", "py_data.count()"],
    "cvar": [
        "vpy_data.cvar()",
        "py_data[py_data >= py_data.quantile(0.95, numeric_only=True)].mean(numeric_only=True)",
    ],
    "iqr": [
        "vpy_data.iqr()",
        "py_data.quantile(0.75, numeric_only=True) - py_data.quantile(0.25, numeric_only=True)",
    ],
    "kurt": ["vpy_data.kurt()", "py_data.kurt(numeric_only=True)"],
    "kurtosis": [
        "vpy_data.kurtosis()",
        "py_data.kurtosis(numeric_only=True)",
    ],
    "jb": ["vpy_data.jb()", "jarque_bera(py_data, nan_policy='omit').statistic"],
    "mad": [
        "vpy_data.mad()",
        "median_abs_deviation(py_data, nan_policy='omit')",
    ],
    "max": ["vpy_data.max()", "py_data.max(numeric_only=True)"],
    "mean": ["vpy_data.mean()", "py_data.mean(numeric_only=True)"],
    "avg": ["vpy_data.avg()", "py_data.mean(numeric_only=True)"],
    "median": ["vpy_data.median()", "py_data.median(numeric_only=True)"],
    "min": ["vpy_data.min()", "py_data.min(numeric_only=True)"],
    "mode": ["vpy_data.mode()", "py_data.mode(numeric_only=True, dropna=False).values"],
    "percent": ["vpy_data.percent()", "py_data.count()/len(py_data)*100"],
    "quantile": [
        "vpy_data.quantile(q=[0.2, 0.5])",
        "py_data.quantile(q=[0.2, 0.5],numeric_only=True).values",
    ],
    "10%": ["vpy_data.q10()", "py_data.quantile(0.1, numeric_only=True)"],
    "90%": ["vpy_data.q90", "py_data.quantile(0.9, numeric_only=True)"],
    "prod": ["vpy_data.prod()", "py_data.prod(numeric_only=True)"],
    "product": ["vpy_data.product()", "py_data.product(numeric_only=True)"],
    "range": [
        "vpy_data.range()",
        "py_data.max(numeric_only=True) - py_data.min(numeric_only=True)",
    ],
    "sem": ["vpy_data.sem()", "py_data.sem(numeric_only=True)"],
    "skew": ["vpy_data.skew()", "py_data.skew(numeric_only=True)"],
    "skewness": ["vpy_data.skewness()", "py_data.skew(numeric_only=True)"],
    "sum": ["vpy_data.sum()", "py_data.sum(numeric_only=True)"],
    "std": ["vpy_data.std()", "py_data.std(numeric_only=True)"],
    "stddev": ["vpy_data.stddev()", "py_data.std(numeric_only=True)"],
    "topk": ["vpy_data.topk(k=3)", "py_data.value_counts(dropna=False)"],
    "top1": ["vpy_data.topk(k=1)", "py_data.value_counts(dropna=False).index[0]"],
    "top1_percent": [
        "vpy_data.top1_percent()",
        "py_data.value_counts(dropna=False).iloc[0]/len(py_data)*100",
    ],
    "nunique": ["vpy_data.nunique(approx=False)", "py_data.nunique()"],
    "unique": ["vpy_data.nunique(approx=False)", "py_data.nunique()"],
    "var": ["vpy_data.var()", "py_data.var(numeric_only=True)"],
    "variance": ["vpy_data.variance()", "py_data.var(numeric_only=True)"],
    "value_counts": [
        "vpy_data.value_counts()",
        "py_data.value_counts(dropna=False)",
    ],
    "distinct": [
        "vpy_data.distinct()",
        "py_data.unique()",
    ],
}
