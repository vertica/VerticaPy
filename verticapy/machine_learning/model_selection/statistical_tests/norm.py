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
import math
from scipy.stats import chi2, norm

from verticapy._typing import SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs

from verticapy.core.vdataframe.base import vDataFrame

"""
Normality Tests.
"""


@save_verticapy_logs
def jarque_bera(input_relation: SQLRelation, column: str) -> tuple[float, float]:
    """
    Jarque-Bera test (Distribution Normality).

    Parameters
    ----------
    input_relation: SQLRelation
        Input relation.
    column: str
        Input vDataColumn to test.

    Returns
    -------
    tuple
        statistic, p_value
    """
    if isinstance(input_relation, vDataFrame):
        vdf = input_relation.copy()
    else:
        vdf = vDataFrame(input_relation)
    column = vdf.format_colnames(column)
    jb = vdf[column].agg(["jb"]).values[column][0]
    pvalue = chi2.sf(jb, 2)
    return jb, pvalue


@save_verticapy_logs
def kurtosistest(input_relation: SQLRelation, column: str) -> tuple[float, float]:
    """
    Test whether the kurtosis is different from the
    Normal distribution.

    Parameters
    ----------
    input_relation: SQLRelation
        Input relation.
    column: str
        Input vDataColumn to test.

    Returns
    -------
    tuple
        statistic, p_value
    """
    if isinstance(input_relation, vDataFrame):
        vdf = input_relation.copy()
    else:
        vdf = vDataFrame(input_relation)
    column = vdf.format_colnames(column)
    g2, n = vdf[column].agg(["kurtosis", "count"]).values[column]
    mu1 = -6 / (n + 1)
    mu2 = 24 * n * (n - 2) * (n - 3) / (((n + 1) ** 2) * (n + 3) * (n + 5))
    gamma1 = 6 * (n**2 - 5 * n + 2) / ((n + 7) * (n + 9))
    gamma1 = gamma1 * math.sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    A = 6 + 8 / gamma1 * (2 / gamma1 + math.sqrt(1 + 4 / (gamma1**2)))
    B = (1 - 2 / A) / (1 + (g2 - mu1) / math.sqrt(mu2) * math.sqrt(2 / (A - 4)))
    B = B ** (1 / 3) if B > 0 else (-B) ** (1 / 3)
    Z2 = math.sqrt(9 * A / 2) * (1 - 2 / (9 * A) - B)
    pvalue = 2 * norm.sf(abs(Z2))
    return Z2, pvalue


@save_verticapy_logs
def normaltest(input_relation: SQLRelation, column: str) -> tuple[float, float]:
    """
    Test whether a sample differs from a normal
    distribution.

    Parameters
    ----------
    input_relation: SQLRelation
        Input relation.
    column: str
        Input vDataColumn to test.

    Returns
    -------
    tuple
        statistic, p_value
    """
    if isinstance(input_relation, vDataFrame):
        vdf = input_relation.copy()
    else:
        vdf = vDataFrame(input_relation)
    Z1 = skewtest(vdf, column)[0]
    Z2 = kurtosistest(vdf, column)[0]
    Z = Z1**2 + Z2**2
    pvalue = chi2.sf(Z, 2)
    return Z, pvalue


@save_verticapy_logs
def skewtest(input_relation: SQLRelation, column: str) -> tuple[float, float]:
    """
    Test whether the skewness is different from the
    normal distribution.

    Parameters
    ----------
    input_relation: SQLRelation
        Input relation.
    column: str
        Input vDataColumn to test.

    Returns
    -------
    tuple
        statistic, p_value
    """
    if isinstance(input_relation, vDataFrame):
        vdf = input_relation.copy()
    else:
        vdf = vDataFrame(input_relation)
    column = vdf.format_colnames(column)
    g1, n = vdf[column].agg(["skewness", "count"]).values[column]
    mu2 = 6 * (n - 2) / ((n + 1) * (n + 3))
    gamma2 = 36 * (n - 7) * (n**2 + 2 * n - 5)
    gamma2 = gamma2 / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    W2 = math.sqrt(2 * gamma2 + 4) - 1
    delta = 1 / math.sqrt(math.log(math.sqrt(W2)))
    alpha2 = 2 / (W2 - 1)
    Z1 = delta * math.asinh(g1 / math.sqrt(alpha2 * mu2))
    pvalue = 2 * norm.sf(abs(Z1))
    return Z1, pvalue
