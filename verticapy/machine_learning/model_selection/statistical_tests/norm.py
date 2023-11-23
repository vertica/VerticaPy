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

    Examples
    ---------

    Let's try this test on two set of distribution to
    obverse the contrast in test results:

    - normally distributed dataset
    - uniformly distributed dataset

    Normally Distributed
    ^^^^^^^^^^^^^^^^^^^^^

    Import the necessary libraries:

    .. code-block:: python

        import verticapy as vp
        import numpy as np
        import random

    .. ipython:: python
        :suppress:

        import verticapy as vp
        import numpy as np
        import random
        N = 100
        mean = 0
        std_dev = 1
        data = np.random.normal(mean, std_dev, N)

    Then we can define the basic parameters for the
    normal distribution:

    .. code-block:: python

        # Distribution parameters
        N = 100 # Number of rows
        mean = 0
        std_dev = 1

        # Dataset
        data = np.random.normal(mean, std_dev, N)

    Now we can create the :py:class:`vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame({"col": data})

    We can visualize the distribution:

    .. code-block::

        vdf["col"].hist()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = vdf["col"].hist(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_jarque_bera_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_jarque_bera_1.html

    To find the test p-value, we can import the test function:

    .. ipython:: python

        from verticapy.machine_learning.model_selection.statistical_tests import jarque_bera

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        jarque_bera(vdf, column = "col")

    We can see that the p-value is high meaning that
    we cannot reject the null hypothesis. This is supported
    by the low Jarque-Bera Test Statistic value, providing
    further evidence that the distribution is normal.

    .. note::

        A ``p_value`` in statistics represents the
        probability of obtaining results as extreme
        as, or more extreme than, the observed data,
        assuming the null hypothesis is true.
        A *smaller* p-value typically suggests
        stronger evidence against the null hypothesis
        i.e. the test distribution does not belong
        to a normal distribution.

        However, *small* is a relative term. And
        the choice for the threshold value which
        determines a "small" should be made before
        analyzing the data.

        Generally a ``p-value`` less than 0.05
        is considered the threshold to reject the
        null hypothesis. But it is not always
        the case -
        `read more <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10232224/#:~:text=If%20the%20p%2Dvalue%20is,necessarily%20have%20to%20be%200.05.>`_

    Uniform Distribution
    ^^^^^^^^^^^^^^^^^^^^^

    .. ipython:: python
        :suppress:

        low = 0
        high = 1
        data = np.random.uniform(low, high, N)
        vdf = vp.vDataFrame({"col": data})

    We can define the basic parameters for the
    uniform distribution:

    .. code-block:: python

        # Distribution parameters
        low = 0
        high = 1

        # Dataset
        data = np.random.uniform(low, high, N)

        # vDataFrame
        vdf = vp.vDataFrame({"col": data})

    We can visualize the distribution:

    .. code-block::

        vdf["col"].hist()

    .. ipython:: python
        :suppress:

        fig = vdf["col"].hist(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_jarque_bera_2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_jarque_bera_2.html

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        jarque_bera(vdf, column = "col")

    .. note::

        In this case, the p-value is quite low
        meaning that it is highly probable that
        the data is not normally distributed.
        This is supported by the elevated Jarque-Bera
        Test Statistic value, providing further evidence
        that the distribution deviates from normality.
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

    Examples
    ---------

    Let's try this test on two set of distribution to
    obverse the contrast in test results:

    - normally distributed dataset
    - uniformly distributed dataset

    Normally Distributed
    ^^^^^^^^^^^^^^^^^^^^^

    Import the necessary libraries:

    .. code-block:: python

        import verticapy as vp
        import numpy as np
        import random

    .. ipython:: python
        :suppress:

        import verticapy as vp
        import numpy as np
        import random
        N = 100
        mean = 0
        std_dev = 1
        data = np.random.normal(mean, std_dev, N)

    Then we can define the basic parameters for the
    normal distribution:

    .. code-block:: python

        # Distribution parameters
        N = 100 # Number of rows
        mean = 0
        std_dev = 1

        # Dataset
        data = np.random.normal(mean, std_dev, N)

    Now we can create the :py:class:`vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame({"col": data})

    We can visualize the distribution:

    .. code-block::

        vdf["col"].hist()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = vdf["col"].hist(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_kurtosistest_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_kurtosistest_1.html

    To find the test p-value, we can import the test function:

    .. ipython:: python

        from verticapy.machine_learning.model_selection.statistical_tests import kurtosistest

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        kurtosistest(vdf, column = "col")

    We can see that the p-value is high meaning that
    we cannot reject the null hypothesis. This finding
    is corroborated by the Kurtoises Test Statistic,
    which is closer to 0, providing additional evidence
    that the distribution is normal.

    .. note::

        A ``p_value`` in statistics represents the
        probability of obtaining results as extreme
        as, or more extreme than, the observed data,
        assuming the null hypothesis is true.
        A *smaller* p-value typically suggests
        stronger evidence against the null hypothesis
        i.e. the test distribution does not belong
        to a normal distribution.

        However, *small* is a relative term. And
        the choice for the threshold value which
        determines a "small" should be made before
        analyzing the data.

        Generally a ``p-value`` less than 0.05
        is considered the threshold to reject the
        null hypothesis. But it is not always
        the case -
        `read more <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10232224/#:~:text=If%20the%20p%2Dvalue%20is,necessarily%20have%20to%20be%200.05.>`_

    Uniform Distribution
    ^^^^^^^^^^^^^^^^^^^^^

    .. ipython:: python
        :suppress:

        low = 0
        high = 1
        data = np.random.uniform(low, high, N)
        vdf = vp.vDataFrame({"col": data})

    We can define the basic parameters for the
    uniform distribution:

    .. code-block:: python

        # Distribution parameters
        low = 0
        high = 1

        # Dataset
        data = np.random.uniform(low, high, N)

        # vDataFrame
        vdf = vp.vDataFrame({"col": data})

    We can visualize the distribution:

    .. code-block::

        vdf["col"].hist()

    .. ipython:: python
        :suppress:

        fig = vdf["col"].hist(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_kurtosistest_2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_kurtosistest_2.html

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        kurtosistest(vdf, column = "col")

    .. note::

        In this case, the p-value is quite low
        meaning that it is highly probable that
        the data is not normally distributed.
        This finding is corroborated by the Kurtoises
        Test Statistic, which is away from 0, providing
        additional evidence that the distribution deviates
        from normality.
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
    This function tests the null hypothesis that a
    sample comes from a normal distribution.

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

    Examples
    ---------

    Let's try this test on two set of distribution to
    obverse the contrast in test results:

    - normally distributed dataset
    - uniformly distributed dataset

    Normally Distributed
    ^^^^^^^^^^^^^^^^^^^^^

    Import the necessary libraries:

    .. code-block:: python

        import verticapy as vp
        import numpy as np
        import random

    .. ipython:: python
        :suppress:

        import verticapy as vp
        import numpy as np
        import random
        N = 100
        mean = 0
        std_dev = 1
        data = np.random.normal(mean, std_dev, N)

    Then we can define the basic parameters for the
    normal distribution:

    .. code-block:: python

        # Distribution parameters
        N = 100 # Number of rows
        mean = 0
        std_dev = 1

        # Dataset
        data = np.random.normal(mean, std_dev, N)

    Now we can create the :py:class:`vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame({"col": data})

    We can visualize the distribution:

    .. code-block::

        vdf["col"].hist()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = vdf["col"].hist(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_normaltest_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_normaltest_1.html

    To find the test p-value, we can import the test function:

    .. ipython:: python

        from verticapy.machine_learning.model_selection.statistical_tests import normaltest

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        normaltest(vdf, column = "col")

    We can see that the p-value is high meaning that
    we cannot reject the null hypothesis. The low
    normal test statistic value further supports
    the conclusion that the distribution is normal.

    .. note::

        A ``p_value`` in statistics represents the
        probability of obtaining results as extreme
        as, or more extreme than, the observed data,
        assuming the null hypothesis is true.
        A *smaller* p-value typically suggests
        stronger evidence against the null hypothesis
        i.e. the test distribution does not belong
        to a normal distribution.

        However, *small* is a relative term. And
        the choice for the threshold value which
        determines a "small" should be made before
        analyzing the data.

        Generally a ``p-value`` less than 0.05
        is considered the threshold to reject the
        null hypothesis. But it is not always
        the case -
        `read more <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10232224/#:~:text=If%20the%20p%2Dvalue%20is,necessarily%20have%20to%20be%200.05.>`_

    Uniform Distribution
    ^^^^^^^^^^^^^^^^^^^^^

    .. ipython:: python
        :suppress:

        low = 0
        high = 1
        data = np.random.uniform(low, high, N)
        vdf = vp.vDataFrame({"col": data})

    We can define the basic parameters for the
    uniform distribution:

    .. code-block:: python

        # Distribution parameters
        low = 0
        high = 1

        # Dataset
        data = np.random.uniform(low, high, N)

        # vDataFrame
        vdf = vp.vDataFrame({"col": data})

    We can visualize the distribution:

    .. code-block::

        vdf["col"].hist()

    .. ipython:: python
        :suppress:

        fig = vdf["col"].hist(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_normaltest_2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_normaltest_2.html

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        normaltest(vdf, column = "col")

    .. note::

        In this case, the p-value is quite low
        meaning that it is highly probable that
        the data is not normally distributed.
        The high normal test statistic value further
        supports the conclusion that the distribution
        is not normal.
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

    Examples
    ---------

    Let's try this test on two set of distribution to
    obverse the contrast in test results:

    - normally distributed dataset
    - uniformly distributed dataset

    Normally Distributed
    ^^^^^^^^^^^^^^^^^^^^^

    Import the necessary libraries:

    .. code-block:: python

        import verticapy as vp
        import numpy as np
        import random

    .. ipython:: python
        :suppress:

        import verticapy as vp
        import numpy as np
        import random
        N = 100
        mean = 0
        std_dev = 1
        data = np.random.normal(mean, std_dev, N)

    Then we can define the basic parameters for the
    normal distribution:

    .. code-block:: python

        # Distribution parameters
        N = 100 # Number of rows
        mean = 0
        std_dev = 1

        # Dataset
        data = np.random.normal(mean, std_dev, N)

    Now we can create the :py:class:`vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame({"col": data})

    We can visualize the distribution:

    .. code-block::

        vdf["col"].hist()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = vdf["col"].hist(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_skewtest_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_skewtest_1.html

    To find the test p-value, we can import the test function:

    .. ipython:: python

        from verticapy.machine_learning.model_selection.statistical_tests import skewtest

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        skewtest(vdf, column = "col")

    We can see that the p-value is high meaning that
    we cannot reject the null hypothesis. The
    skewtest test statistic value closer to 0
    further corraborates that the distribution is
    normal.

    .. note::

        A **positive skewness** value suggests that
        the data distribution has a longer right
        tail. In other words, the distribution is
        skewed to the right, and the majority of
        the data points are concentrated on the left side.

        A **negative sekweness** value indicates that
        the data distribution has a longer left tail.
        The distribution is skewed to the left, and most
        of the data points are concentrated on the right side.

    .. note::

        A ``p_value`` in statistics represents the
        probability of obtaining results as extreme
        as, or more extreme than, the observed data,
        assuming the null hypothesis is true.
        A *smaller* p-value typically suggests
        stronger evidence against the null hypothesis
        i.e. the test distribution does not belong
        to a normal distribution.

        However, *small* is a relative term. And
        the choice for the threshold value which
        determines a "small" should be made before
        analyzing the data.

        Generally a ``p-value`` less than 0.05
        is considered the threshold to reject the
        null hypothesis. But it is not always
        the case -
        `read more <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10232224/#:~:text=If%20the%20p%2Dvalue%20is,necessarily%20have%20to%20be%200.05.>`_

    Uniform Distribution
    ^^^^^^^^^^^^^^^^^^^^^

    .. ipython:: python
        :suppress:

        low = 0
        high = 1
        data = np.random.uniform(low, high, N)
        vdf = vp.vDataFrame({"col": data})

    We can define the basic parameters for the
    uniform distribution:

    .. code-block:: python

        # Distribution parameters
        low = 0
        high = 1

        # Dataset
        data = np.random.uniform(low, high, N)

        # vDataFrame
        vdf = vp.vDataFrame({"col": data})

    We can visualize the distribution:

    .. code-block::

        vdf["col"].hist()

    .. ipython:: python
        :suppress:

        fig = vdf["col"].hist(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_skewtest_2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_norm_skewtest_2.html

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        skewtest(vdf, column = "col")

    .. note::

        In this case, the p-value is quite low
        meaning that it is highly probable that
        the data is not normally distributed.
        The skewtest test statistic value away from 0
        further confirms that the distribution is
        not normal.
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
