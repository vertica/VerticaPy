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
import copy
from typing import Literal

import numpy as np

from verticapy._typing import ArrayLike, NoneType
from verticapy._utils._sql._format import format_magic


from verticapy.machine_learning.memmodel.base import InMemoryModel


class Scaler(InMemoryModel):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of scalers.

    Parameters
    ----------
    sub: ArrayLike
        Model's features first aggregation.
    den: ArrayLike
        Model's features second aggregation.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').
    """

    # Properties.

    @property
    def object_type(self) -> Literal["Scaler"]:
        return "Scaler"

    @property
    def _attributes(self) -> list[str]:
        return ["sub_", "den_"]

    # System & Special Methods.

    def __init__(self, sub: ArrayLike, den: ArrayLike) -> None:
        self.sub_ = np.array(sub)
        self.den_ = np.array(den)

    # Prediction / Transformation Methods - IN MEMORY.

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transforms  and applies the scaler model to  the
        input matrix.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the transformation.

        Returns
        -------
        numpy.array
            Transformed values.
        """
        return (np.array(X) - self.sub_) / self.den_

    # Prediction / Transformation Methods - IN DATABASE.

    def transform_sql(self, X: ArrayLike) -> list[str]:
        """
        Transforms and returns the SQL needed to deploy
        the Scaler.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        if not len(X) == len(self.den_) == len(self.sub_):
            raise ValueError(
                "The length of parameter 'X' must be equal to the length "
                "of the vector 'sub' and 'den'."
            )
        return [f"({X[i]} - {self.sub_[i]}) / {self.den_[i]}" for i in range(len(X))]


class StandardScaler(Scaler):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of standard scaler.

    Parameters
    ----------
    mean: ArrayLike
        Model's features averages.
    std: ArrayLike
        Model's features standard deviations.

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel` are defined
        entirely by their attributes. For example, 'mean', and
        'standard deviation' of feature(S) define a StandardScaler
        model.

    Attributes
    ----------
    Attributes are identical to
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.Scaler`.

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.preprocessing import StandardScaler

    A StandardScaler model is defined by mean and standard deviation
    values. In this example, we will use the following:

    .. ipython:: python

        mean = [0.4, 0.1]
        std = [0.5, 0.2]

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.StandardScaler`
    model.

    .. ipython:: python

        model_sts = StandardScaler(mean, std)

    Create a dataset.

    .. ipython:: python

        data = [[0.45, 0.17]]

    **Making In-Memory Transformation**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.StandardScaler.transform`
    method to do transformation.

    .. ipython:: python

        model_sts.transform(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.StandardScaler.transform_sql`
    method to get the SQL code needed to deploy the model using its attributes.

    .. ipython:: python

        model_mms.transform_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["StandardScaler"]:
        return "StandardScaler"

    # System & Special Methods.

    def __init__(self, mean: ArrayLike, std: ArrayLike) -> None:
        self.sub_ = np.array(mean)
        self.den_ = np.array(std)


class MinMaxScaler(Scaler):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of MinMax scaler.

    Parameters
    ----------

    min_: ArrayLike
        Model's features minimums.
    max_: ArrayLike
        Model's features maximums.

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel` are defined
        entirely by their attributes. For example, 'minimum',
        and 'maximum' values of the input features define a
        MinMaxScaler model.

    Attributes
    ----------
    Attributes are identical to
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.Scaler`.

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.preprocessing import MinMaxScaler

    A MinMaxScaler model is defined by minimum and maximum values.
    In this example, we will use the following:

    .. ipython:: python

        min = [0.4, 0.1]
        max = [0.5, 0.2]

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.MinMaxScaler`
    model.

    .. ipython:: python

        model_mms = MinMaxScaler(min, max)

    Create a dataset.

    .. ipython:: python

        data = [[0.45, 0.17]]

    **Making In-Memory Transformation**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.MinMaxScaler.transform`
    method to do transformation.

    .. ipython:: python

        model_mms.transform(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.MinMaxScaler.transform_sql`
    method to get the SQL code needed to deploy the model using its attributes.

    .. ipython:: python

        model_mms.transform_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["MinMaxScaler"]:
        return "MinMaxScaler"

    # System & Special Methods.

    def __init__(self, min_: ArrayLike, max_: ArrayLike) -> None:
        self.sub_ = np.array(min_)
        self.den_ = np.array(max_) - np.array(min_)


class OneHotEncoder(InMemoryModel):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of one-hot encoder.

    Parameters
    ----------

    categories: ArrayLike
        ArrayLike  of the categories of  the different  features.
    column_naming: str, optional
        Appends  categorical  levels  to column  names  according
        to the specified method:

        - indices              : Uses integer  indices to represent
            categorical levels.

        - values/values_relaxed: Both methods use categorical level names.
            If duplicate column  names occur,  the  function attempts  to
            disambiguate them by appending _n, where  n  is a zero-based
            integer index (_0, _1,…).

    drop_first: bool, optional
        If set to False, the first dummy of each category is
        dropped.

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel` are defined
        entirely by their attributes. For example, 'categories' to
        encode defines a OneHotEncoder model. You can optionally
        provide 'column naming' criteria and a 'drop_first' flag to
        denote whether to drop first dummy of each category.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.preprocessing import OneHotEncoder

    A OneHotEncoder model is defined by categories, column naming
    criteria and drop_first flag.

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.OneHotEncoder`
    model.

    .. ipython:: python

        model_ohe = OneHotEncoder(
            categories = [["male", "female"], [1, 2, 3]],
            drop_first = False,
            column_naming = None,
        )

    Create a dataset.

    .. ipython:: python

        data = [["male", 1], ["female", 3]]

    **Making In-Memory Transformation**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.OneHotEncoder.transform`
    method to do transformation.

    .. ipython:: python

        model_ohe.transform(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['sex', 'pclass']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.preprocessing.OneHotEncoder.transform_sql`
    method to get the SQL code needed to deploy the model using its attributes.

    .. ipython:: python

        model_ohe.transform_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["OneHotEncoder"]:
        return "OneHotEncoder"

    @property
    def _attributes(self) -> list[str]:
        return ["categories_", "column_naming_", "drop_first_"]

    # System & Special Methods.

    def __init__(
        self,
        categories: ArrayLike,
        column_naming: Literal["indices", "values", "values_relaxed"] = "indices",
        drop_first: bool = True,
    ) -> None:
        self.categories_ = copy.deepcopy(categories)
        self.column_naming_ = column_naming
        self.drop_first_ = drop_first

    # Prediction / Transformation Methods - IN MEMORY.

    def _transform_row(self, X: ArrayLike) -> list:
        """
        Transforms and applies the OneHotEncoder model to the
        input row.
        """
        X_trans = []
        for i, x in enumerate(X):
            for j, c in enumerate(self.categories_[i]):
                if j != 0 or not self.drop_first_:
                    if str(x) == str(c):
                        X_trans += [1]
                    else:
                        X_trans += [0]
        return X_trans

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transforms and applies the OneHotEncoder model to the
        input matrix.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the transformation.

        Returns
        -------
        numpy.array
            Transformed values.
        """
        return np.apply_along_axis(self._transform_row, 1, X)

    # Prediction / Transformation Methods - IN DATABASE.

    def transform_sql(self, X: ArrayLike) -> list[str]:
        """
        Transforms and returns the SQL needed to deploy the Scaler.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        if len(X) != len(self.categories_):
            raise ValueError(
                "The length of parameter 'X' must be equal to the "
                "length of the attribute 'categories'."
            )
        sql = []
        for i in range(len(X)):
            sql_tmp = []
            for j in range(len(self.categories_[i])):
                if not self.drop_first_ or j > 0:
                    val = format_magic(self.categories_[i][j])
                    sql_tmp_feature = f"(CASE WHEN {X[i]} = {val} THEN 1 ELSE 0 END)"
                    X_i = str(X[i]).replace('"', "")
                    if self.column_naming_ == "indices":
                        sql_tmp_feature += f' AS "{X_i}_{j}"'
                    elif self.column_naming_ in ("values", "values_relaxed"):
                        if not isinstance(self.categories_[i][j], NoneType):
                            categories_i_j = self.categories_[i][j]
                        else:
                            categories_i_j = "NULL"
                        sql_tmp_feature += f' AS "{X_i}_{categories_i_j}"'
                    sql_tmp += [sql_tmp_feature]
            sql += [sql_tmp]
        return sql
