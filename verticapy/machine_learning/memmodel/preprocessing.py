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
import copy
from typing import Literal

import numpy as np

from verticapy._typing import ArrayLike, NoneType
from verticapy._utils._sql._format import format_magic


from verticapy.machine_learning.memmodel.base import InMemoryModel


class Scaler(InMemoryModel):
    """
    InMemoryModel implementation of scalers.

    Parameters
    ----------
    sub: ArrayLike
        Model's features first aggregation.
    den: ArrayLike
        Model's features second aggregation.
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
    InMemoryModel implementation of standard scaler.

    Parameters
    ----------
    mean: ArrayLike
        Model's features averages.
    std: ArrayLike
        Model's features standard deviations.
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
    InMemoryModel implementation of MinMax scaler.

    Parameters
    ----------
    min_: ArrayLike
        Model's features minimums.
    max_: ArrayLike
        Model's features maximums.
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
    InMemoryModel implementation of one-hot encoder.

    Parameters
    ----------
    categories: ArrayLike
        ArrayLike  of the categories of  the different  features.
    column_naming: str, optional
        Appends  categorical  levels  to column  names  according
        to the specified method:
        indices              : Uses integer  indices to represent
                               categorical levels.
        values/values_relaxed: Both methods use categorical level
                               names.  If duplicate column  names
                               occur,  the  function attempts  to
                               disambiguate them by appending _n,
                               where  n  is a zero-based  integer
                               index (_0, _1,…).
    drop_first: bool, optional
        If set to False, the first dummy of each category is
        dropped.
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
