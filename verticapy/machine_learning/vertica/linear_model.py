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
from typing import Literal, Optional, Union
import numpy as np

from matplotlib.axes import Axes

from verticapy._typing import PythonNumber
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import (
    check_minimum_version,
    vertica_version,
)
from verticapy.errors import VersionError

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.vertica.base import Regressor, BinaryClassifier

import verticapy.plotting._matplotlib as vpy_matplotlib_plt

"""
General Classes.
"""


class LinearModel:

    # Properties.

    @property
    def _attributes(self) -> list[str]:
        return ["coef_", "intercept_", "features_importance_"]

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        details = self.get_vertica_attributes("details")
        self.coef_ = np.array(details["coefficient"][1:])
        self.intercept_ = details["coefficient"][0]
        return None

    # Features Importance Methods.

    def _compute_features_importance(self) -> None:
        """
        Computes the features importance.
        """
        vertica_version(condition=[8, 1, 1])
        query = f"""
        SELECT /*+LABEL('learn.VerticaModel.features_importance')*/
            predictor, 
            sign * ROUND(100 * importance / SUM(importance) OVER(), 2) AS importance
        FROM (SELECT 
                stat.predictor AS predictor, 
                ABS(coefficient * (max - min))::float AS importance, 
                SIGN(coefficient)::int AS sign 
              FROM (SELECT 
                        LOWER("column") AS predictor, 
                        min, 
                        max 
                    FROM (SELECT 
                            SUMMARIZE_NUMCOL({', '.join(self.X)}) OVER() 
                          FROM {self.input_relation}) VERTICAPY_SUBTABLE) stat 
                          NATURAL JOIN (SELECT GET_MODEL_ATTRIBUTE(
                                                USING PARAMETERS 
                                                model_name='{self.model_name}',
                                                attr_name='details') ) coeff) importance_t 
                          ORDER BY 2 DESC;"""
        importance = _executeSQL(
            query=query, title="Computing Features Importance.", method="fetchall"
        )
        self.features_importance_ = self._format_vector(self.X, importance)

    def _get_features_importance(self) -> np.ndarray:
        """
        Returns the features' importance.
        """
        if not (hasattr(self, "features_importance_")):
            self._compute_features_importance()
        return copy.deepcopy(self.features_importance_)

    def features_importance(
        self, show: bool = True, ax: Optional[Axes] = None, **style_kwargs
    ) -> TableSample:
        """
        Computes the model's features importance.

        Parameters
        ----------
        show: bool
            If set to True,  draw the features  importance.
        ax: Axes, optional
            The axes to plot on.
        **style_kwargs
            Any optional parameter to pass to the Matplotlib 
            functions.

        Returns
        -------
        TableSample
            features importance.
        """
        fi = self._get_features_importance()
        if show:
            data = {
                "importance": fi,
            }
            layout = {"columns": copy.deepcopy(self.X)}
            vpy_plt, kwargs = self._get_plotting_lib(
                matplotlib_kwargs={"ax": ax,}, style_kwargs=style_kwargs,
            )
            return vpy_plt.ImportanceBarChart(data=data, layout=layout).draw(**kwargs)
        importances = {
            "index": [quote_ident(x)[1:-1].lower() for x in self.X],
            "importance": list(abs(fi)),
            "sign": list(np.sign(fi)),
        }
        return TableSample(values=importances).sort(column="importance", desc=True)

    # I/O Methods.

    def to_memmodel(self) -> mm.LinearModel:
        """
        Converts  the model to an InMemory object  which
        can be used to do different types of predictions.
        """
        return mm.LinearModel(self.coef_, self.intercept_)

    # Plotting Methods.

    def plot(
        self, max_nb_points: int = 100, ax: Optional[Axes] = None, **style_kwargs
    ) -> Axes:
        """
        Draws the model.

        Parameters
        ----------
        max_nb_points: int
            Maximum number of points to display.
        ax: Axes, optional
            The axes to plot on.
        **style_kwargs
            Any optional parameter to pass to the 
            Matplotlib functions.

        Returns
        -------
        Axes
            Axes.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax,}, style_kwargs=style_kwargs,
        )
        return vpy_plt.RegressionPlot(
            vdf=vDataFrame(self.input_relation),
            columns=self.X + [self.y],
            max_nb_points=max_nb_points,
            misc_data={"coef": np.concatenate(([self.intercept_], self.coef_))},
        ).draw(**kwargs)


class LinearModelClassifier(LinearModel):

    # Properties.

    @property
    def _attributes(self) -> list[str]:
        return ["coef_", "intercept_", "classes_", "features_importance_"]

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        details = self.get_vertica_attributes("details")
        self.coef_ = np.array(details["coefficient"][1:])
        self.intercept_ = details["coefficient"][0]
        return None

    # I/O Methods.

    def to_memmodel(self) -> mm.LinearModelClassifier:
        """
        Converts  the  model  to an InMemory object which
        can be used to do different types of predictions.
        """
        return mm.LinearModelClassifier(self.coef_, self.intercept_)

    # Plotting Methods.

    def plot(
        self, max_nb_points: int = 100, ax: Optional[Axes] = None, **style_kwargs
    ) -> Axes:
        """
        Draws the model.

        Parameters
        ----------
        max_nb_points: int
            Maximum number of points to display.
        ax: Axes, optional
            The axes to plot on.
        **style_kwargs
            Any optional parameter to pass to the 
            Matplotlib functions.

        Returns
        -------
        Axes
            Axes.
        """
        vpy_plt, kwargs = self._get_plotting_lib(
            matplotlib_kwargs={"ax": ax,}, style_kwargs=style_kwargs,
        )
        return vpy_plt.LogisticRegressionPlot(
            vdf=vDataFrame(self.input_relation),
            columns=self.X + [self.y],
            max_nb_points=max_nb_points,
            misc_data={"coef": np.concatenate(([self.intercept_], self.coef_))},
        ).draw(**kwargs)


"""
Algorithms used for regression.
"""


class ElasticNet(Regressor, LinearModel):
    """
    Creates a ElasticNet  object using the Vertica 
    Linear Regression  algorithm  on the data. The 
    Elastic Net is a regularized regression method 
    that linearly combines the L1 and L2 penalties 
    of the Lasso and Ridge methods.

    Parameters
    ----------
    name: str
    	Name of the  model.  The model will be stored 
        in the DB.
    tol: float, optional
    	Determines  whether the algorithm has reached 
        the specified accuracy result.
    C: PythonNumber, optional
    	The regularization parameter value. The value 
        must be zero or non-negative.
    max_iter: int, optional
    	Determines  the maximum number of  iterations 
        the  algorithm performs before  achieving the 
        specified accuracy result.
    solver: str, optional
    	The optimizer method to use to train the model. 
    		newton : Newton Method.
    		bfgs   : Broyden Fletcher Goldfarb Shanno.
    		cgd    : Coordinate Gradient Descent.
    l1_ratio: float, optional
    	ENet mixture parameter that defines how much L1 
        versus L2 regularization to provide.
    fit_intercept: bool, optional
        Boolean,  specifies whether the model includes an 
        intercept. By setting to false, no intercept will 
        be used in training the model.  Note that setting 
        fit_intercept  to false does  not work well  with 
        the BFGS optimizer.
	"""

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["LINEAR_REG"]:
        return "LINEAR_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LINEAR_REG"]:
        return "PREDICT_LINEAR_REG"

    @property
    def _model_category(self) -> Literal["SUPERVISED"]:
        return "SUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearRegression"]:
        return "LinearRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        C: PythonNumber = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs", "cgd"] = "cgd",
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
    ) -> None:
        self.model_name = name
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "enet",
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "l1_ratio": l1_ratio,
            "fit_intercept": fit_intercept,
        }
        return None


class Lasso(Regressor, LinearModel):
    """
    Creates  a  Lasso  object using the  Vertica 
    Linear  Regression  algorithm  on the  data. 
    The Lasso is a regularized regression method 
    which uses an L1 penalty.

    Parameters
    ----------
    name: str
    	Name of the model. The  model  will  be stored 
        in the DB.
    tol: float, optional
    	Determines  whether the  algorithm has reached 
        the specified accuracy result.
    C: PythonNumber, optional
        The regularization  parameter value. The value 
        must be zero or non-negative.
    max_iter: int, optional
    	Determines  the  maximum number of  iterations 
        the  algorithm  performs before achieving  the 
        specified accuracy result.
    solver: str, optional
    	The optimizer method to use to train the model. 
    		newton : Newton Method.
    		bfgs   : Broyden Fletcher Goldfarb Shanno.
    		cgd    : Coordinate Gradient Descent.
    fit_intercept: bool, optional
        Boolean,  specifies  whether the model includes an 
        intercept.  By setting to false, no intercept will 
        be used in  training the model.  Note that setting 
        fit_intercept to false does not work well with the 
        BFGS optimizer.
	"""

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["LINEAR_REG"]:
        return "LINEAR_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LINEAR_REG"]:
        return "PREDICT_LINEAR_REG"

    @property
    def _model_category(self) -> Literal["SUPERVISED"]:
        return "SUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearRegression"]:
        return "LinearRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        C: PythonNumber = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs", "cgd"] = "cgd",
        fit_intercept: bool = True,
    ) -> None:
        self.model_name = name
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "l1",
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "fit_intercept": fit_intercept,
        }
        return None


class LinearRegression(Regressor, LinearModel):
    """
    Creates a LinearRegression object using the Vertica 
    Linear Regression algorithm on the data.

    Parameters
    ----------
    name: str
    	Name of the model. The  model will be stored  in 
        the DB.
    tol: float, optional
    	Determines whether the  algorithm has reached the 
        specified accuracy result.
    max_iter: int, optional
    	Determines the  maximum number of  iterations the 
        algorithm performs before achieving the specified 
        accuracy result.
    solver: str, optional
    	The optimizer method to use to train the model. 
    		newton : Newton Method.
    		bfgs   : Broyden Fletcher Goldfarb Shanno.
    fit_intercept: bool, optional
        Boolean,  specifies  whether the model includes an 
        intercept. By setting to false,  no intercept will 
        be used in  training the model.  Note that setting 
        fit_intercept to false does not work well with the 
        BFGS optimizer.
	"""

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["LINEAR_REG"]:
        return "LINEAR_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LINEAR_REG"]:
        return "PREDICT_LINEAR_REG"

    @property
    def _model_category(self) -> Literal["SUPERVISED"]:
        return "SUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearRegression"]:
        return "LinearRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs"] = "newton",
        fit_intercept: bool = True,
    ) -> None:
        self.model_name = name
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "none",
            "tol": tol,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "fit_intercept": fit_intercept,
        }
        return None


class Ridge(Regressor, LinearModel):
    """
    Creates  a  Ridge  object using the  Vertica 
    Linear  Regression  algorithm  on  the data. 
    The Ridge is a regularized regression method 
    which uses an L2 penalty. 

    Parameters
    ----------
    name: str
    	Name of the model.  The  model will be stored 
        in the DB.
    tol: float, optional
    	Determines whether  the algorithm has reached 
        the specified accuracy result.
    C: PythonNumber, optional
        The regularization parameter value. The value 
        must be zero or non-negative.
    max_iter: int, optional
    	Determines  the maximum number of  iterations 
        the  algorithm performs before  achieving the 
        specified accuracy result.
    solver: str, optional
    	The optimizer method to use to train the model. 
    		newton : Newton Method.
    		bfgs   : Broyden Fletcher Goldfarb Shanno.
    fit_intercept: bool, optional
        Boolean,  specifies whether the model  includes 
        an intercept. By setting to false, no intercept 
        will be used in training the model. 
        Note  that setting fit_intercept to false  does 
        not work well with the BFGS optimizer.
	"""

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["LINEAR_REG"]:
        return "LINEAR_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LINEAR_REG"]:
        return "PREDICT_LINEAR_REG"

    @property
    def _model_category(self) -> Literal["SUPERVISED"]:
        return "SUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearRegression"]:
        return "LinearRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        C: PythonNumber = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs"] = "newton",
        fit_intercept: bool = True,
    ) -> None:
        self.model_name = name
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "l2",
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "fit_intercept": fit_intercept,
        }
        return None


"""
Algorithms used for classification.
"""


class LogisticRegression(BinaryClassifier, LinearModelClassifier):
    """
    Creates a LogisticRegression  object using the Vertica 
    Logistic Regression algorithm on the data.

    Parameters
    ----------
    name: str
        Name of the model.  The model will be stored in the
        DB.
    penalty: str, optional
        Determines the method of regularization.
            None : No Regularization.
            l1   : L1 Regularization.
            l2   : L2 Regularization.
            enet : Combination between L1 and L2.
    tol: float, optional
        Determines  whether  the  algorithm has reached  the 
        specified accuracy result.
    C: PythonNumber, optional
        The  regularization parameter value.  The value must 
        be zero or non-negative.
    max_iter: int, optional
        Determines  the  maximum number  of  iterations  the 
        algorithm  performs  before achieving  the specified 
        accuracy result.
    solver: str, optional
        The  optimizer method  to use to train the  model. 
            newton : Newton Method.
            bfgs   : Broyden Fletcher Goldfarb Shanno.
            cgd    : Coordinate Gradient Descent.
    l1_ratio: float, optional
        ENet  mixture parameter  that  defines  how much  L1 
        versus L2 regularization to provide.
    fit_intercept: bool, optional
        Boolean,  specifies  whether  the model  includes an 
        intercept. 
        By  setting  to false,  no intercept will be used in 
        training the model.  Note that setting fit_intercept 
        to false does not work well with the BFGS optimizer.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["LOGISTIC_REG"]:
        return "LOGISTIC_REG"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_LOGISTIC_REG"]:
        return "PREDICT_LOGISTIC_REG"

    @property
    def _model_category(self) -> Literal["SUPERVISED"]:
        return "SUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["LogisticRegression"]:
        return "LogisticRegression"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        penalty: Literal["none", "l1", "l2", "enet", None] = "none",
        tol: float = 1e-6,
        C: PythonNumber = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs", "cgd"] = "newton",
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
    ) -> None:
        penalty = str(penalty).lower()
        solver = str(solver).lower()
        self.model_name = name
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise VersionError(
                "The parameter 'fit_intercept' can be activated for "
                "Vertica versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": penalty,
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": solver,
            "l1_ratio": l1_ratio,
            "fit_intercept": fit_intercept,
        }
        if str(penalty).lower() == "none":
            del self.parameters["l1_ratio"]
            del self.parameters["C"]
            if "solver" == "cgd":
                raise ValueError(
                    "solver can not be set to 'cgd' when there is no regularization."
                )
        elif str(penalty).lower() in ("l1", "l2"):
            del self.parameters["l1_ratio"]
        return None
