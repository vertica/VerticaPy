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
from typing import Literal, Optional, Union
import numpy as np

from matplotlib.axes import Axes

from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.machine_learning.vertica.base import BinaryClassifier, Regressor
from verticapy.machine_learning.vertica.linear_model import (
    LinearModel,
    LinearModelClassifier,
)

import verticapy.plotting._matplotlib as vpy_plt

"""
Algorithms used for regression.
"""


class LinearSVR(Regressor, LinearModel):
    """
    Creates  a  LinearSVR  object  using the Vertica  SVM 
    (Support Vector Machine)  algorithm.  This  algorithm 
    finds the hyperplane used to approximate distribution 
    of the data.

    Parameters
    ----------
    name: str
        Name of the model.  The model will be 
        stored in the DB.
    tol: float, optional
        To use to control accuracy.
    C: float, optional
        The  weight  for  misclassification  cost. 
        The algorithm minimizes the regularization 
        cost and the misclassification cost.
    fit_intercept: bool, optional
        A bool to fit also the intercept.
    intercept_scaling: float
        A  float value, serves  as the value of a 
        dummy feature  whose  coefficient Vertica  
        uses to calculate the model intercept. 
        Because  the dummy feature is not in  the 
        training data,  its values  are  set to a 
        constant, by default set to 1. 
    intercept_mode: str, optional
        Specify how to treat the intercept.
            regularized   : Fits  the intercept  and 
                            applies a regularization 
                            on it.
            unregularized : Fits  the  intercept but 
                            does  not include it  in 
                            regularization. 
    acceptable_error_margin: float, optional
        Defines the acceptable error margin. Any data 
        points  outside this region add a penalty  to 
        the cost function. 
    max_iter: int, optional
        The  maximum  number of iterations  that  the 
        algorithm performs.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["SVM_REGRESSOR"]:
        return "SVM_REGRESSOR"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_SVM_REGRESSOR"]:
        return "PREDICT_SVM_REGRESSOR"

    @property
    def _model_category(self) -> Literal["SUPERVISED"]:
        return "SUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["REGRESSOR"]:
        return "REGRESSOR"

    @property
    def _model_type(self) -> Literal["LinearSVR"]:
        return "LinearSVR"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        intercept_mode: Literal["regularized", "unregularized"] = "regularized",
        acceptable_error_margin: float = 0.1,
        max_iter: int = 100,
    ) -> None:
        self.model_name = name
        self.parameters = {
            "tol": tol,
            "C": C,
            "fit_intercept": fit_intercept,
            "intercept_scaling": intercept_scaling,
            "intercept_mode": str(intercept_mode).lower(),
            "acceptable_error_margin": acceptable_error_margin,
            "max_iter": max_iter,
        }
        return None

    # Parameters Methods.

    @staticmethod
    def _map_to_vertica_param_dict() -> dict:
        return {
            "tol": "epsilon",
            "max_iter": "max_iterations",
        }


"""
Algorithms used for classification.
"""


class LinearSVC(BinaryClassifier, LinearModelClassifier):
    """
    Creates  a LinearSVC object  using the  Vertica
    Support Vector Machine  (SVM)  algorithm on the 
    data.  Given a set of  training examples,  each 
    marked as belonging to  one or the other of two 
    categories, an SVM  training algorithm builds a 
    model that assigns new examples to one category 
    or  the other,  making it  a  non-probabilistic 
    binary linear classifier.

    Parameters
    ----------
    name: str
    	Name  of the  model. The model will  be 
        stored in the DB.
    tol: float, optional
    	to use to control accuracy.
    C: float, optional
    	The weight for misclassification cost.  The 
        algorithm minimizes the regularization cost 
        and the misclassification cost.
    fit_intercept: bool, optional
    	A bool to fit also the intercept.
    intercept_scaling: float
    	A  float  value,  serves as  the  value of a 
        dummy feature whose coefficient Vertica uses
        to calculate the model intercept. 
    	Because  the  dummy  feature  is not in  the 
        training  data,  its  values  are  set to  a 
        constant, by default set to 1. 
    intercept_mode: str, optional
    	Specify how to treat the intercept.
    		regularized   : Fits  the intercept  and 
                            applies a regularization 
                            on it.
    		unregularized : Fits the  intercept  but 
                            does not  include  it in 
                            regularization. 
    class_weight: str / list, optional
    	Specifies how to determine weights of the two 
        classes.  It can be a  list of 2 elements  or 
        one of the following method:
    		auto : Weights  each class  according  to 
                   the number of samples.
    		none : No weights are used.
    max_iter: int, optional
    	The  maximum  number of iterations  that  the 
        algorithm performs.
	"""

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["SVM_CLASSIFIER"]:
        return "SVM_CLASSIFIER"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_SVM_CLASSIFIER"]:
        return "PREDICT_SVM_CLASSIFIER"

    @property
    def _model_category(self) -> Literal["SUPERVISED"]:
        return "SUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["LinearSVC"]:
        return "LinearSVC"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        intercept_mode: Literal["regularized", "unregularized"] = "regularized",
        class_weight: Union[Literal["auto", "none"], list] = [1, 1],
        max_iter: int = 100,
    ) -> None:
        self.model_name = name
        self.parameters = {
            "tol": tol,
            "C": C,
            "fit_intercept": fit_intercept,
            "intercept_scaling": intercept_scaling,
            "intercept_mode": str(intercept_mode).lower(),
            "class_weight": class_weight,
            "max_iter": max_iter,
        }
        return None

    # Parameters Methods.

    @staticmethod
    def _map_to_vertica_param_dict() -> dict[str, str]:
        return {
            "class_weights": "class_weight",
            "tol": "epsilon",
            "max_iter": "max_iterations",
        }

    # Plotting Methods.

    def plot(
        self, max_nb_points: int = 100, ax: Optional[Axes] = None, **style_kwds
    ) -> Axes:
        """
        Draws the model.

        Parameters
        ----------
        max_nb_points: int
            Maximum  number of points to display.
        ax: Axes, optional
            The axes to plot on.
        **style_kwds
            Any optional parameter to pass to the 
            Matplotlib functions.

        Returns
        -------
        Axes
            Axes.
        """
        return vpy_plt.SVMClassifierPlot().svm_classifier_plot(
            self.X,
            self.y,
            self.input_relation,
            np.concatenate(([self.intercept_], self.coef_)),
            max_nb_points,
            ax=ax,
            **style_kwds,
        )
