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
from typing import Literal, Optional

from verticapy._typing import SQLColumns, SQLRelation
from verticapy._utils._gen import gen_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.base import VerticaModel


class PMMLModel(VerticaModel):
    """
    Creates a PMML object.

    .. versionadded:: 10.0.0

    Parameters
    ----------
    name: str, optional
        Name of the model. The  model must be stored in
        the database. If it is not the case, you can use
        :py:meth:`verticapy.machine_learning.vertica.import_models`
        to import your PMML model.

    Attributes
    ----------
    All attributes can be accessed using the
    :py:meth:`verticapy.machine_learning.vertica.base.VerticaModel.get_attributes``
    method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.base.VerticaModel.get_vertica_attributes``
        method.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> None:
        return None

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_PMML"]:
        return "PREDICT_PMML"

    @property
    def _model_category(self) -> Literal["INTEGRATION"]:
        return "INTEGRATION"

    @property
    def _model_subcategory(self) -> Literal["PMML"]:
        return "PMML"

    @property
    def _model_type(self) -> Literal["PMMLModel"]:
        return "PMMLModel"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__(name, False)
        self.parameters = {}
        self.X = self.get_attributes("data_fields")["name"]
        if self.get_attributes("is_supervised"):
            self.y = self.X[-1]
            self.X = self.X[:-1]

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: SQLRelation,
        X: SQLColumns,
        name: Optional[str] = None,
        inplace: bool = True,
    ) -> vDataFrame:
        """
        Predicts using the input relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object  used to run  the prediction.  You can
            also  specify a  customized  relation,  but you
            must  enclose  it with an alias.  For  example,
            "(SELECT 1) x" is valid, whereas "(SELECT 1)"
            and "SELECT 1" are invalid.
        X: SQLColumns
            List of the columns  used to deploy the models.
        name: str, optional
            Name of the added vDataColumn. If empty, a name
            is generated.
        inplace: bool, optional
            If set to True, the prediction is added to the
            vDataFrame.

        Returns
        -------
        vDataFrame
            the input object.
        """
        if isinstance(X, str):
            X = [X]
        X = quote_ident(X)
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
            inplace = True
        if not name:
            name = gen_name([self._model_type, self.model_name])
        if inplace:
            return vdf.eval(name, self.deploySQL(X=X))
        else:
            return vdf.copy().eval(name, self.deploySQL(X=X))
