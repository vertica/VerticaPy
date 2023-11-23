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
from verticapy._utils._sql._format import format_type, quote_ident
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.base import VerticaModel


class TensorFlowModel(VerticaModel):
    """
    Creates a TensorFlow object.

    .. versionadded:: 10.0.0

    Parameters
    ----------
    name: str, optional
        Name of the model. The  model must be stored in
        the database. If it is not the case, you can use
        :py:meth:`verticapy.machine_learning.vertica.import_models`
        to import your TensorFlow model.

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
    def _vertica_predict_sql(self) -> Literal["PREDICT_TENSORFLOW_SCALAR"]:
        return "PREDICT_TENSORFLOW_SCALAR"

    @property
    def _vertica_predict_transform_sql(self) -> Literal["PREDICT_TENSORFLOW"]:
        return "PREDICT_TENSORFLOW"

    @property
    def _model_category(self) -> Literal["INTEGRATION"]:
        return "INTEGRATION"

    @property
    def _model_subcategory(self) -> Literal["TENSORFLOW"]:
        return "TENSORFLOW"

    @property
    def _model_type(self) -> Literal["TensorFlowModel"]:
        return "TensorFlowModel"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__(name, False)
        self.parameters = {}
        attr = self.get_attributes()
        if "input_desc" in attr:
            self.X = self.get_attributes("input_desc")["op_name"]
        if "output_desc" in attr:
            self.y = self.get_attributes("output_desc")["op_name"][0]

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
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

            .. note::

                This parameter is only used when the input
                'X' is a complex data type, otherwise it is
                ignored.
        inplace: bool, optional
            If set to True, the prediction is added to the
            vDataFrame.

        Returns
        -------
        vDataFrame
            the input object.
        """
        X = format_type(X, dtype=list, na_out=self.X)
        X = quote_ident(X)
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
            inplace = True
        if not name:
            name = gen_name([self._model_type, self.model_name])
        if len(X) == 1 and vdf[X[0]].category() == "complex":
            if inplace:
                return vdf.eval(name, self.deploySQL(X=X))
            else:
                return vdf.copy().eval(name, self.deploySQL(X=X))
        else:
            columns = vdf.get_columns()
            n = len(columns)
            sql = f"""
            SELECT
                {self._vertica_predict_transform_sql}({', '.join(columns + X)}
                                                      USING PARAMETERS 
                                                      model_name = '{self.model_name}', 
                                                      num_passthru_cols = '{n}')
                                                      OVER(PARTITION BEST) FROM {vdf}"""
            if inplace:
                return vdf.__init__(sql)
            else:
                return vDataFrame(sql)
