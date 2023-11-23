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

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.machine_learning.vertica.base import VerticaModel
from verticapy.machine_learning.vertica.model_management import load_model
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL

from verticapy._typing import (
    NoneType,
    PythonNumber,
    PythonScalar,
    SQLColumns,
    SQLRelation,
)


class RegisteredModel:
    """
    Creates a RegisteredModel object to load and use a registered model.
    This class is intended to be used mostly by a user with an
    MLSUPERVISOR role. Nevertheless, an ordinary users can also use it
    to monitor the status of their registered models.

    Parameters
    ----------
    registered_name: str
        This is a registered model name that represents all the versions
        of models registered with this name.
    """

    @save_verticapy_logs
    def __init__(self, registered_name: str) -> None:
        self.registered_name = registered_name

        # it raises an error if it cannot find any model with such a registered name
        query = (
            f"SELECT count(*) FROM v_catalog.registered_models "
            f"WHERE registered_name='{registered_name}';"
        )
        res = _executeSQL(
            query, title="searching for registered models", method="fetchfirstelem"
        )

        if res == 0:
            raise ValueError(f"Cannot find any model registered as {registered_name}")

    def list_models(self) -> TableSample:
        """
        Returns the list of the registered models with the same name that
        the user has USAGE privilege on.

        Returns
        -------
        TableSample
            List of the registered models with the same name
        """
        query = (
            f"SELECT * FROM v_catalog.registered_models "
            f"WHERE registered_name='{self.registered_name}';"
        )

        return TableSample.read_sql(query, title="Listing the registered models")

    def predict(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: Optional[str] = None,
        cutoff: Optional[PythonNumber] = None,
        inplace: bool = True,
        version: int = None,
    ) -> vDataFrame:
        """
        Makes predictions on the input relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object  used to run  the prediction.  You can
            also  specify a  customized  relation,  but you
            must  enclose  it with an alias.  For  example,
            "(SELECT 1) x" is valid, whereas "(SELECT 1)"
            and "SELECT 1" are invalid.
        X: SQLColumns, optional
            List of the columns  used to deploy the models.
            If empty, the model predictors are used.
        name: str, optional
            Name of the added vDataColumn. If empty, a name
            is generated.
        cutoff: PythonNumber, optional
            Cutoff  for which  the tested category is
            accepted  as a  prediction.  This parameter  is
            only used for binary classification.
        inplace: bool, optional
            If set to True, the prediction is added to
            the vDataFrame.
        version: int, optional
            When the version is None, the registered model
            with "production" status will be used for
            prediction.
            When the version is specified, the registered
            model that version will be used.
            It will throw an error if it doesn't find
            such a model.

        Returns
        -------
        vDataFrame
            the input object.
        """
        model_object = self._loading_selected_model(version)

        if model_object._model_subcategory == "CLASSIFIER":
            if model_object._is_binary_classifier and isinstance(cutoff, NoneType):
                cutoff = 0.5
            return model_object.predict(
                vdf=vdf, X=X, name=name, cutoff=cutoff, inplace=inplace
            )

        return model_object.predict(vdf=vdf, X=X, name=name, inplace=inplace)

    def predict_proba(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: Optional[str] = None,
        pos_label: Optional[PythonScalar] = None,
        inplace: bool = True,
        version: int = None,
    ) -> vDataFrame:
        """
        Returns the model's probabilities using the input
        relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object  used to run  the prediction.  You can
            also  specify a  customized  relation,  but you
            must  enclose  it with an alias.  For  example,
            "(SELECT 1) x" is valid, whereas "(SELECT 1)"
            and "SELECT 1" are invalid.
        X: SQLColumns, optional
            List of the columns  used to deploy the models.
            If empty, the model predictors are used.
        name: str, optional
            Name of the added vDataColumn. If empty, a name
            is generated.
        pos_label: PythonScalar, optional
            Class  label.  For binary classification,  this
            can be either 1 or 0.
        inplace: bool, optional
            If set to True, the prediction is added to
            the vDataFrame.
        version: int, optional
            When the version is None, the registered model
            with "production" status will be used for
            prediction.
            When the version is specified, the registered
            model that version will be used.
            It will throw an error if it doesn't find
            such a model.

        Returns
        -------
        vDataFrame
            the input object.
        """
        model_object = self._loading_selected_model(version)

        if model_object._model_subcategory != "CLASSIFIER":
            raise RuntimeError(
                "The predict_proba function can be used only for classifiers"
            )

        return model_object.predict_proba(
            vdf=vdf, X=X, name=name, pos_label=pos_label, inplace=inplace
        )

    def _loading_selected_model(self, version: int = None) -> VerticaModel:
        """
        Loads a model with given version. When version
        is None, loads a model with PRODUCTION status.
        Raises an error if it can find a model.

        Parameters
        ----------
        version: int, optional
            When the version is None, the registered model
            with "production" status will be used for
            prediction.
            When the version is specified, the registered
            model that version will be used.
            It will throw an error if it doesn't find
            such a model.

        Returns
        -------
        VerticaModel
            the model object.
        """
        if version:
            model_query = (
                f"SELECT schema_name, model_name "
                f"FROM v_catalog.registered_models WHERE "
                f"registered_name='{self.registered_name}' "
                f"AND registered_version={version};"
            )
        else:
            model_query = (
                f"SELECT schema_name, model_name "
                f"FROM v_catalog.registered_models WHERE "
                f"registered_name='{self.registered_name}' "
                f"AND status='PRODUCTION';"
            )

        res = _executeSQL(model_query, title="finding model", method="fetchrow")

        if not res:
            if version:
                raise ValueError(
                    f"Cannot find any registered model named "
                    f"{self.registered_name} with version {version}"
                )
            raise ValueError(
                f"Cannot find any registered model named "
                f"{self.registered_name} in PRODUCTION status"
            )

        model_name = res[0] + "." + res[1]
        return load_model(name=model_name)

    def change_status(
        self,
        version: int,
        new_status: Literal[
            "under_review",
            "staging",
            "production",
            "archived",
            "declined",
            "unregistered",
        ],
    ) -> None:
        """
        Changes the status of a model.
        Only a user with SUPERUSER or MLSUPERVISOR privilege can
        successfully run this function.

        Parameters
        ----------
        version: int
            The version of the model which its status should be changed.
        new_status: str
            The desired new status for the model.

        Returns
        -------
        None
        """
        query = (
            f"SELECT change_model_status('{self.registered_name}', "
            f"{version}, '{new_status}');"
        )
        _executeSQL(query, title="changing model status")

    def list_status_history(self, version: int = None) -> TableSample:
        """
        Returns the model status-change history.
        Only SUPERUSER and users who are granted SELECT privilege
        on the v_monitor.model_status_history table will be able
        to successfully run this function.

        Parameters
        ----------
        version: int
            The version of the model which its history will be listed.
            The history of all versions will be listed when version is None

        Returns
        -------
        TableSample
            The status-change history
        """
        if version:
            query = (
                f"SELECT * FROM v_monitor.model_status_history WHERE "
                f"registered_name='{self.registered_name}' "
                f"AND registered_version={version};"
            )
        else:
            query = (
                f"SELECT * FROM v_monitor.model_status_history WHERE "
                f"registered_name='{self.registered_name}';"
            )

        return TableSample.read_sql(query, title="Listing model_status_history")
