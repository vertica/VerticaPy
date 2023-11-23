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
from typing import Literal, Optional, Union
from tqdm.auto import tqdm

import verticapy._config.config as conf
from verticapy._typing import ArrayLike, SQLColumns, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs

from verticapy.machine_learning.model_selection import best_k
from verticapy.machine_learning.vertica.automl.dataprep import AutoDataPrep
from verticapy.machine_learning.vertica.base import VerticaModel
from verticapy.machine_learning.vertica.cluster import KMeans, KPrototypes


class AutoClustering(VerticaModel):
    """
    Automatically creates k different groups with which to
    generalize the data.

    Parameters
    ----------
    name: str, optional
        Name of the model.
    overwrite_model: bool, optional
        If set to True, training a model with the same name
        as an existing model overwrites the existing model.
    n_cluster: int, optional
        Number  of clusters. If empty, an optimal number  of
        clusters  are determined using multiple  k-means
        models.
    init: str / list, optional
        The method for finding the initial cluster  centers.
            kmeanspp : Uses   the    k-means++   method   to
                       initialize the centers.
                       [Only available  when  use_kprototype
                        is set to False]
            random   : Randomly  subsamples the data to find
                       initial centers.
        Alternatively,  you  can  specify  a list  with  the
        initial cluster centers.
    max_iter: int, optional
        The maximum number of  iterations for the algorithm.
    tol: float, optional
        Determines whether the algorithm has converged. The
        algorithm  is considered converged after no  center
        has  moved more than  a distance of 'tol' from  the
        previous
        iteration.
    use_kprototype: bool, optional
        If set to True,  the function uses the k-prototypes
        algorithm  instead  of  k-means.  k-prototypes  can
        handle categorical features.
    gamma: float, optional
        [Only  if use_kprototype is set to True]  Weighting
        factor  for categorical columns. It determines  the
        relative  importance  of numerical and  categorical
        attributes.
    preprocess_data: bool, optional
        If True, the data will be preprocessed.
    preprocess_dict: dict, optional
        Dictionary  to pass to  the  AutoDataPrep class  in
        order to preprocess the data before clustering.
    print_info: bool
        If True, prints the model information at each step.

    Attributes
    ----------
    preprocess_: object
        Model used to preprocess the data.
    model_: object
        Final model used for clustering.
    """

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _vertica_fit_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_predict_sql(self) -> Literal[""]:
        return ""

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["CLUSTERING"]:
        return "CLUSTERING"

    @property
    def _model_type(self) -> Literal["AutoClustering"]:
        return "AutoClustering"

    @property
    def _attributes(self) -> list[str]:
        return ["preprocess_", "model_"]

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self,
        name: Optional[str] = None,
        overwrite_model: bool = False,
        n_cluster: Optional[int] = None,
        init: Union[Literal["kmeanspp", "random"], ArrayLike] = "kmeanspp",
        max_iter: int = 300,
        tol: float = 1e-4,
        use_kprototype: bool = False,
        gamma: float = 1.0,
        preprocess_data: bool = True,
        preprocess_dict: dict = {
            "identify_ts": False,
            "standardize_min_cat": 0,
            "outliers_threshold": 3.0,
            "na_method": "drop",
        },
        print_info: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_cluster": n_cluster,
            "init": init,
            "max_iter": max_iter,
            "tol": tol,
            "use_kprototype": use_kprototype,
            "gamma": gamma,
            "print_info": print_info,
            "preprocess_data": preprocess_data,
            "preprocess_dict": preprocess_dict,
        }

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        X: Optional[SQLColumns] = None,
        return_report: bool = False,
    ) -> None:
        """
        Trains the model.

        Parameters
        ----------
        input_relation: SQLRelation
            Training Relation.
        X: SQLColumns, optional
            List of the predictors.
        """
        if self.overwrite_model:
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mStarting AutoClustering\033[0m\033[0m\n")
        if self.parameters["preprocess_data"]:
            model_preprocess = AutoDataPrep(**self.parameters["preprocess_dict"])
            model_preprocess.fit(input_relation, X=X)
            input_relation = model_preprocess.final_relation_
            X = copy.deepcopy(model_preprocess.X_out_)
            self.preprocess_ = model_preprocess
        else:
            self.preprocess_ = None
        if not self.parameters["n_cluster"]:
            if self.parameters["print_info"]:
                print(
                    f"\033[1m\033[4mFinding a suitable number of clusters\033[0m\033[0m\n"
                )
            self.parameters["n_cluster"] = best_k(
                input_relation=input_relation,
                X=X,
                n_cluster=(1, 100),
                init=self.parameters["init"],
                max_iter=self.parameters["max_iter"],
                tol=self.parameters["tol"],
                use_kprototype=self.parameters["use_kprototype"],
                gamma=self.parameters["gamma"],
                elbow_score_stop=0.9,
                tqdm=self.parameters["print_info"],
            )
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mBuilding the Final Model\033[0m\033[0m\n")
        if conf.get_option("tqdm") and self.parameters["print_info"]:
            loop = tqdm(range(1))
        else:
            loop = range(1)
        for i in loop:
            if self.parameters["use_kprototype"]:
                self.model_ = KPrototypes(
                    self.model_name,
                    n_cluster=self.parameters["n_cluster"],
                    init=self.parameters["init"],
                    max_iter=self.parameters["max_iter"],
                    tol=self.parameters["tol"],
                    gamma=self.parameters["gamma"],
                )
            else:
                self.model_ = KMeans(
                    self.model_name,
                    n_cluster=self.parameters["n_cluster"],
                    init=self.parameters["init"],
                    max_iter=self.parameters["max_iter"],
                    tol=self.parameters["tol"],
                )
            self.model_.fit(input_relation, X=X)
