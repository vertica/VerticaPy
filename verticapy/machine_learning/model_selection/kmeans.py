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
from typing import Union, Literal

# VerticaPy Modules
from verticapy._utils._collect import save_verticapy_logs
from verticapy.core.vdataframe.vdataframe import vDataFrame
from verticapy.core.tablesample import tablesample
from verticapy._config.config import ISNOTEBOOK
from verticapy.plotting._colors import gen_colors
from verticapy.plotting._matplotlib.base import updated_dict
from verticapy._config.config import OPTIONS
from verticapy.sql._utils._format import schema_relation

# Other Python Modules
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


@save_verticapy_logs
def best_k(
    input_relation: Union[str, vDataFrame],
    X: Union[str, list] = [],
    n_cluster: Union[tuple, list] = (1, 100),
    init: Literal["kmeanspp", "random", None] = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    use_kprototype: bool = False,
    gamma: float = 1.0,
    elbow_score_stop: float = 0.8,
    **kwargs,
):
    """
Finds the k-means / k-prototypes k based on a score.

Parameters
----------
input_relation: str/vDataFrame
    Relation to use to train the model.
X: str / list, optional
	List of the predictor columns. If empty, all numerical columns will
    be used.
n_cluster: tuple/list, optional
	Tuple representing the number of clusters to start and end with.
    This can also be customized list with various k values to test.
init: str/list, optional
	The method used to find the initial cluster centers.
		kmeanspp : Use the k-means++ method to initialize the centers.
                   [Only available when use_kprototype is set to False]
        random   : Randomly subsamples the data to find initial centers.
    Default value is 'kmeanspp' if use_kprototype is False; otherwise, 'random'.
max_iter: int, optional
	The maximum number of iterations for the algorithm.
tol: float, optional
	Determines whether the algorithm has converged. The algorithm is considered 
	converged after no center has moved more than a distance of 'tol' from the 
	previous iteration.
use_kprototype: bool, optional
    If set to True, the function uses the k-prototypes algorithm instead of
    k-means. k-prototypes can handle categorical features.
gamma: float, optional
    [Only if use_kprototype is set to True] Weighting factor for categorical columns. 
    It determines the relative importance of numerical and categorical attributes.
elbow_score_stop: float, optional
	Stops searching for parameters when the specified elbow score is reached.

Returns
-------
int
	the k-means / k-prototypes k
	"""
    if isinstance(X, str):
        X = [X]
    if not (init) and (use_kprototype):
        init = "random"
    elif not (init):
        init = "kmeanspp"

    from verticapy.learn.cluster import KMeans, KPrototypes

    if isinstance(n_cluster, tuple):
        L = range(n_cluster[0], n_cluster[1])
    else:
        L = n_cluster
        L.sort()
    schema, relation = schema_relation(input_relation)
    if not (schema):
        schema = OPTIONS["temp_schema"]
    schema = quote_ident(schema)
    if OPTIONS["tqdm"] and (
        "tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"])
    ):
        loop = tqdm(L)
    else:
        loop = L
    for i in loop:
        model_name = gen_tmp_name(schema=schema, name="kmeans")
        if use_kprototype:
            if init == "kmeanspp":
                init = "random"
            model = KPrototypes(model_name, i, init, max_iter, tol, gamma)
        else:
            model = KMeans(model_name, i, init, max_iter, tol)
        model.fit(input_relation, X)
        score = model.metrics_.values["value"][3]
        if score > elbow_score_stop:
            return i
        score_prev = score
        model.drop()
    print(
        f"\u26A0 The K was not found. The last K (= {i}) "
        f"is returned with an elbow score of {score}"
    )
    return i


@save_verticapy_logs
def elbow(
    input_relation: Union[str, vDataFrame],
    X: Union[str, list] = [],
    n_cluster: Union[tuple, list] = (1, 15),
    init: Literal["kmeanspp", "random", None] = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    use_kprototype: bool = False,
    gamma: float = 1.0,
    ax=None,
    **style_kwds,
):
    """
Draws an elbow curve.

Parameters
----------
input_relation: str / vDataFrame
    Relation to use to train the model.
X: str / list, optional
    List of the predictor columns. If empty all the numerical vcolumns will
    be used.
n_cluster: tuple / list, optional
    Tuple representing the number of cluster to start with and to end with.
    It can also be customized list with the different K to test.
init: str / list, optional
    The method used to find the initial cluster centers.
        kmeanspp : Use the k-means++ method to initialize the centers.
                   [Only available when use_kprototype is set to False]
        random   : Randomly subsamples the data to find initial centers.
    Default value is 'kmeanspp' if use_kprototype is False; otherwise, 'random'.
max_iter: int, optional
    The maximum number of iterations for the algorithm.
tol: float, optional
    Determines whether the algorithm has converged. The algorithm is considered 
    converged after no center has moved more than a distance of 'tol' from the 
    previous iteration.
use_kprototype: bool, optional
    If set to True, the function uses the k-prototypes algorithm instead of
    k-means. k-prototypes can handle categorical features.
gamma: float, optional
    [Only if use_kprototype is set to True] Weighting factor for categorical columns. 
    It determines the relative importance of numerical and categorical attributes.
ax: Matplotlib axes object, optional
    The axes to plot on.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    if isinstance(X, str):
        X = [X]
    if not (init) and (use_kprototype):
        init = "random"
    elif not (init):
        init = "kmeanspp"
    from verticapy.learn.cluster import KMeans, KPrototypes

    if isinstance(n_cluster, tuple):
        L = range(n_cluster[0], n_cluster[1])
    else:
        L = n_cluster
        L.sort()
    schema, relation = schema_relation(input_relation)
    all_within_cluster_SS, model_name = (
        [],
        gen_tmp_name(schema=schema, name="kmeans"),
    )
    if isinstance(n_cluster, tuple):
        L = [i for i in range(n_cluster[0], n_cluster[1])]
    else:
        L = n_cluster
        L.sort()
    if OPTIONS["tqdm"]:
        loop = tqdm(L)
    else:
        loop = L
    for i in loop:
        if use_kprototype:
            if init == "kmeanspp":
                init = "random"
            model = KPrototypes(model_name, i, init, max_iter, tol, gamma)
        else:
            model = KMeans(model_name, i, init, max_iter, tol)
        model.fit(input_relation, X)
        all_within_cluster_SS += [float(model.metrics_.values["value"][3])]
        model.drop()
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
            fig.set_size_inches(8, 6)
        ax.grid(axis="y")
    param = {
        "color": gen_colors()[0],
        "marker": "o",
        "markerfacecolor": "white",
        "markersize": 7,
        "markeredgecolor": "black",
    }
    ax.plot(L, all_within_cluster_SS, **updated_dict(param, style_kwds))
    ax.set_title("Elbow Curve")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Between-Cluster SS / Total SS")
    values = {"index": L, "Within-Cluster SS": all_within_cluster_SS}
    return tablesample(values=values)
