# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# VerticaPy Modules
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.learn.vmodel import *

# ---#
class MCA(Decomposition):
    """
---------------------------------------------------------------------------
Creates a MCA (multiple correspondence analysis) object using the Vertica PCA
algorithm on the data. It uses the property that the MCA is a PCA applied 
to a complete disjunctive table. The input relation is transformed to a
TCDT (transformed complete disjunctive table) before applying the PCA.
 
Parameters
----------
name: str
    Name of the the model. The model will be stored in the database.
cursor: DBcursor, optional
    Vertica database cursor.
    """
    def __init__(
        self,
        name: str,
        cursor=None,
    ):
        check_types([("name", name, [str], False)])
        self.type, self.name = "MCA", name
        self.set_params({})
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor
        version(cursor=cursor, condition=[9, 1, 0])

    # ---#
    def plot_var(
        self, dimensions: tuple = (1, 2), method: str = "auto", ax=None, **style_kwds
    ):
        """
    ---------------------------------------------------------------------------
    Draws the MCA (multiple correspondence analysis) graph.

    Parameters
    ----------
    dimensions: tuple, optional
        Tuple of two IDs of the model's components.
    method: str, optional
        Method used to draw the plot.
            auto   : Only the variables are displayed.
            cos2   : The cos2 is used as CMAP.
            contrib: The feature contribution is used as CMAP.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        check_types([("dimensions", dimensions, [tuple],),
                     ("method", method, ["auto", "cos2", "contrib"],),])
        x = self.components_["PC{}".format(dimensions[0])]
        y = self.components_["PC{}".format(dimensions[1])]
        n = len(self.cos2_["PC{}".format(dimensions[0])])
        if method in ("cos2", "contrib"):
            if method == "cos2":
                c = [self.cos2_["PC{}".format(dimensions[0])][i] + self.cos2_["PC{}".format(dimensions[1])][i] for i in range(n)]
            else:
                sum_1, sum_2 = sum(self.cos2_["PC{}".format(dimensions[0])]), sum(self.cos2_["PC{}".format(dimensions[1])])
                c = [0.5 * 100 * (self.cos2_["PC{}".format(dimensions[0])][i] / sum_1 + self.cos2_["PC{}".format(dimensions[1])][i] / sum_2) for i in range(n)]
            style_kwds["c"] = c
            if "cmap" not in style_kwds:
                from verticapy.plot import gen_cmap, gen_colors

                style_kwds["cmap"] = gen_cmap(color=[gen_colors()[0], gen_colors()[1], gen_colors()[2]])
        explained_variance = self.explained_variance_["explained_variance"]
        return plot_var(x, y, self.X, (explained_variance[dimensions[0] - 1], explained_variance[dimensions[1] - 1]), dimensions, method, ax, **style_kwds,)

    # ---#
    def plot_contrib(
        self, dimension: int = 1, ax=None, **style_kwds
    ):
        """
    ---------------------------------------------------------------------------
    Draws a decomposition contribution plot of the input dimension.

    Parameters
    ----------
    dimension: int, optional
        Integer representing the IDs of the model's component.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        contrib = self.components_["PC{}".format(dimension)]
        contrib = [elem ** 2 for elem in contrib]
        total = sum(contrib)
        contrib = [100 * elem / total for elem in contrib]
        n = len(contrib)
        variables, contribution = zip(*sorted(zip(self.X, contrib), key=lambda t: t[1], reverse=True))
        contrib = tablesample({"row_nb": [i + 1 for i in range(n)], "contrib": contribution}).to_vdf(self.cursor)
        contrib["row_nb_2"] = contrib["row_nb"] + 0.5
        ax = contrib["row_nb"].hist(method="avg", of="contrib", max_cardinality=1, h=1, ax=ax, **style_kwds)
        ax = contrib["contrib"].plot(ts="row_nb_2", ax=ax, color="black")
        ax.set_xlim(1, n + 1)
        ax.set_xticks([i + 1.5 for i in range(n)],)
        ax.set_xticklabels(variables,)
        ax.set_ylabel('Cos2 - Quality of Representation')
        ax.set_xlabel('')
        ax.set_title('Contribution of variables to Dim {}'.format(dimension))
        ax.plot([1, n + 1], [1 / n * 100, 1 / n * 100], c='r', linestyle='--',)
        for i in range(n):
            ax.text(i + 1.5, contribution[i] + 1, "{}%".format(round(contribution[i], 1)))
        return ax

    # ---#
    def plot_cos2(
        self, dimensions: tuple = (1, 2), ax=None, **style_kwds
    ):
        """
    ---------------------------------------------------------------------------
    Draws a MCA (multiple correspondence analysis) cos2 plot of 
    the two input dimensions.

    Parameters
    ----------
    dimensions: tuple, optional
        Tuple of two IDs of the model's components.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        cos2_1 = self.cos2_["PC{}".format(dimensions[0])]
        cos2_2 = self.cos2_["PC{}".format(dimensions[1])]
        n = len(cos2_1)
        quality = []
        for i in range(n):
            quality += [cos2_1[i] + cos2_2[i]]
        variables, quality = zip(*sorted(zip(self.X, quality), key=lambda t: t[1], reverse=True))
        quality = tablesample({"variables": variables, "quality": quality}).to_vdf(self.cursor)
        ax = quality["variables"].hist(method="avg", of="quality", max_cardinality=n, ax=ax, **style_kwds)
        ax.set_ylabel('Cos2 - Quality of Representation')
        ax.set_xlabel('')
        ax.set_title('Cos2 of variables to Dim {}-{}'.format(dimensions[0], dimensions[1]))
        return ax

# ---#
class PCA(Decomposition):
    """
---------------------------------------------------------------------------
Creates a PCA (Principal Component Analysis) object using the Vertica PCA
algorithm on the data.
 
Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica database cursor.
n_components: int, optional
	The number of components to keep in the model. If this value is not provided, 
	all components are kept. The maximum number of components is the number of 
	non-zero singular values returned by the internal call to SVD. This number is 
	less than or equal to SVD (number of columns, number of rows). 
scale: bool, optional
	A Boolean value that specifies whether to standardize the columns during the 
	preparation step.
method: str, optional
	The method to use to calculate PCA.
		lapack: Lapack definition.
	"""

    def __init__(
        self,
        name: str,
        cursor=None,
        n_components: int = 0,
        scale: bool = False,
        method: str = "lapack",
    ):
        check_types([("name", name, [str], False)])
        self.type, self.name = "PCA", name
        self.set_params(
            {"n_components": n_components, "scale": scale, "method": method.lower()}
        )
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor
        version(cursor=cursor, condition=[9, 1, 0])


# ---#
class SVD(Decomposition):
    """
---------------------------------------------------------------------------
Creates an SVD (Singular Value Decomposition) object using the Vertica SVD
algorithm on the data.
 
Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica database cursor.
n_components: int, optional
	The number of components to keep in the model. If this value is not provided, 
	all components are kept. The maximum number of components is the number of 
	non-zero singular values returned by the internal call to SVD. This number is 
	less than or equal to SVD (number of columns, number of rows).
method: str, optional
	The method to use to calculate SVD.
		lapack: Lapack definition.
	"""

    def __init__(
        self, name: str, cursor=None, n_components: int = 0, method: str = "lapack"
    ):
        check_types([("name", name, [str], False)])
        self.type, self.name = "SVD", name
        self.set_params({"n_components": n_components, "method": method.lower()})
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor
        version(cursor=cursor, condition=[9, 1, 0])
