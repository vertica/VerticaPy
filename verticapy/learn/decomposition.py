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
