# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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
# Standard Python Modules
import warnings

# VerticaPy Modules
import verticapy, vertica_python
from verticapy.toolbox import *
from verticapy.utilities import *
from verticapy.stats import apply
from verticapy import vDataFrame

# ---#
def create_index(
    vdf: vDataFrame,
    gid: str,
    g: str,
    index: str,
    overwrite: bool = False,
    max_mem_mb: int = 256,
    skip_nonindexable_polygons: bool = False,
):
    """
---------------------------------------------------------------------------
Creates a spatial index on a set of polygons to speed up spatial intersection 
with a set of points.

Parameters
----------
vdf: vDataFrame
    vDataFrame to use to compute the spatial join.
gid: str
    Name of an integer column that uniquely identifies the polygon. The gid 
    cannot be NULL.
g: str
    Name of a geometry or geography (WGS84) column or expression that contains 
    polygons and multipolygons. Only polygon and multipolygon can be indexed. 
    Other shape types are excluded from the index.
index: str
    Name of the index.
overwrite: bool, optional
    BOOLEAN value that specifies whether to overwrite the index, if an index exists.
max_mem_mb: int, optional
    A positive integer that assigns a limit to the amount of memory in megabytes 
    that create_index can allocate during index construction.
skip_nonindexable_polygons: bool, optional
    In rare cases, intricate polygons (for instance, with too high resolution or 
    anomalous spikes) cannot be indexed. These polygons are considered non-indexable. 
    When set to False, non-indexable polygons cause the index creation to fail. 
    When set to True, index creation can succeed by excluding non-indexable polygons 
    from the index.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("vdf", vdf, [vDataFrame],),
            ("gid", gid, [str],),
            ("index", index, [str],),
            ("g", g, [str],),
            ("overwrite", overwrite, [bool],),
            ("max_mem_mb", max_mem_mb, [int],),
            ("skip_nonindexable_polygons", skip_nonindexable_polygons, [bool],),
        ]
    )
    columns_check([gid, g], vdf)
    gid, g = vdf_columns_names([gid, g], vdf)
    query = "SELECT STV_Create_Index({}, {} USING PARAMETERS index='{}', overwrite={} , max_mem_mb={}, skip_nonindexable_polygons={}) OVER() FROM {}"
    query = query.format(gid, g, index, overwrite, max_mem_mb, skip_nonindexable_polygons, vdf.__genSQL__())
    return to_tablesample(query, vdf._VERTICAPY_VARIABLES_["cursor"])
