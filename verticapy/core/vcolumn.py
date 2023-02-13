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
from verticapy.core.str_sql import str_sql

from verticapy.core.vdataframe.aggregate import vDCAGG
from verticapy.core.vdataframe.plot import vDCPLOT
from verticapy.core.vdataframe.math import vDCMATH
from verticapy.core.vdataframe.typing import vDCTYPING
from verticapy.core.vdataframe.filter import vDCFILTER
from verticapy.core.vdataframe.preprocessing import vDCPREP
from verticapy.core.vdataframe.read import vDCREAD
from verticapy.core.vdataframe.sys import vDCSYS
from verticapy.core.vdataframe.text import vDCTEXT
from verticapy.core.vdataframe.corr import vDCCORR

##
#
#   __   ___  ______     ______     __         __  __     __    __     __   __
#  /\ \ /  / /\  ___\   /\  __ \   /\ \       /\ \/\ \   /\ "-./  \   /\ "-.\ \
#  \ \ \' /  \ \ \____  \ \ \/\ \  \ \ \____  \ \ \_\ \  \ \ \-./\ \  \ \ \-.  \
#   \ \__/    \ \_____\  \ \_____\  \ \_____\  \ \_____\  \ \_\ \ \_\  \ \_\\"\_\
#    \/_/      \/_____/   \/_____/   \/_____/   \/_____/   \/_/  \/_/   \/_/ \/_/
#
##


class vColumn(
    str_sql,
    vDCAGG,
    vDCPLOT,
    vDCMATH,
    vDCTYPING,
    vDCFILTER,
    vDCPREP,
    vDCREAD,
    vDCSYS,
    vDCTEXT,
    vDCCORR,
):
    """
Python object which that stores all user transformations. If the vDataFrame
represents the entire relation, a vColumn can be seen as one column of that
relation. vColumns simplify several processes with its abstractions.

Parameters
----------
alias: str
	vColumn alias.
transformations: list, optional
	List of the different transformations. Each transformation must be similar
	to the following: (function, type, category)  
parent: vDataFrame, optional
	Parent of the vColumn. One vDataFrame can have multiple children vColumns 
	whereas one vColumn can only have one parent.
catalog: dict, optional
	Catalog where each key corresponds to an aggregation. vColumns will memorize
	the already computed aggregations to gain in performance. The catalog will
	be updated when the parent vDataFrame is modified.

Attributes
----------
	alias, str           : vColumn alias.
	catalog, dict        : Catalog of pre-computed aggregations.
	parent, vDataFrame   : Parent of the vColumn.
	transformations, str : List of the different transformations.
	"""

    #
    # Special Methods
    #

    def __init__(
        self, alias: str, transformations: list = [], parent=None, catalog: dict = {},
    ):
        self.parent, self.alias, self.transformations = (
            parent,
            alias,
            [elem for elem in transformations],
        )
        self.catalog = {
            "cov": {},
            "pearson": {},
            "spearman": {},
            "spearmand": {},
            "kendall": {},
            "cramer": {},
            "biserial": {},
            "regr_avgx": {},
            "regr_avgy": {},
            "regr_count": {},
            "regr_intercept": {},
            "regr_r2": {},
            "regr_slope": {},
            "regr_sxx": {},
            "regr_sxy": {},
            "regr_syy": {},
        }
        for elem in catalog:
            self.catalog[elem] = catalog[elem]
        self.init_transf = self.transformations[0][0]
        if self.init_transf == "___VERTICAPY_UNDEFINED___":
            self.init_transf = self.alias
        self.init = True
