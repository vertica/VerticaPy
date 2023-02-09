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
from verticapy.io.parsers.csv import read_csv, pcsv
from verticapy.io.parsers.json import read_json, pjson
from verticapy.io.parsers.avro import read_avro
from verticapy.io.parsers.shp import read_shp
from verticapy.io.parsers.all import read_file
from verticapy.io.parsers.pandas import pandas_to_vertica
from verticapy.io.flex import compute_flextable_keys, compute_vmap_keys, isflextable, isvmap
