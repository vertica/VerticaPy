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
import itertools
import numpy as np

def all_comb(X: list):
    all_configuration = []
    for r in range(len(X) + 1):
        combinations_object = itertools.combinations(X, r)
        combinations_list = list(combinations_object)
        if combinations_list[0]:
            all_configuration += combinations_list
    return all_configuration


def heuristic_length(i: int):
    GAMMA = 0.5772156649
    if i == 2:
        return 1
    elif i > 2:
        return 2 * (np.log(i - 1) + GAMMA) - 2 * (i - 1) / i
    else:
        return 0


def levenshtein(s: str, t: str):
    rows = len(s) + 1
    cols = len(t) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    for i in range(1, rows):
        dist[i][0] = i
    for i in range(1, cols):
        dist[0][i] = i
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(
                dist[row - 1][col] + 1,
                dist[row][col - 1] + 1,
                dist[row - 1][col - 1] + cost,
            )
    return dist[row][col]
