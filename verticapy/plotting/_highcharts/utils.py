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

"""
High Charts - Utils.
"""


def sort_classes(categories: list) -> list:
    try:
        try:
            order = []
            for item in categories:
                order += [float(item.split(";")[0].split("[")[1])]
        except:
            order = []
            for item in all_subcategories:
                order += [float(item)]
        order = [x for _, x in sorted(zip(order, categories))]
    except:
        return categories
    return order


def data_to_columns(data: list, n: int) -> list:
    columns = [[]] * n
    for elem in data:
        for i in range(n):
            try:
                columns[i] = columns[i] + [float(elem[i])]
            except:
                columns[i] = columns[i] + [elem[i]]
    return columns
