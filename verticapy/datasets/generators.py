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
import datetime

from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._vertica_version import check_minimum_version


from verticapy.core.vdataframe.base import vDataFrame


@check_minimum_version
@save_verticapy_logs
def gen_dataset(features_ranges: dict, nrows: int = 1000) -> vDataFrame:
    """
    Generates a dataset using the input parameters.

    Parameters
    ----------
    features_ranges: dict
        Dictionary including the features types and ranges.

        **For str** :
                        The  subdictionary must  include
                        two keys: 'type' must be set  to
                        'str'  and 'value' must  include
                        the feature categories.
        **For int** :
                        The subdictionary must include
                        two keys: 'type'  must be set to
                        'int' and 'range'  must  include
                        two integers  that represent the
                        lower and the upper bounds.
        **For float** :
                        The subdictionary must
                        include two keys: 'type' must be
                        set to'float' and 'range' must
                        include two floats that represent
                        the lower and the upper bounds.
        **For date** :
                        The subdictionary must include
                        two keys: 'type'  must be set to
                        'date' and 'range'  must include
                        the start date and the number of
                        days after.
        **For datetime** :
                        The  subdictionary must
                        include two keys: 'type' must be
                        set to 'date' and 'range'  must
                        include the start date and the
                        number of days after.
    nrows: int, optional
        The maximum number of rows in the dataset.

    Returns
    -------
    vDataFrame
        Generated dataset.

    Examples
    ---------
    .. code-block:: python

        from verticapy.datasets import gen_dataset
        import datetime

        gen_dataset(features_ranges = {"name": {"type": str, "values": ["Badr", "Badr", "Raghu", "Waqas",]},
                                       "age": {"type": int, "range": [20, 40]},
                                       "distance": {"type": float, "range": [1000, 4000]},
                                       "date": {"type": datetime.date, "range": ["1993-11-03", 365]},
                                       "datetime": {"type": datetime.datetime, "range": ["1993-11-03", 365]},},)

    .. ipython:: python
        :suppress:

        from verticapy.datasets import gen_dataset
        import datetime
        import verticapy as vp
        html_file = open("SPHINX_DIRECTORY/figures/datasets_generators_gen_dataset.html", "w")
        html_file.write(gen_dataset(features_ranges = {"name": {"type": str, "values": ["Badr", "Badr", "Raghu", "Waqas",]},
                                    "age": {"type": int, "range": [20, 40]},
                                    "distance": {"type": float, "range": [1000, 4000]},
                                    "date": {"type": datetime.date, "range": ["1993-11-03", 365]},
                                    "datetime": {"type": datetime.datetime, "range": ["1993-11-03", 365]},},)._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_generators_gen_dataset.html

    """
    sql = []

    for param in features_ranges:
        if features_ranges[param]["type"] == str:
            val = features_ranges[param]["values"]
            if isinstance(val, str):
                sql += [f"'{val}' AS \"{param}\""]
            else:
                n = len(val)
                val = ", ".join(["'" + str(v) + "'" for v in val])
                sql += [f'(ARRAY[{val}])[RANDOMINT({n})] AS "{param}"']

        elif features_ranges[param]["type"] == float:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            sql += [
                f"""({lower} + RANDOM() 
                  * ({upper} - {lower}))::FLOAT AS "{param}" """
            ]

        elif features_ranges[param]["type"] == int:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            sql += [
                f"""({lower} + RANDOM() 
                      * ({upper} - {lower}))::INT AS "{param}" """
            ]

        elif features_ranges[param]["type"] == datetime.date:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            sql += [
                f"""('{start_date}'::DATE 
                   + RANDOMINT({number_of_days})) AS "{param}" """
            ]

        elif features_ranges[param]["type"] == datetime.datetime:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            sql += [
                f"""('{start_date}'::TIMESTAMP 
                   + {number_of_days} * RANDOM()) AS "{param}" """
            ]

        elif features_ranges[param]["type"] == bool:
            sql += [f'RANDOMINT(2)::BOOL AS "{param}"']

        else:
            ptype = features_ranges[param]["type"]
            raise ValueError(f"Parameter {param}: Type {ptype} is not supported.")

    query = f"""
        SELECT 
            {', '.join(sql)} 
        FROM 
            (SELECT 
                tm 
            FROM 
                (SELECT '03-11-1993'::TIMESTAMP + INTERVAL '1 second' AS t 
                 UNION ALL 
                 SELECT '03-11-1993'::TIMESTAMP + INTERVAL '{nrows} seconds' AS t) x 
                TIMESERIES tm AS '1 second' OVER(ORDER BY t)) VERTICAPY_SUBTABLE"""

    return vDataFrame(query)


@save_verticapy_logs
def gen_meshgrid(features_ranges: dict) -> vDataFrame:
    """
    Generates a dataset using regular steps.

    Parameters
    ----------
    features_ranges: dict
        Dictionary including the features types and ranges.

        **For str** :
                        The  subdictionary must  include
                        two keys: 'type' must be set  to
                        'str'  and 'value' must  include
                        the feature categories.
        **For int** :
                        The subdictionary must include
                        two keys: 'type'  must be set to
                        'int' and 'range'  must  include
                        two integers  that represent the
                        lower and the upper bounds.
        **For float** :
                        The subdictionary must
                        include two keys:  'type' must be
                        set to 'float' and 'range' must
                        include two floats that represent
                        the lower and the upper bounds.
        **For date** :
                        The subdictionary must
                        include two keys: 'type' must be
                        set to 'date' and 'range'  must
                        include the start date and the
                        number of days after.
        **For datetime** :
                        The  subdictionary must
                        include two keys: 'type' must be
                        set to 'date' and 'range'  must
                        include the start date and the
                        number of days after.

        Numerical and date-like features must have an extra
        key in the  dictionary named 'nbins', which
        corresponds to the number of bins used to compute
        the different categories.

    Returns
    -------
    vDataFrame
        Generated dataset.

    Examples
    ---------
    .. code-block:: python

        from verticapy.datasets import gen_meshgrid
        import datetime

        gen_meshgrid(features_ranges = {"name": {"type": str, "values": ["Badr", "Badr", "Raghu", "Waqas",]},
                                        "age": {"type": int, "range": [20, 40]},
                                        "distance": {"type": float, "range": [1000, 4000]},
                                        "date": {"type": datetime.date, "range": ["1993-11-03", 365]},
                                        "datetime": {"type": datetime.datetime, "range": ["1993-11-03", 365]},},)

    .. ipython:: python
        :suppress:

        from verticapy.datasets import gen_meshgrid
        import datetime
        import verticapy as vp
        html_file = open("SPHINX_DIRECTORY/figures/datasets_generators_gen_meshgrid.html", "w")
        html_file.write(gen_meshgrid(features_ranges = {"name": {"type": str, "values": ["Badr", "Badr", "Raghu", "Waqas",]},
                                "age": {"type": int, "range": [20, 40], "nbins": 3,},
                                "distance": {"type": float, "range": [1000, 4000], "nbins": 3,},
                                "date": {"type": datetime.date, "range": ["1993-11-03", 365], "nbins": 2,},
                                "datetime": {"type": datetime.datetime, "range": ["1993-11-03", 365], "nbins": 2,},},)._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_generators_gen_meshgrid.html

    """
    sql = []

    for idx, param in enumerate(features_ranges):
        nbins = 100
        if "nbins" in features_ranges[param]:
            nbins = features_ranges[param]["nbins"]
        ts_table = f"""
            (SELECT 
                DAY(tm - '03-11-1993'::TIMESTAMP) AS tm 
             FROM 
                (SELECT '03-11-1993'::TIMESTAMP AS t 
                 UNION ALL 
                 SELECT '03-11-1993'::TIMESTAMP 
                        + INTERVAL '{nbins} days' AS t) x 
            TIMESERIES tm AS '1 day' OVER(ORDER BY t)) y"""

        if features_ranges[param]["type"] == str:
            val = features_ranges[param]["values"]
            if isinstance(val, str):
                val = [val]
            val = " UNION ALL ".join([f"""(SELECT '{v}' AS "{param}")""" for v in val])
            sql += [f"({val}) x{idx}"]

        elif features_ranges[param]["type"] == float:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            h = (upper - lower) / nbins
            sql += [
                f"""
                (SELECT 
                    ({lower} + {h} * tm)::FLOAT AS "{param}" 
                 FROM {ts_table}) x{idx}"""
            ]

        elif features_ranges[param]["type"] == int:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            h = (upper - lower) / nbins
            sql += [
                f"""
                (SELECT 
                    ({lower} + {h} * tm)::INT AS "{param}" 
                 FROM {ts_table}) x{idx}"""
            ]

        elif features_ranges[param]["type"] == datetime.date:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            h = number_of_days / nbins
            sql += [
                f"""
                (SELECT 
                    ('{start_date}'::DATE + {h} * tm)::DATE AS "{param}" 
                 FROM {ts_table}) x{idx}"""
            ]

        elif features_ranges[param]["type"] == datetime.datetime:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            h = number_of_days / nbins
            sql += [
                f"""
                    (SELECT 
                        ('{start_date}'::DATE + {h} * tm)::TIMESTAMP 
                            AS "{param}" 
                     FROM {ts_table}) x{idx}"""
            ]

        elif features_ranges[param]["type"] == bool:
            sql += [
                f"""
                ((SELECT False AS "{param}") 
                 UNION ALL
                (SELECT True AS "{param}")) x{idx}"""
            ]

        else:
            ptype = features_ranges[param]["type"]
            raise ValueError(f"Parameter {param}: Type {ptype} is not supported.")

    query = f"SELECT * FROM {' CROSS JOIN '.join(sql)}"

    return vDataFrame(query)
