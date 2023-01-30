# (c) Copyright [2018-2023] Micro Focus or one of its affiliates.
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
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
# Errors & Exceptions
#
# ---#
class ConnectionError(Exception):
    pass


# ---#
class ConversionError(Exception):
    pass


# ---#
class EnvironmentError(Exception):
    pass


# ---#
class EmptyParameter(Exception):
    pass


# ---#
class ExtensionError(Exception):
    pass


# ---#
class FunctionError(Exception):
    pass


# ---#
class MissingColumn(Exception):
    pass


# ---#
class MissingRelation(Exception):
    pass


# ---#
class MissingSchema(Exception):
    pass


# ---#
class ModelError(Exception):
    pass


# ---#
class ParameterError(Exception):
    pass


# ---#
class ParsingError(Exception):
    pass


# ---#
class QueryError(Exception):
    pass


# ---#
class VersionError(Exception):
    pass


# ---#
def raise_error_if_not_in(variable_name, variable, options):
    # Raises an error if the input variable does not belong to the input list.
    from verticapy.toolbox import levenshtein

    if variable not in options:
        min_distance, min_distance_op = 1000, ""
        for op in options:
            if str(variable).lower() == str(op).lower():
                error_message = f"Parameter '{variable_name}' is not correctly formatted. The correct option is {op}"
                raise ParameterError(error_message)
            else:
                ldistance = levenshtein(variable, op)
                if ldistance < min_distance:
                    min_distance, min_distance_op = ldistance, op
        error_message = "Parameter '{0}' must be in [{1}], found '{2}'.".format(
            variable_name, "|".join(options), variable
        )
        if min_distance < 6:
            error_message += f"\nDid you mean '{min_distance_op}'?"
        raise ParameterError(error_message)
