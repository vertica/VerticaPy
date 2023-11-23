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


class ConversionError(Exception):
    ...


class EnvironmentError(Exception):
    ...


class EmptyParameter(Exception):
    ...


class ExtensionError(Exception):
    ...


class FunctionError(Exception):
    ...


class MissingColumn(Exception):
    ...


class MissingRelation(Exception):
    ...


class MissingSchema(Exception):
    ...


class ModelError(Exception):
    ...


class OptionError(Exception):
    ...


class ParsingError(Exception):
    ...


class QueryError(Exception):
    ...


class VersionError(Exception):
    ...
