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

from vertica_python.errors import ConnectionError


class OAuthConfigurationError(ConnectionError):
    """
    Error class which covers errors pertaining to
    OAuth configuration setup.
    """


class OAuthEndpointDiscoveryError(ConnectionError):
    """
    Error class which covers errors pertaining to
    failure to get token url from discovery url.
    """


class OAuthTokenRefreshError(ConnectionError):
    """
    Error class which covers errors pertaining to
    failure to authenticate using Refresh Token.
    """
