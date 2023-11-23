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

from __future__ import print_function, division, absolute_import

import logging
from .os_utils import ensure_dir_exists


class VerticaLogging(object):
    @classmethod
    def setup_logging(cls, logger_name, logfile, log_level=logging.INFO, context=""):
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        if logfile:
            formatter = logging.Formatter(
                fmt=(
                    "%(asctime)s.%(msecs)03d [%(module)s] "
                    "{}/%(process)d:0x%(thread)x <%(levelname)s> "
                    "%(message)s".format(context)
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ensure_dir_exists(logfile)
            file_handler = logging.FileHandler(logfile, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
