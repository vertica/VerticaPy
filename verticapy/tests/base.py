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

import os
import sys
import logging
import unittest
import inspect
import getpass
import vertica_python

from configparser import ConfigParser

from .utils.log import VerticaLogging


default_configs = {
    "log_dir": "vp_test_log",
    "log_level": logging.WARNING,
    "host": "localhost",
    "port": 5433,
    "user": getpass.getuser(),
    "password": "",
}


class VerticaPyTestBase(unittest.TestCase):
    """
    Base class for all tests
    """

    @classmethod
    def _load_test_config(cls, config_list):
        test_config = {}

        # load default configurations
        for key in config_list:
            if key != "database":
                test_config[key] = default_configs[key]

        # override with the configuration file
        confparser = ConfigParser()
        confparser.optionxform = str
        SECTION = "vp_test_config"  # section name in the configuration file
        # the configuration file is placed in the same directory as this file
        conf_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "verticaPy_test.conf"
        )
        confparser.read(conf_file)
        for k in config_list:
            option = "VP_TEST_" + k.upper()
            if confparser.has_option(SECTION, option):
                test_config[k] = confparser.get(SECTION, option)

        # override again with VP_TEST_* environment variables
        for k in config_list:
            env = "VP_TEST_" + k.upper()
            if env in os.environ:
                test_config[k] = os.environ[env]

        # data preprocessing
        # value is string when loaded from configuration file and environment variable
        if "port" in test_config:
            test_config["port"] = int(test_config["port"])
        if "database" in config_list and "user" in test_config:
            test_config.setdefault("database", test_config["user"])
        if "log_level" in test_config:
            levels = ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if isinstance(test_config["log_level"], str):
                if test_config["log_level"] not in levels:
                    raise ValueError(
                        "Invalid value for VP_TEST_LOG_LEVEL: '{}'".format(
                            test_config["log_level"]
                        )
                    )
                test_config["log_level"] = eval("logging." + test_config["log_level"])
        if "log_dir" in test_config:
            test_config["log_dir"] = os.path.join(
                test_config["log_dir"],
                "py{0}{1}".format(sys.version_info.major, sys.version_info.minor),
            )
        return test_config

    @classmethod
    def _setup_logger(cls, tag, log_dir, log_level):
        # Setup test logger
        # E.g. If the class is defined in tests/learn/test_dates.py
        #      and test cases run under python3.7, then
        #      the log would write to $VP_TEST_LOG_DIR/py37/learn/test_dates.log

        testfile = os.path.splitext(os.path.basename(inspect.getsourcefile(cls)))[0]
        logfile = os.path.join(log_dir, tag, testfile + ".log")
        VerticaLogging.setup_logging(cls.__name__, logfile, log_level, cls.__name__)
        cls.logger = logging.getLogger(cls.__name__)
        return logfile

    def setUp(self):
        self.setUpClass()
        self.logger.info(
            "\n\n" + "-" * 50 + "\n Begin " + self.__class__.__name__ + "\n" + "-" * 50
        )
        self._connection = vertica_python.connect(**self._conn_info)
        self.cursor = self._connection.cursor()

    def tearDown(self):
        self._connection.close()
        self.logger.info(
            "\n" + "-" * 10 + " End " + self.__class__.__name__ + " " + "-" * 10 + "\n"
        )

    @classmethod
    def setUpClass(cls):
        config_list = [
            "log_dir",
            "log_level",
            "host",
            "port",
            "user",
            "password",
            "database",
        ]
        cls.test_config = cls._load_test_config(config_list)

        # Test logger
        logfile = cls._setup_logger(
            "tests", cls.test_config["log_dir"], cls.test_config["log_level"]
        )

        # Connection info
        cls._conn_info = {
            "host": cls.test_config["host"],
            "port": cls.test_config["port"],
            "database": cls.test_config["database"],
            "user": cls.test_config["user"],
            "password": cls.test_config["password"],
            "log_level": cls.test_config["log_level"],
            "log_path": logfile,
        }

        cls.db_node_num = cls._get_node_num()
        cls.logger.info("Number of database node(s) = {}".format(cls.db_node_num))

    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    def _connect(cls):
        """Connects to vertica.
        :return: a connection to vertica.
        """
        return vertica_python.connect(**cls._conn_info)

    @classmethod
    def _get_node_num(cls):
        """Executes a query to get the number of nodes in the database
        :return: the number of database nodes
        """
        with cls._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM nodes WHERE node_state='UP'")
            return cur.fetchone()[0]

    # Some tests require server-side setup
    # In that case, tests that depend on that setup should be skipped to prevent false failures
    # Tests that depend on the server-setup should call these methods to express requirements
    def require_DB_nodes_at_least(self, min_node_num):
        if not isinstance(min_node_num, int):
            err_msg = "Node number '{0}' must be an instance of 'int'".format(
                min_node_num
            )
            raise TypeError(err_msg)
        if min_node_num <= 0:
            err_msg = "Node number {0} must be a positive integer".format(min_node_num)
            raise ValueError(err_msg)

        if self.db_node_num < min_node_num:
            msg = (
                "The test requires a database that has at least {0} node(s), "
                "but this database has only {1} available node(s)."
            ).format(min_node_num, self.db_node_num)
            self.skipTest(msg)

    @classmethod
    def _get_DB_version(cls):
        """Executes a query to get the version of the database
        :return: the version of database
        """
        with cls._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT version()")
            return cur.fetchone()[0]

    def require_DB_version_at_least(self, min_version):
        # TODO: implement
        pass

    # Common assertions
