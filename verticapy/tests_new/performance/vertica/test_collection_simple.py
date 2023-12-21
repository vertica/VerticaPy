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
import logging
import os
import pandas
import zipfile

from verticapy.performance.vertica import (
    Step00VerticaVersion,
    CollectionMetadata,
    ProfileCollector,
)

_module_logger = None


def get_logger():
    global _module_logger
    _module_logger = (
        _module_logger
        if _module_logger is not None
        else logging.getLogger("TestCollectionSimple")
    )
    return _module_logger


class TestCollectionSimple:
    def test_collect_version_info(self, tmp_path):
        logger = get_logger()
        logger.info("Testing vertica version collection")
        md = CollectionMetadata()
        step = Step00VerticaVersion()
        step.setup_collect(transaction_id=123, statement_id=4, output_dir=str(tmp_path))

        step.collect()
        step.store(md)

        all_files = md.get_all_files()
        assert len(all_files) == 1

        file_meta = all_files[0]
        file_info = file_meta.to_dict()
        logging.info(f"FileMetadata serialized is: {file_info}")
        assert "file_name" in file_info
        assert "step_version" in file_info
        assert file_info["step_name"] == "Step00VerticaVersion"

        output_file = os.path.join(tmp_path, file_info["file_name"])
        assert os.path.exists(output_file)

        df = pandas.read_parquet(output_file)
        assert df.shape[0] == 1
        assert df.shape[1] == 3 or df.shape[1] == 4
        row = df.iloc(0)
        print(f"Here is a row: {row[0]['col0']}, {row[0]['col1']}, {row[0]['col2']}")
        assert row[0]["col0"] >= 23
        assert row[0]["col1"] >= 0
        assert row[0]["col2"] >= 0

    def test_collect_all(self, tmp_path):
        p = ProfileCollector()
        p.collect_all(transaction_id=123, statement_id=4, output_dir=str(tmp_path))
        expected_output = tmp_path / "profile.zip"
        assert os.path.exists(expected_output)
        with zipfile.ZipFile(expected_output, "r") as zfile:
            self._check_zipfile_info(zfile)

    def _check_zipfile_info(self, zfile: zipfile.ZipFile) -> None:
        foundMetadata = False
        foundStep00 = False
        all_file_names = []
        for i, zinfo in enumerate(zfile.infolist()):
            print(f"Item {i} is {zinfo}")
            all_file_names.append(zinfo.filename)
            if zinfo.filename.startswith("step00_"):
                foundStep00 = True
            if zinfo.filename == "metadata.json":
                foundMetadata = True
        assert foundMetadata, f"Did not find metadata.json in {all_file_names}"
        assert foundStep00, f"Did not find step00_* in {all_file_names}"
