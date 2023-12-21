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
import json
import logging
import os
from typing import List

from .file_metadata import FileMetadata


class CollectionMetadata:
    def __init__(self):
        # List of FileMetadata
        self.files = []
        self.date = datetime.datetime.now()
        self.version = "1.0"
        self.logger = logging.getLogger(type(self).__name__)

    def add_file(self, file_info: dict) -> None:
        self.files.append(file_info)

    def get_all_files(self) -> List[FileMetadata]:
        return self.files

    def to_json(self, output_dir) -> None:
        if not os.path.exists(output_dir):
            raise ValueError(f"Directory {output_dir} does not exist")

        fname = os.path.join(output_dir, "metadata.json")
        with open(fname, "w") as md_file:
            self._write_json(md_file)

    def _write_json(self, output_file):
        serialized = {
            "collection_metadata_version": self.version,
            "collection_time": str(self.date),
            "files": [x.to_dict() for x in self.files],
        }
        json.dump(serialized, output_file)
