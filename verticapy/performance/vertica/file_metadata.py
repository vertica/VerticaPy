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


class FileMetadata:
    def __init__(
        self,
        file_name: str,
        step_name: str,
        step_id: int,
        step_version: str,
        creation_func_name: str,
        load_func_name: str,
    ):
        self.file_name = file_name
        self.step_version = step_version
        self.step_id = step_id
        self.step_name = step_name
        self.creation_func_name = creation_func_name
        self.load_func_name = load_func_name

    def to_dict(self):
        return {
            "file_name": self.file_name,
            "step_name": self.step_name,
            "step_id": self.step_id,
            "step_version": self.step_version,
            "creation_func_name": self.creation_func_name,
            "load_func_name": self.load_func_name,
        }
