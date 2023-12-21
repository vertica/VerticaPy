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

from .profile_step_base import ProfileStepBase

from verticapy.core.vdataframe import vDataFrame
from verticapy._utils._sql._vertica_version import vertica_version
from .collection_metadata import CollectionMetadata
from .file_metadata import FileMetadata
from .step_info import StepId


class Step00VerticaVersion(ProfileStepBase):
    def __init__(self):
        self.transaction_id = None
        self.statement_id = None
        self.output_dir = None
        self.version_vdf = None
        self.step_name = type(self).__name__
        self.step_id = StepId.VERTICA_VERSION.value
        self.step_version = "1.0"
        self.logger = logging.getLogger(self.step_name)

    def setup_collect(
        self, transaction_id: int, statement_id: int, output_dir: str
    ) -> None:
        # setup_collect is effectively a no-op.
        # Step00VerticaVersion is an exceptional step
        # where the output doesn't actually require the statement_id or
        # the transaction_id.
        self.transaction_id = transaction_id
        self.statement_id = statement_id
        self.output_dir = output_dir

    def collect(self) -> None:
        version_tuple = vertica_version()
        # vDataFrame can only be created with two-dimensional objects
        version_table = [version_tuple]
        self.version_vdf = vDataFrame(version_table)

    def store(self, metadata: CollectionMetadata) -> None:
        if self.version_vdf is None:
            raise ValueError("Step00 store expects collect to be called first")

        pandas = self.version_vdf.to_pandas()
        output_path = os.path.join(self.output_dir, "step00_vertica_version.parquet")
        pandas.to_parquet(path=output_path, compression="gzip")

        self.logger.info(
            "Wrote output to %s parquet file",
        )
        self._update_metadata(output_path, metadata)

    def _update_metadata(self, output_path: str, metadata: CollectionMetadata) -> None:
        basename = os.path.basename(output_path)
        metadata.add_file(self._get_file_metadata(basename))

    def _get_file_metadata(self, file_name: str) -> FileMetadata:
        return FileMetadata(
            file_name=file_name,
            step_name=self.step_name,
            step_id=self.step_id,
            step_version=self.step_version,
            creation_func_name="store",
            load_func_name="load",
        )

    def setup_load(self, collection_metadata: dict) -> None:
        pass

    def load(self) -> None:
        pass

    def analyze(self) -> None:
        pass
