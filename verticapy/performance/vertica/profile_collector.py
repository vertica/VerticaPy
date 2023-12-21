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
from typing import List
import zipfile

from .step00_vertica_version import Step00VerticaVersion
from .collection_metadata import CollectionMetadata
from .profile_step_base import ProfileStepBase


class ProfileCollector:
    def __init__(self):
        self.logger = logging.getLogger("ProfileCollector")

    def collect_all(
        self, transaction_id: int, statement_id: int, output_dir: str
    ) -> None:
        """Returns None

        Calls the following functions for all steps in the collection
          - setup_collect()
          - collect()
          - store()

        """
        steps = [Step00VerticaVersion()]
        meta = CollectionMetadata()

        self._run_all_steps(transaction_id, statement_id, steps, meta, output_dir)
        self._write_metadata(meta, output_dir)
        self._create_bundle(meta, output_dir)

    def _run_all_steps(
        self,
        transaction_id: int,
        statement_id: int,
        steps: List[ProfileStepBase],
        meta: CollectionMetadata,
        output_dir: str,
    ) -> None:
        for step in steps:
            try:
                step.setup_collect(
                    transaction_id=transaction_id,
                    statement_id=statement_id,
                    output_dir=output_dir,
                )
                step.collect()
                step.store(meta)
            except Exception as e:
                # Someday we can improve better exception handling
                # Like raising custom errors when the step cannot be completed
                self.logger.error(f"Caughht exception {e} running step {step}")
                raise

    def _write_metadata(self, meta: CollectionMetadata, output_dir: str) -> None:
        meta.to_json(output_dir=output_dir)

    def _create_bundle(self, meta: CollectionMetadata, output_dir: str):
        self.logger.info("Called _create_bundle")
        bundle_file_name = os.path.join(output_dir, "profile.zip")
        with zipfile.ZipFile(bundle_file_name, mode="w") as zfile:
            md_full_path = os.path.join(output_dir, "metadata.json")
            zfile.write(md_full_path, arcname="metadata.json")
            for f in meta.get_all_files():
                full_path = os.path.join(output_dir, f.file_name)
                zfile.write(full_path, arcname=f.file_name)
