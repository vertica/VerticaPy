"""
Copyright  (c)  2018-2023 Open Text  or  one  of its
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
from verticapy._typing import SQLRelation
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.base import VerticaModel


class Tfidf(VerticaModel):
    """
    [Beta Version]
    Create tfidf representation of documents.

    Parameters
    ----------
    name: str
        Name of the model.
    overwrite_model: bool, optional
        If set to True, training a model with the
        same name as an existing model overwrites
        the existing model.
    lowercase: bool, optional
        Converts  all  the elements to lowercase
        before processing.
    """

    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        lowercase: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "lowercase": lowercase,
        }

    def fit(self, input_relation: SQLRelation, index: str, column: str) -> None:
        """
        Applies basic pre-processing. Creates table
        with fitted vocabulary and idf values.

        Parameters
        ----------
        input_relation: SQLRelation
            Training relation.
        index: str
            Column name of the document id.
        column: str
            Column name which contains the text.

        Returns
        ----------
            None
        """

        if isinstance(input_relation, vDataFrame):
            vdf = input_relation.copy()
        else:
            vdf = vDataFrame(input_relation)

        if not self.parameters["lowercase"]:
            text = {column}
        else:
            text = f"LOWER({column})"

        self.idf_ = self.model_name

        if self.parameters["overwrite_model"]:
            _executeSQL(f"DROP TABLE IF EXISTS {self.idf_}", print_time_sql=False)

        q_idf = f"""
            CREATE TABLE {self.idf_} AS
                WITH 
                    tdc AS (
                      SELECT 
                          count({index}) count_docs 
                      FROM {vdf}
                    ),
                    words_by_post AS (
                      SELECT
                          {index} as row_id
                          ,{text} as content
                          ,string_to_array(
                              REGEXP_REPLACE(
                                  TRIM(
                                      REGEXP_REPLACE(
                                          REGEXP_REPLACE(
                                              {text},'[^\w ]',''
                                          ),
                                          ' {2,}',' '
                                      )
                                  ),
                              '\s',',')
                          ) words 
                          ,COUNT(*) OVER() docs_n
                      FROM {vdf}
                    ),
                    exploded AS (
                      SELECT
                          EXPLODE(words, words, content, row_id ) OVER(PARTITION best) 
                      FROM words_by_post
                    )
                    SELECT
                        value AS word
                        , COUNT(DISTINCT row_id) AS word_doc_count
                        , count_docs
                        ,LN((1+count_docs)/(1+word_doc_count)+1) idf_log
                    FROM exploded
                    CROSS JOIN tdc
                    GROUP BY word,count_docs
                    ORDER BY word_doc_count desc"""

        _executeSQL(q_idf, print_time_sql=False)

    def transform(
        self, vdf: SQLRelation, index: str, column: str, pivot: bool = False
    ) -> vDataFrame:
        """
        Applies basic pre-processing. Creates table with
        vocabulary and idf values.

        Parameters
        ----------
        vdf: SQLRelation
            Object used to run  the prediction.  You can
            also  specify a  customized  relation,  but you
            must  enclose  it with an alias.  For  example,
            "(SELECT 1) x" is valid, whereas "(SELECT 1)"
            and "SELECT 1" are invalid.
        index: str
            Column name of the document id.
        column: str
            Column name which contains the text.
        pivot: str
            If True it will Pivot the final table, to have
            1 row per document and a sparse matrix.

        Returns
        ----------
            vDataFrame
        """

        if isinstance(vdf, vDataFrame):
            vdf = vdf.copy()
        else:
            vdf = vDataFrame(vdf)

        if not self.parameters["lowercase"]:
            text = {column}
        else:
            text = f"LOWER({column})"

        q_tfidf = f"""
            WITH 
                words_by_post AS (
                      SELECT
                          {index} as row_id
                          ,{text} as content
                          ,string_to_array(
                              REGEXP_REPLACE(
                                  TRIM(
                                      REGEXP_REPLACE(
                                          REGEXP_REPLACE(
                                              {text},'[^\w ]',''
                                          ),
                                          ' {2,}',' '
                                      )
                                  ),
                              '\s',',')
                          ) words 
                          ,COUNT(*) OVER() docs_n
                      FROM {vdf}
                ),
                exploded AS (
                      SELECT
                          EXPLODE(words, words, content, row_id ) OVER(PARTITION best) 
                      FROM words_by_post
                ),
                tf AS (
                    SELECT
                        row_id
                        ,value as word
                        ,count(*) as  tf
                    FROM exploded
                    GROUP BY row_id,word,words
                )
                SELECT 
                    tf.row_id
                    ,{self.idf_}.word
                    ,tf*idf_log/sqrt(sum((tf*idf_log)^2) OVER(partition by tf.row_id)) tf_idf
                FROM tf
                INNER JOIN {self.idf_} on tf.word = {self.idf_}.word
                ORDER BY tf.row_id"""

        result = vDataFrame(q_tfidf)
        if not pivot:
            return result
        else:
            return result.pivot(
                index="row_id", columns="word", values="tf_idf", prefix=""
            )
