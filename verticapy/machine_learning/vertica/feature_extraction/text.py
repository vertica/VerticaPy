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
from typing import Literal
from verticapy._typing import SQLRelation
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.vdataframe.base import vDataFrame
from verticapy.utilities import drop

from verticapy.machine_learning.vertica.base import VerticaModel


class Tfidf(VerticaModel):
    """
    Create tfidf representation of documents.

    The formula that is used to compute the tf-idf for a term t of a document d
    in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is
    computed as idf(t) = log [ n / df(t) ] + 1 (if ``smooth_idf=False``), where
    n is the total number of documents in the document set and df(t) is the
    document frequency of t; the document frequency is the number of documents
    in the document set that contain the term t. The effect of adding "1" to
    the idf in the equation above is that terms with zero idf, i.e., terms
    that occur in all documents in a training set, will not be entirely
    ignored.

    If ``smooth_idf=True`` (the default), the constant "1" is added to the
    numerator and denominator of the idf as if an extra document was seen
    containing every term in the collection exactly once, which prevents
    zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.

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
    norm: {'l1','l2'} or None, default='l2'
        The tfidf values of each document will have unit norm, either:
            'l2': Sum of squares of vector elements is 1.
            'l1': Sum of absolute values of vector elements is 1.
            None: No normalization.
    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    Examples
        --------
        For this example, let's generate a dataset:

        .. ipython:: python

            import verticapy as vp
            from verticapy.machine_learning.vertica.feature_extraction.text import Tfidf

            data = vp.vDataFrame(
                {
                    "id": [1, 2, 3],
                    "values": [
                                "this is a test",
                                "this is another test",
                                "this is different"
                              ]
                }
            )

        First we initialize the object and fit the model, to learn the idf weigths.

        .. code-block:: python
            model = Tfidf(name = "test_idf")
            model.fit(input_relation = data, index = "id", x = "values")

        We apply the transform function to obtain the idf representation.

        .. code-block:: python
            model.transform(vdf = data, index = "id", x = "values", pivot = True)

        .. ipython:: python
            :suppress:

            model = Tfidf(name = "test_idf")
            model.fit(input_relation = data, index = "id", x = "values")
            model.transform(vdf = data, index = "id", x = "values", pivot = True)

            result = model.transform(vdf = data, index = "id", x = "values", pivot = True)
            html_file = open("SPHINX_DIRECTORY/figures/machine_learning_feature_extraction_text_tfidf.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/machine_learning_feature_extraction_text_tfidf.html

        .. seealso::
            | :py:mod:`verticapy.vDataColumn.pivot` : pivot vDataFrame.
    """

    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        lowercase: bool = True,
        norm: Literal["l1", "l2", None] = "l2",
        smooth_idf: bool = True,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "lowercase": lowercase,
            "norm": norm,
            "smooth_idf": smooth_idf,
        }

    @staticmethod
    def _wbd(vdf: SQLRelation, index: str, text: str):
        _query = f"""SELECT
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
                      FROM {vdf}"""
        return _query

    def fit(self, input_relation: SQLRelation, index: str, x: str) -> None:
        """
        Applies basic pre-processing. Creates table
        with fitted vocabulary and idf values.

        Parameters
        ----------
        input_relation: SQLRelation
            Training relation.
        index: str
            Column name of the document id.
        x: str
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
            text = {x}
        else:
            text = f"LOWER({x})"

        if self.parameters["smooth_idf"]:
            idf_expr = "LN((1 + count_docs)/(1 + word_doc_count)) + 1"
        else:
            idf_expr = "LN((count_docs)/(word_doc_count)) + 1"

        self.idf_ = self.model_name

        if self.overwrite_model:
            drop(self.idf_)

        q_tdc = f"""SELECT
                          count({index}) count_docs 
                      FROM {vdf}"""

        q_wbd = self._wbd(vdf=vdf, index=index, text=text)

        q_e = """SELECT
                        EXPLODE(words, words, content, row_id ) OVER(PARTITION best) 
                  FROM words_by_doc"""

        q_idf = f"""CREATE TABLE {self.idf_} AS
                        WITH 
                            tdc AS ({q_tdc}),
                            words_by_doc AS ({q_wbd}),
                            exploded AS ({q_e})
                            SELECT
                                value AS word
                                , COUNT(DISTINCT row_id) AS word_doc_count
                                , count_docs
                                ,{idf_expr} idf_log
                            FROM exploded
                            CROSS JOIN tdc
                            GROUP BY word,count_docs
                            ORDER BY word_doc_count desc"""

        _executeSQL(q_idf, print_time_sql=False)

    def transform(
        self, vdf: SQLRelation, index: str, x: str, pivot: bool = False
    ) -> vDataFrame:
        """
        Transforms input data to tf-idf representation.

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
        x: str
            Column name which contains the text.
        pivot: str
            If True it will pivot the final table, to have
            1 row per document and a sparse matrix.
            When working with a big dictionary, pivot operation is
            resource intensive.
            Might be better to set pivot=False, filter the output
            and then manually pivot.

        Returns
        ----------
            vDataFrame
        """

        if isinstance(vdf, vDataFrame):
            vdf = vdf.copy()
        else:
            vdf = vDataFrame(vdf)

        if not self.parameters["lowercase"]:
            text = {x}
        else:
            text = f"LOWER({x})"

        if self.parameters["norm"] is None:
            t_norm = "1"

        if self.parameters["norm"] == "l2":
            t_norm = "sqrt(sum((tf*idf_log)^2) OVER(partition by tf.row_id))"

        if self.parameters["norm"] == "l1":
            t_norm = "sum(abs(tf*idf_log)) OVER(partition by tf.row_id)"

        q_wbd = self._wbd(vdf=vdf, index=index, text=text)

        q_e = """SELECT
                       EXPLODE(words, words, content, row_id ) OVER(PARTITION best) 
                  FROM words_by_doc"""

        q_tf = """SELECT
                        row_id
                        ,value as word
                        ,count(*) as tf
                    FROM exploded
                    GROUP BY row_id,word,words"""

        q_tfidf = f"""WITH
                          words_by_doc AS ({q_wbd}),
                          exploded AS ({q_e}),
                          tf AS ({q_tf})
                          SELECT 
                              tf.row_id
                              ,{self.idf_}.word
                              ,tf*idf_log/{t_norm} tf_idf
                          FROM tf
                          INNER JOIN {self.idf_} on tf.word = {self.idf_}.word
                          ORDER BY tf.row_id"""

        result = vDataFrame(q_tfidf)
        if not pivot:
            return result
        return result.pivot(index="row_id", columns="word", values="tf_idf", prefix="")
