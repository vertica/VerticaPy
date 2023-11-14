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

from verticapy._typing import NoneType, SQLRelation
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.vdataframe.base import vDataFrame
from verticapy.sql.drop import drop

from verticapy.machine_learning.vertica.base import VerticaModel


class TfidfVectorizer(VerticaModel):
    """
    Create tfidf representation of documents.

    The formula that is used to compute the tf-idf for a
    term t of a document d in a document set is

    .. math::

        tf-idf(t, d) = tf(t, d) * idf(t),

    and if ``smooth_idf = False``, the idf is computed as

    .. math::

        idf(t) = log [ n / df(t) ] + 1,

    where n is the total number of documents in the document
    set and df(t) is the document frequency of t; the document
    frequency is the number of documents in the document set
    that contain the term t. The effect of adding "1" to the
    idf in the equation above is that terms with zero idf, i.e.,
    terms that occur in all documents in a training set, will
    not be entirely ignored.

    If ``smooth_idf=True`` (the default), the constant "1" is
    added to the numerator and denominator of the idf as if an
    extra document was seen containing every term in the
    collection exactly once, which prevents zero divisions:

    .. math::
        idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.

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
        The tfidf values of each document will have unit norm,
        either:

        - l2:
            Sum of squares of vector elements is 1.

        - l1:
            Sum of absolute values of vector elements is 1.

        - None:
            No normalization.

    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies,
        as if an extra document was seen containing every term in
        the collection exactly once. Prevents zero divisions.

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    idf_: str
        name of the IDF table or SQL code needed to generate it.

    .. note::

        All attributes can be accessed using the
        :py:mod:`verticapy.machine_learning.vertica.base.VerticaModel.get_attributes``
        method.

    Examples
    --------

    We import ``verticapy``:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to ``verticapy``, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, let's generate a dataset:

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "id": [1, 2, 3],
                "values": [
                    "this is a test",
                    "this is another test",
                    "this is different",
                ]
            }
        )

    First we initialize the object and fit the model, to learn the
    idf weigths.

    .. code-block:: python

        from verticapy.machine_learning.vertica.feature_extraction.text import TfidfVectorizer

        model = TfidfVectorizer(name = "test_idf")
        model.fit(
            input_relation = data,
            index = "id",
            x = "values",
        )

    We apply the transform function to obtain the idf representation.

    .. code-block:: python

        model.transform(
            vdf = data,
            index = "id",
            x = "values",
            pivot = True,
        )

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica.feature_extraction.text import TfidfVectorizer

        model = TfidfVectorizer(name = "test_idf", overwrite_model = True)
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

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _vertica_fit_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_transform_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_inverse_transform_sql(self) -> Literal[""]:
        return ""

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["TEXT"]:
        return "TEXT"

    @property
    def _model_type(self) -> Literal["TfidfVectorizer"]:
        return "TfidfVectorizer"

    @property
    def _attributes(self) -> list[str]:
        return [
            "idf_",
        ]

    # System & Special Methods.

    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        lowercase: bool = True,
        norm: Literal["l1", "l2", None] = "l2",
        smooth_idf: bool = True,
    ) -> None:
        self.create_table_ = not (isinstance(name, None))
        super().__init__(name, overwrite_model)
        self.parameters = {
            "lowercase": lowercase,
            "norm": norm,
            "smooth_idf": smooth_idf,
        }

    # Model Fitting Method.

    @staticmethod
    def _wbd(vdf: SQLRelation, index: str, text: str):
        query = f"""
            SELECT
                {index} AS row_id,
                {text} AS content,
                STRING_TO_ARRAY(
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
                ) AS words 
            FROM {vdf}"""
        return query

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
        """

        if isinstance(input_relation, vDataFrame):
            vdf = input_relation.copy()
        else:
            vdf = vDataFrame(input_relation)

        if not self.parameters["lowercase"]:
            text = str(x)
        else:
            text = f"LOWER({x})"

        if self.parameters["smooth_idf"]:
            idf_expr = "LN((1 + count_docs) / (1 + word_doc_count)) + 1"
        else:
            idf_expr = "LN((count_docs) / (word_doc_count)) + 1"

        self.idf_ = self.model_name

        if self.overwrite_model:
            drop(self.idf_)

        q_tdc = f"""
            SELECT
                count({index}) count_docs 
            FROM {vdf}"""

        q_wbd = self._wbd(vdf=vdf, index=index, text=text)

        q_e = """
            SELECT
                EXPLODE(words, 
                        words, 
                        content, 
                        row_id) OVER(PARTITION best) 
              FROM words_by_doc"""

        create_table = ""
        if self.create_table_:
            create_table = f"CREATE TABLE {self.idf_} AS"

        q_idf = f"""
            {create_table}
            WITH 
            tdc AS ({q_tdc}),
            words_by_doc AS ({q_wbd}),
            exploded AS ({q_e})
            SELECT
                value AS word,
                COUNT(DISTINCT row_id) AS word_doc_count,
                count_docs,
                {idf_expr} AS idf_log
            FROM exploded
            CROSS JOIN tdc
            GROUP BY word, count_docs
            ORDER BY word_doc_count desc"""

        if self.create_table_:
            self.idf_ = f"({q_idf}) VERTICAPY_SUBTABLE"
        else:
            _executeSQL(q_idf, print_time_sql=False)

    # Prediction / Transformation Methods.

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
            If set to True, the final table will be pivoted
            to have one row per document, resulting in a
            sparse matrix. It's important to note that when
            dealing with a large dictionary, the pivot
            operation can be resource-intensive. In such
            cases, it might be more efficient to set ``pivot``
            to False, filter the output as needed, and then
            manually perform the pivot operation.

        Returns
        -------
        vDataFrame
            object result of the model transformation.
        """

        if isinstance(vdf, vDataFrame):
            vdf = vdf.copy()
        else:
            vdf = vDataFrame(vdf)

        if not self.parameters["lowercase"]:
            text = str(x)
        else:
            text = f"LOWER({x})"

        if isinstance(self.parameters["norm"], NoneType):
            t_norm = "1"

        if self.parameters["norm"] == "l2":
            t_norm = "SQRT(POWER(SUM((tf * idf_log), 2)) OVER (PARTITION BY tf.row_id))"

        if self.parameters["norm"] == "l1":
            t_norm = "SUM(ABS(tf * idf_log)) OVER (PARTITION BY tf.row_id)"

        q_wbd = self._wbd(vdf=vdf, index=index, text=text)

        q_e = """
            SELECT
                EXPLODE(words, 
                        words, 
                        content, 
                        row_id) OVER (PARTITION BEST) 
            FROM words_by_doc"""

        q_tf = """
            SELECT
                row_id,
                value as word,
                COUNT(*) as tf
            FROM exploded
            GROUP BY row_id, word, words"""

        q_tfidf = f"""
            WITH
                words_by_doc AS ({q_wbd}),
                exploded AS ({q_e}),
                tf AS ({q_tf})
                SELECT 
                    tf.row_id,
                    {self.idf_}.word,
                    tf * idf_log / {t_norm} AS tf_idf
                FROM tf
                INNER JOIN {self.idf_} 
                ON tf.word = {self.idf_}.word
                ORDER BY tf.row_id"""

        result = vDataFrame(q_tfidf)
        if not pivot:
            return result
        return result.pivot(index="row_id", columns="word", values="tf_idf", prefix="")
