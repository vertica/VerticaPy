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
from itertools import chain
import pytest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as py_TfidfVectorizer
import verticapy as vp
from verticapy.utilities import drop
from verticapy.machine_learning.vertica.feature_extraction.text import (
    TfidfVectorizer as vpy_TfidfVectorizer,
)
from verticapy.tests_new.machine_learning.vertica import abs_tolerance_map


class TestTFIDF:
    """
    test class for base TFIDF
    """
    documents = [
        "Natural language processing is a field of study in artificial intelligence.",
        # "TF-IDF stands for Term Frequency-Inverse Document Frequency.",
        "Machine learning algorithms can be applied to text data for classification.",
        "The 20 Newsgroups dataset is a collection of text documents used for text classification.",
        "Clustering is a technique used to group similar documents together.",
        "Python is a popular programming language for natural language processing tasks.",
        "TF-IDF is a technique widely used in information retrieval.",
        "An algorithm is a set of instructions designed to perform a specific task.",
        "Data preprocessing is an important step in preparing data for machine learning.",
    ]

    def test_transform(self):
        """
        test function for fit TFIDF
        """

        data = vp.vDataFrame(
            {
                "id": (list(range(1, len(self.documents) + 1))),
                "values": self.documents,
            }
        )
        drop("test_idf")
        vpy_model = vpy_TfidfVectorizer(name="test_idf")
        vpy_model.fit(
            input_relation=data,
            index="id",
            x="values",
        )
        vdf_merge = vpy_model.transform(vdf=data, index="id", x="values", pivot=False)
        vdf_merge_pandas = vdf_merge.to_pandas()

        # python
        py_model = py_TfidfVectorizer()
        py_model.fit(self.documents)
        py_term_vectors = py_model.transform(self.documents)
        array_len = len(py_term_vectors.toarray())
        merge_pdf = pd.DataFrame()

        for i in range(array_len):
            chain_array = py_term_vectors[i].T.toarray().tolist()
            pdf = pd.DataFrame(
                {
                    "row_id": len(chain_array) * [i + 1],
                    "word": py_model.get_feature_names_out(),
                    "tfidf_py": chain(*chain_array),
                }
            )
            nonzero_pdf = pdf[~pdf["tfidf_py"].isin([0])]
            merge_pdf = pd.concat([merge_pdf, nonzero_pdf])

        # inner join as some of the stop words are ignored in sklearn(by default)
        merge_vdf_pdf = vdf_merge_pandas.merge(
            merge_pdf, how="inner", on=["row_id", "word"]
        )

        def compare(x):
            print(x)
            assert x["tfidf"] == pytest.approx(
                x["tfidf_py"], abs=abs_tolerance_map["TFIDF"]
            )

        merge_vdf_pdf.apply(compare, axis=1)

    def test_get_attributes(self):
        pass
