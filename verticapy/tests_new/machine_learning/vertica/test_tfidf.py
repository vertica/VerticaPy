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
from sklearn.feature_extraction.text import CountVectorizer
import verticapy as vp
from verticapy.machine_learning.vertica.feature_extraction.text import (
    TfidfVectorizer as vpy_TfidfVectorizer,
)
from verticapy.tests_new.machine_learning.vertica import rel_abs_tol_map


class TestTFIDF:
    """
    test class for base TFIDF
    """

    documents = [
        "Natural language processing is a field of study in artificial intelligence.",
        "TFIDF stands for Term Frequency Inverse Document Frequency.",
        "Machine learning algorithms can be applied to text data for classification.",
        "The 20 Newsgroups dataset is a collection of text documents used for text classification.",
        "Clustering is a technique used to group similar documents together.",
        "Python is a popular programming language for natural language processing tasks.",
        "TFIDF is a technique widely used in information retrieval.",
        "An algorithm is a set of instructions designed to perform a specific task.",
        "Data preprocessing is an important step in preparing data for machine learning.",
    ]

    data = vp.vDataFrame(
        {
            "id": (list(range(1, len(documents) + 1))),
            "values": documents,
        }
    )

    def create_tfidf_model(
        self,
        name="test_tfidf",
        overwrite_model=False,
        lowercase=True,
        vocabulary=None,
        max_df=None,
        min_df=None,
        norm="l2",
        smooth_idf=True,
        compute_vocabulary=True,
    ):
        """
        function to create TFIDF
        """
        vp.drop(name)
        vpy_model = vpy_TfidfVectorizer(
            name=name,
            overwrite_model=overwrite_model,
            lowercase=lowercase,
            vocabulary=vocabulary,
            max_df=max_df,
            min_df=min_df,
            norm=norm,
            smooth_idf=smooth_idf,
            compute_vocabulary=compute_vocabulary,
        )
        vpy_model.fit(
            input_relation=self.data,
            index="id",
            x="values",
        )

        # python
        py_model = py_TfidfVectorizer(
            lowercase=lowercase,
            vocabulary=[vocabulary] if vocabulary else None,
            max_df=max_df if max_df else 1.0,
            min_df=min_df if min_df else 1,
            norm=norm,
        )
        py_model.fit(self.documents)

        return vpy_model, py_model

    @pytest.mark.parametrize(
        "overwrite_model, lowercase, vocabulary, max_df, min_df, norm, smooth_idf, compute_vocabulary",
        [
            (False, True, None, None, None, "l2", True, True),
            (False, False, None, None, None, "l2", True, True),
            # (False, True, "together", None, None, "l2", True, True),  # getting empty vdf. Need to check
            (False, True, None, 4, 1, "l2", True, True),
            (False, True, None, 4, 2, "l2", True, True),
            (False, True, None, None, None, "l1", True, True),
            (False, True, None, None, None, None, True, True),
            (False, True, None, None, None, "l2", False, True),
            (False, True, None, None, None, "l2", True, False),
        ],
    )
    def test_tfidf(
        self,
        schema_loader,
        overwrite_model,
        lowercase,
        vocabulary,
        max_df,
        min_df,
        norm,
        smooth_idf,
        compute_vocabulary,
    ):
        """
        test function for fit TFIDF
        """
        vpy_model, py_model = self.create_tfidf_model(
            name=f"{schema_loader}.test_idf",
            overwrite_model=overwrite_model,
            lowercase=lowercase,
            vocabulary=vocabulary,
            max_df=max_df,
            min_df=min_df,
            norm=norm,
            smooth_idf=smooth_idf,
            compute_vocabulary=compute_vocabulary,
        )

        vdf_merge = vpy_model.transform(
            vdf=self.data, index="id", x="values", pivot=False
        )
        vdf_merge_pandas = vdf_merge.to_pandas()

        # python
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

        # inner join as some of the alphabets (like a) are ignored in sklearn(by default)
        merge_vdf_pdf = vdf_merge_pandas.merge(
            merge_pdf, how="inner", on=["row_id", "word"]
        )

        pd.testing.assert_series_equal(
            merge_vdf_pdf["tfidf"],
            merge_vdf_pdf["tfidf_py"],
            check_names=False,
            atol=rel_abs_tol_map["TFIDF"],
        )

    def test_get_attributes(self, schema_loader):
        """
        test function for test_get_attributes
        """
        vpy_res = self.create_tfidf_model(name=f"{schema_loader}.test_idf")[
            0
        ].get_attributes()
        assert vpy_res == [
            "idf_",
            "tf_",
            "vocabulary_",
            "fixed_vocabulary_",
            "stop_words_",
            "n_document_",
        ]

    @pytest.mark.parametrize(
        "attribute_name, expected",
        [
            ("idf_", ["a"]),
            ("tf_", ["a"]),
            ("vocabulary_", ["a"]),
            ("fixed_vocabulary_", ""),
            ("stop_words_", ""),
            ("n_document_", ""),
        ],
    )
    def test_attributes(self, schema_loader, attribute_name, expected):
        """
        test function for test_attributes
        """
        vpy_model, py_model = self.create_tfidf_model(name=f"{schema_loader}.test_idf")
        X_vpy = getattr(vpy_model, attribute_name)

        if attribute_name == "tf_":
            vpy_pdf = X_vpy.to_pandas()
            cnt_vec = CountVectorizer()
            X_py = cnt_vec.fit_transform(self.documents)
            raw_pdf = pd.DataFrame(
                X_py.toarray(), columns=cnt_vec.get_feature_names_out()
            )
            data = []
            for i in list(raw_pdf.columns):
                for idx, row in raw_pdf[i].items():
                    if row:
                        data.append([idx + 1, i, row])
            py_tf_pdf = pd.DataFrame(data, columns=["py_row_id", "py_word", "py_tf"])

            merge_vdf_pdf = vpy_pdf.merge(
                py_tf_pdf,
                how="left",
                left_on=["row_id", "word", "tf"],
                right_on=["py_row_id", "py_word", "py_tf"],
            )
            res = (
                merge_vdf_pdf[
                    merge_vdf_pdf[["py_row_id", "py_word", "py_tf"]]
                    .isnull()
                    .any(axis=1)
                ]["word"]
                .unique()
                .tolist()
            )

            assert res == expected
        elif attribute_name == "idf_":
            vpy_pdf = X_vpy.to_pandas()
            py_idf_pdf = pd.DataFrame(
                {
                    "word": py_model.get_feature_names_out().tolist(),
                    "idf": py_model.idf_.tolist(),
                }
            )
            # inner join as some of the alphabets (like a) are ignored in sklearn(by default)
            merge_vdf_pdf = vpy_pdf.merge(py_idf_pdf, how="inner", on=["word"])

            pd.testing.assert_series_equal(
                merge_vdf_pdf["idf_log"],
                merge_vdf_pdf["idf"],
                check_names=False,
            )
        elif attribute_name == "vocabulary_":
            assert (
                list(
                    set(vpy_model.vocabulary_.tolist())
                    - set(list(py_model.vocabulary_.keys()))
                )
                == expected
            )
        elif attribute_name == "fixed_vocabulary_":
            assert vpy_model.fixed_vocabulary_ == py_model.fixed_vocabulary_
        elif attribute_name == "stop_words_":
            assert list(vpy_model.stop_words_) == list(py_model.stop_words_)
        else:
            assert vpy_model.n_document_ == len(self.documents)
