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

# Pytest
import pytest

# VerticaPy
from verticapy import drop, set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_market
from verticapy.learn.decomposition import MCA

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def market_vd():
    market = load_market()
    yield market
    drop(
        name="public.market",
    )


@pytest.fixture(scope="module")
def model(market_vd):
    model_class = MCA(
        "mca_model_test",
    )
    model_class.drop()
    model_class.fit(market_vd.cdt())
    yield model_class
    model_class.drop()


class TestMCA:
    def test_repr(self, model):
        assert model.__repr__() == "<MCA>"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_PCA("Form_Boiled", "Form_Canned",'
        result_sql = model.deploySQL()

        assert expected_sql in result_sql

    def test_deployInverseSQL(self, model):
        expected_sql = 'APPLY_INVERSE_PCA("Form_Boiled", "Form_Canned",'
        result_sql = model.deployInverseSQL()

        assert expected_sql in result_sql

    def test_drop(self, market_vd):
        current_cursor().execute("DROP MODEL IF EXISTS mca_model_test_drop")
        model_test = MCA(
            "mca_model_test_drop",
        )
        model_test.fit(market_vd.cdt())

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'mca_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "mca_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'mca_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_get_vertica_attributes(self, model):
        m_att = model.get_vertica_attributes()

        assert m_att["attr_name"] == [
            "columns",
            "singular_values",
            "principal_components",
            "counters",
            "call_string",
        ]
        assert m_att["attr_fields"] == [
            "index, name, mean, sd",
            "index, value, explained_variance, accumulated_explained_variance",
            "index, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, PC11, PC12, PC13, PC14, PC15, PC16, PC17, PC18, PC19, PC20, PC21, PC22, PC23, PC24, PC25, PC26, PC27, PC28, PC29, PC30, PC31, PC32, PC33, PC34, PC35, PC36, PC37, PC38, PC39, PC40, PC41, PC42, PC43, PC44, PC45, PC46, PC47, PC48, PC49, PC50, PC51, PC52",
            "counter_name, counter_value",
            "call_string",
        ]
        assert m_att["#_of_rows"] == [52, 52, 52, 3, 1]

        m_att_details = model.get_vertica_attributes(attr_name="principal_components")

        assert m_att_details["PC1"][0] == pytest.approx(-8.40681285066429e-18, abs=1)
        assert m_att_details["PC1"][1] == pytest.approx(6.69930488797486e-17, abs=1)
        assert m_att_details["PC1"][2] == pytest.approx(-8.57930855866453e-17, abs=1)
        assert m_att_details["PC2"][0] == pytest.approx(-0.00490495400370651, abs=1)
        assert m_att_details["PC2"][1] == pytest.approx(-0.00591611533838104, abs=1)
        assert m_att_details["PC2"][2] == pytest.approx(-0.00508231564856379, abs=1)
        assert m_att_details["PC3"][0] == pytest.approx(-0.00205641099322215, abs=1)
        assert m_att_details["PC3"][1] == pytest.approx(-0.129029981029071, abs=1)
        assert m_att_details["PC3"][2] == pytest.approx(-0.00343091837157043, abs=1)

    def test_get_params(self, model):
        assert model.get_params() == {}

    @skip_plt
    def test_plot(self, model):
        result = model.plot()
        assert len(result.get_default_bbox_extra_artists()) == 8
        result = model.plot(dimensions=(2, 3))
        assert len(result.get_default_bbox_extra_artists()) == 8

    @skip_plt
    def test_plot_var(self, model):
        result = model.plot_var()
        assert len(result.get_default_bbox_extra_artists()) == 62
        result = model.plot_var(dimensions=(2, 3))
        assert len(result.get_default_bbox_extra_artists()) == 62
        result = model.plot_var(dimensions=(2, 3), method="cos2")
        assert len(result.get_default_bbox_extra_artists()) == 62
        result = model.plot_var(dimensions=(2, 3), method="contrib")
        assert len(result.get_default_bbox_extra_artists()) == 62

    @skip_plt
    def test_plot_contrib(self, model):
        result = model.plot_contrib()
        assert len(result.get_default_bbox_extra_artists()) == 114
        result = model.plot_contrib(dimension=2)
        assert len(result.get_default_bbox_extra_artists()) == 114

    @skip_plt
    def test_plot_cos2(self, model):
        result = model.plot_cos2()
        assert len(result.get_default_bbox_extra_artists()) == 111
        result = model.plot_cos2(dimensions=(2, 3))
        assert len(result.get_default_bbox_extra_artists()) == 111

    @skip_plt
    def test_plot_scree(self, model):
        result = model.plot_scree()
        assert len(result.get_default_bbox_extra_artists()) == 113

    @skip_plt
    def test_plot_circle(self, model):
        result = model.plot_circle()
        assert len(result.get_default_bbox_extra_artists()) == 114
        result = model.plot_circle(dimensions=(2, 3))
        assert len(result.get_default_bbox_extra_artists()) == 114

    @pytest.mark.skip(reason="test is not stable")
    def test_to_python(self, model):
        prediction = model.to_python()([[0 for i in range(52)]])
        assert sum(sum(prediction)) == pytest.approx(27.647893864490204, abs=1)

    @pytest.mark.skip(reason="to_sql not yet implemented for MCA")
    def test_to_sql(self, model):
        pass

    @pytest.mark.skip(reason="to_memmodel not yet implemented for MCA")
    def test_to_memmodel(self, model):
        pass

    def test_get_transform(self, model):
        market_trans = model.transform()
        assert market_trans["col1"].mean() == pytest.approx(0.0, abs=1e-6)
        assert market_trans["col2"].mean() == pytest.approx(0.0, abs=1e-6)
        assert market_trans["col3"].mean() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.skip(reason="test is not stable")
    def test_get_inverse_transform(self, model):
        market_trans = model.transform()
        market_trans = model.inverse_transform(market_trans)
        assert market_trans["name_apples"].sum() == pytest.approx(-313.0)

    def test_pca_score(self, model):
        result = model.score()
        assert result["Score"][0] == pytest.approx(0.0, abs=1e-6)
        assert result["Score"][1] == pytest.approx(0.0, abs=1e-6)
        assert result["Score"][2] == pytest.approx(0.0, abs=1e-6)

    def test_set_params(self, model):
        model.set_params({})
        assert model.get_params() == {}

    def test_model_from_vDF(self, market_vd):
        current_cursor().execute("DROP MODEL IF EXISTS mca_vDF")
        model_test = MCA(
            "mca_vDF",
        )
        model_test.fit(market_vd.cdt())
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'mca_vDF'"
        )
        assert current_cursor().fetchone()[0] == "mca_vDF"
        model_test.drop()

    def test_optional_name(self):
        model = MCA()
        assert model.model_name is not None
