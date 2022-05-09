# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pytest
import pytest

# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
from verticapy import drop, set_option
from verticapy.datasets import load_titanic, load_amazon

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def amazon_vd():
    amazon = load_amazon()
    yield amazon
    drop(name="public.amazon")


class TestvDFCorrelation:
    def test_vDF_acf(self, amazon_vd):
        # spearmann method
        result1 = amazon_vd.acf(
            ts="date",
            column="number",
            p=20,
            by=["state"],
            unit="month",
            method="spearman",
        )
        plt.close("all")
        assert result1["value"][0] == pytest.approx(1)
        assert result1["confidence"][0] == pytest.approx(0.024396841824873748, 1e-2)
        assert result1.values["value"][10] == pytest.approx(0.494663471420921, 1e-2)
        assert result1.values["confidence"][10] == pytest.approx(
            0.06977116419369607, 1e-2
        )

        # pearson method
        result2 = amazon_vd.acf(
            ts="date",
            column="number",
            by=["state"],
            p=[1, 3, 6, 7],
            unit="year",
            method="pearson",
        )
        plt.close("all")
        assert result2["value"][0] == pytest.approx(1)
        assert result2["confidence"][0] == pytest.approx(0.024396841824873748, 1e-2)
        assert result2["value"][4] == pytest.approx(0.367, 1e-2)
        assert result2["confidence"][4] == pytest.approx(0.04080280865931269, 1e-2)

        # Autocorrelation Heatmap for each 'month' lag
        result3 = amazon_vd.acf(
            ts="date",
            column="number",
            by=["state"],
            p=12,
            unit="month",
            method="pearson",
            round_nb=3,
            acf_type="heatmap",
        )
        plt.close("all")
        assert result3["index"][1].replace('"', "") == "lag_12_number"
        assert result3["number"][1] == pytest.approx(0.778, 1e-2)
        assert result3["index"][5].replace('"', "") == "lag_10_number"
        assert result3["number"][5] == pytest.approx(0.334, 1e-2)

        # Autocorrelation Line for each 'month' lag
        result4 = amazon_vd.acf(
            ts="date",
            column="number",
            by=["state"],
            p=12,
            unit="month",
            method="pearson",
            acf_type="line",
        )
        plt.close("all")
        assert result4["value"][1] == pytest.approx(0.752, 1e-2)
        assert result4["confidence"][1] == pytest.approx(0.03627598368700659, 1e-2)
        assert result4["value"][6] == pytest.approx(-0.06, 1e-2)
        assert result4["confidence"][6] == pytest.approx(0.05273251493184901, 1e-2)

        # spearmanD method
        result5 = amazon_vd.acf(
            ts="date",
            column="number",
            p=20,
            by=["state"],
            unit="month",
            method="spearmanD",
        )
        plt.close("all")
        assert result5["value"][0] == pytest.approx(1)
        assert result5["confidence"][0] == pytest.approx(0.024396841824873748, 1e-2)
        assert result5.values["value"][10] == pytest.approx(0.5, 1e-2)
        assert result5.values["confidence"][10] == pytest.approx(
            0.06977116419369607, 1e-2
        )

    def test_vDF_chaid(self, titanic_vd):
        result = titanic_vd.chaid("survived", ["age", "fare", "sex"])
        tree = result.attributes_["tree"]
        assert tree["chi2"] == pytest.approx(345.12775126385327)
        assert tree["children"]["female"]["chi2"] == pytest.approx(10.472532457814179)
        assert tree["children"]["female"]["children"][127.6]["chi2"] == pytest.approx(
            3.479525088868805
        )
        assert tree["children"]["female"]["children"][127.6]["children"][19.0][
            "prediction"
        ][0] == pytest.approx(0.325581395348837)
        assert tree["children"]["female"]["children"][127.6]["children"][38.0][
            "prediction"
        ][1] == pytest.approx(0.720496894409938)
        assert not (tree["split_is_numerical"])
        assert tree["split_predictor"] == '"sex"'
        assert tree["split_predictor_idx"] == 2
        pred = result.predict([[3.0, 11.0, "male"], [11.0, 1.0, "female"]])
        assert pred[0] == 0
        assert pred[1] == 1
        pred = result.predict_proba([[3.0, 11.0, "male"], [11.0, 1.0, "female"]])
        assert pred[0][0] == pytest.approx(0.75968992)
        assert pred[0][1] == pytest.approx(0.24031008)
        assert pred[1][0] == pytest.approx(0.3255814)
        assert pred[1][1] == pytest.approx(0.6744186)

    def test_vDF_corr(self, titanic_vd):
        #
        # PEARSON
        #
        # testing vDataFrame.corr (method = 'pearson')
        result1 = titanic_vd.corr(
            columns=["survived", "age", "fare"], method="pearson",
        )
        plt.close("all")
        assert result1["survived"][0] == 1.0
        assert result1["survived"][1] == pytest.approx(-0.0422446185581737, 1e-2)
        assert result1["survived"][2] == pytest.approx(0.264150360783869, 1e-2)
        assert result1["age"][0] == pytest.approx(-0.0422446185581737, 1e-2)
        assert result1["age"][1] == 1.0
        assert result1["age"][2] == pytest.approx(0.178575164117464, 1e-2)
        assert result1["fare"][0] == pytest.approx(0.264150360783869, 1e-2)
        assert result1["fare"][1] == pytest.approx(0.178575164117464, 1e-2)
        assert result1["fare"][2] == 1.0

        # testing vDataFrame.corr (method = 'pearson') with focus
        result1_f = titanic_vd.corr(method="pearson", focus="survived")
        plt.close("all")
        assert result1_f["survived"][1] == pytest.approx(-0.336, 1e-2)
        assert result1_f["survived"][2] == pytest.approx(0.264, 1e-2)

        #
        # SPEARMAN
        #
        # testing vDataFrame.corr (method = 'spearman')
        titanic_vd_gb = titanic_vd.groupby(
            ["age"], ["AVG(survived) AS survived", "AVG(fare) AS fare"]
        )
        titanic_vd_gb = titanic_vd_gb.groupby(
            ["fare"], ["AVG(age) AS age", "AVG(survived) AS survived"]
        )
        titanic_vd_gb = titanic_vd_gb.groupby(
            ["survived"], ["AVG(age) AS age", "AVG(fare) AS fare"]
        )
        result2 = titanic_vd_gb.corr(
            columns=["survived", "age", "fare"], method="spearman",
        )
        plt.close("all")
        assert result2["survived"][0] == 1.0
        assert result2["survived"][1] == pytest.approx(-0.221388367729831, 1e-2)
        assert result2["survived"][2] == pytest.approx(0.425515947467167, 1e-2)
        assert result2["age"][0] == pytest.approx(-0.221388367729831, 1e-2)
        assert result2["age"][1] == 1.0
        assert result2["age"][2] == pytest.approx(0.287617260787992, 1e-2)
        assert result2["fare"][0] == pytest.approx(0.425515947467167, 1e-2)
        assert result2["fare"][1] == pytest.approx(0.287617260787992, 1e-2)
        assert result2["fare"][2] == 1.0

        # testing vDataFrame.corr (method = 'spearman') with focus
        result2_f = titanic_vd_gb.corr(focus="survived", method="spearman")
        plt.close("all")
        assert result2_f["survived"][1] == pytest.approx(0.425515947467167, 1e-2)
        assert result2_f["survived"][2] == pytest.approx(-0.221388367729831, 1e-2)

        #
        # KENDALL
        #
        # testing vDataFrame.corr (method = 'kendall')
        result3 = titanic_vd.corr(
            columns=["survived", "age", "fare"], method="kendall",
        )
        plt.close("all")
        assert result3["survived"][0] == 1.0
        assert result3["survived"][1] == pytest.approx(-0.0149530691050183, 1e-2)
        assert result3["survived"][2] == pytest.approx(0.264138930414481, 1e-2)
        assert result3["age"][0] == pytest.approx(-0.0149530691050183, 1e-2)
        assert result3["age"][1] == 1.0
        assert result3["age"][2] == pytest.approx(0.0844989716189637, 1e-2)
        assert result3["fare"][0] == pytest.approx(0.264138930414481, 1e-2)
        assert result3["fare"][1] == pytest.approx(0.0844989716189637, 1e-2)
        assert result3["fare"][2] == 1.0

        # testing vDataFrame.corr (method = 'kendall') with focus
        result3_f = titanic_vd.corr(focus="survived", method="kendall")
        plt.close("all")
        assert result3_f["survived"][1] == pytest.approx(-0.317426126117454, 1e-2)
        assert result3_f["survived"][2] == pytest.approx(0.264138930414481, 1e-2)

        #
        # BISERIAL POINT
        #
        # testing vDataFrame.corr (method = 'biserial')
        result4 = titanic_vd.corr(
            columns=["survived", "age", "fare"], method="biserial",
        )
        plt.close("all")
        assert result4["survived"][0] == 1.0
        assert result4["survived"][1] == pytest.approx(-0.0422234273762242, 1e-2)
        assert result4["survived"][2] == pytest.approx(0.264043222121672, 1e-2)
        assert result4["age"][0] == pytest.approx(-0.0422234273762242, 1e-2)
        assert result4["age"][1] == 1.0
        assert result4["fare"][0] == pytest.approx(0.264043222121672, 1e-2)
        assert result4["fare"][2] == 1.0

        # testing vDataFrame.corr (method = 'biserial') with focus
        result4_f = titanic_vd.corr(focus="survived", method="biserial")
        plt.close("all")
        assert result4_f["survived"][1] == pytest.approx(-0.335720838027055, 1e-2)
        assert result4_f["survived"][2] == pytest.approx(0.264043222121672, 1e-2)

        #
        # CRAMER'S V
        #
        # testing vDataFrame.corr (method = 'cramer')
        result5 = titanic_vd.corr(
            columns=["survived", "pclass", "embarked"], method="cramer"
        )
        plt.close("all")
        assert result5["survived"][0] == 1.0
        assert result5["survived"][1] == pytest.approx(0.3358661117846154, 1e-2)
        assert result5["survived"][2] == pytest.approx(0.18608072188932145, 1e-2)
        assert result5["pclass"][0] == pytest.approx(0.3358661117846154, 1e-2)
        assert result5["pclass"][1] == 1.0
        assert result5["pclass"][2] == pytest.approx(0.27453049870161333, 1e-2)
        assert result5["embarked"][0] == pytest.approx(0.18608072188932145, 1e-2)
        assert result5["embarked"][1] == pytest.approx(0.27453049870161333, 1e-2)
        assert result5["embarked"][2] == 1.0

        # testing vDataFrame.corr (method = 'cramer') with focus
        result5_f = titanic_vd.corr(focus="survived", method="cramer")
        plt.close("all")
        assert result5_f["survived"][1] == pytest.approx(0.73190924565401, 1e-2)
        assert result5_f["survived"][2] == pytest.approx(0.6707486879228794, 1e-2)

        #
        # DENSE SPEARMAN
        #
        # testing vDataFrame.corr (method = 'spearmanD')
        result6 = titanic_vd_gb.corr(
            columns=["survived", "age", "fare"], method="spearmanD",
        )
        plt.close("all")
        assert result6["survived"][0] == 1.0
        assert result6["survived"][1] == pytest.approx(-0.221388367729831, 1e-2)
        assert result6["survived"][2] == pytest.approx(0.425515947467167, 1e-2)
        assert result6["age"][0] == pytest.approx(-0.221388367729831, 1e-2)
        assert result6["age"][1] == 1.0
        assert result6["age"][2] == pytest.approx(0.287617260787992, 1e-2)
        assert result6["fare"][0] == pytest.approx(0.425515947467167, 1e-2)
        assert result6["fare"][1] == pytest.approx(0.287617260787992, 1e-2)
        assert result6["fare"][2] == 1.0

        # testing vDataFrame.corr (method = 'spearmanD') with focus
        result6_f = titanic_vd_gb.corr(focus="survived", method="spearmanD")
        plt.close("all")
        assert result6_f["survived"][1] == pytest.approx(0.425515947467167, 1e-2)
        assert result6_f["survived"][2] == pytest.approx(-0.221388367729831, 1e-2)

    def test_vDF_corr_pvalue(self, titanic_vd):
        assert titanic_vd.corr_pvalue("age", "fare", "pearson") == (
            pytest.approx(0.178575164117468, 1e-2),
            pytest.approx(1.3923308548466764e-08, 1e-2),
        )
        assert titanic_vd.corr_pvalue("age", "fare", "spearman") == (
            pytest.approx(0.0045193585753828, 1e-2),
            pytest.approx(0.8899833744540833, 1e-2),
        )
        assert titanic_vd.corr_pvalue("age", "fare", "spearmanD") == (
            pytest.approx(0.0045193585753828, 1e-2),
            pytest.approx(0.8899833744540833, 1e-2),
        )
        assert titanic_vd.corr_pvalue("age", "fare", "kendallA") == (
            pytest.approx(0.12796714496175657, 1e-2),
            pytest.approx(1.4735187810450437e-09, 1e-2),
        )
        assert titanic_vd.corr_pvalue("age", "fare", "kendallB") == (
            pytest.approx(0.0844989716189637, 1e-2),
            pytest.approx(1.1056646764730614e-09, 1e-2),
        )
        assert titanic_vd.corr_pvalue("age", "fare", "kendallC") == (
            pytest.approx(0.12919864967860847, 1e-2),
            pytest.approx(1.1056646764730614e-09, 1e-2),
        )
        assert titanic_vd.corr_pvalue("survived", "fare", "biserial") == (
            pytest.approx(0.264043222121672, 1e-2),
            pytest.approx(4.097598100216442e-21, 1e-2),
        )
        assert titanic_vd.corr_pvalue("survived", "pclass", "cramer") == (
            pytest.approx(0.3358661117846154, 1e-2),
            pytest.approx(3.507947423216931e-61, 1e-2),
        )

    def test_vDF_cov(self, titanic_vd):
        # testing vDataFrame.cov
        result = titanic_vd.cov(columns=["survived", "age", "fare"])
        plt.close("all")
        assert result["survived"][0] == pytest.approx(0.231685181342251, 1e-2)
        assert result["survived"][1] == pytest.approx(-0.297583583247234, 1e-2)
        assert result["survived"][2] == pytest.approx(6.69214075159394, 1e-2)
        assert result["age"][0] == pytest.approx(-0.297583583247234, 1e-2)
        assert result["age"][1] == pytest.approx(208.169014723609, 1e-2)
        assert result["age"][2] == pytest.approx(145.057125218791, 1e-2)
        assert result["fare"][0] == pytest.approx(6.69214075159394, 1e-2)
        assert result["fare"][1] == pytest.approx(145.057125218791, 1e-2)
        assert result["fare"][2] == pytest.approx(2769.36114247479, 1e-2)

        # testing vDataFrame.cov with focus
        result_f = titanic_vd.cov(
            columns=["survived", "age", "fare"], focus="survived",
        )
        assert result_f["survived"][0] == pytest.approx(6.69214075159394, 1e-2)
        assert result_f["survived"][1] == pytest.approx(-0.297583583247234, 1e-2)
        assert result_f["survived"][2] == pytest.approx(0.231685181342251, 1e-2)
        plt.close("all")

    def test_vDF_iv_woe(self, titanic_vd):
        # testing vDataFrame.iv_woe
        result = titanic_vd.iv_woe("survived")
        plt.close("all")
        assert result["iv"][0] == pytest.approx(1.272254799126849)
        assert result["iv"][1] == pytest.approx(1.148751293230747)
        assert result["iv"][2] == pytest.approx(0.4951028280802058)
        # testing vDataFrame[].iv_woe
        result2 = titanic_vd["pclass"].iv_woe("survived")
        assert result2["iv"][-1] == pytest.approx(0.4951028280802058)
        assert result2["non_events"][-1] == pytest.approx(784)

    def test_vDF_pivot_table_chi2(self, titanic_vd):
        result = titanic_vd.pivot_table_chi2("survived")
        assert result["chi2"][0] == pytest.approx(345.12775126385327)
        assert result["chi2"][1] == pytest.approx(139.2026595859198)
        assert result["chi2"][2] == pytest.approx(103.48373526643627)
        assert result["chi2"][3] == pytest.approx(53.47413727076916)
        assert result["chi2"][4] == pytest.approx(44.074789072247576)
        assert result["chi2"][5] == pytest.approx(42.65927519250441)
        assert result["chi2"][6] == pytest.approx(34.54227539207669)
        assert result["chi2"][7] == pytest.approx(-2.6201263381153694e-14)
        assert result["p_value"][0] == pytest.approx(4.8771014178794746e-77)
        assert result["p_value"][1] == pytest.approx(6.466448884215497e-32)
        assert result["p_value"][2] == pytest.approx(5.922792773022131e-31)
        assert result["p_value"][3] == pytest.approx(2.9880194205753303e-09)
        assert result["p_value"][4] == pytest.approx(7.143900900120299e-08)
        assert result["p_value"][5] == pytest.approx(5.4532585750903e-10)
        assert result["p_value"][6] == pytest.approx(0.0028558258758971957)
        set_option("random_state", 0)
        result = titanic_vd.pivot_table_chi2("survived", method="smart")
        assert result["chi2"][0] == pytest.approx(345.12775126385327)
        assert result["chi2"][1] == pytest.approx(187.75090682844288)
        assert result["chi2"][2] == pytest.approx(139.2026595859198)
        assert result["chi2"][3] == pytest.approx(53.474137270768885)
        assert result["chi2"][4] == pytest.approx(44.074789072247576)
        assert result["chi2"][5] == pytest.approx(42.65927519250441)
        assert result["chi2"][6] == pytest.approx(38.109904618027755)
        assert result["chi2"][7] == pytest.approx(0.0)

    def test_vDF_pacf(self, amazon_vd):
        # testing vDataFrame.pacf
        result = amazon_vd.pacf(column="number", ts="date", by=["state"], p=5,)
        plt.close("all")
        assert result["value"][0] == 1.0
        assert result["value"][1] == pytest.approx(0.672667529541858, 1e-2)
        assert result["value"][2] == pytest.approx(-0.188727403801382, 1e-2)
        assert result["value"][3] == pytest.approx(0.022206688265849, 1e-2)
        assert result["value"][4] == pytest.approx(-0.0819798501305434, 1e-2)
        assert result["value"][5] == pytest.approx(-0.00663606854011195, 1e-2)

    def test_vDF_regr(self, titanic_vd):
        # testing vDataFrame.regr (method = 'alpha')
        result1 = titanic_vd.regr(columns=["survived", "age", "fare"], method="alpha",)
        plt.close("all")
        assert result1["survived"][0] == 0.0
        assert result1["survived"][1] == pytest.approx(0.435280333103508, 1e-2)
        assert result1["survived"][2] == pytest.approx(0.282890247028015, 1e-2)
        assert result1["age"][0] == pytest.approx(30.6420462046205, 1e-2)
        assert result1["age"][1] == 0.0
        assert result1["age"][2] == pytest.approx(28.4268042866199, 1e-2)
        assert result1["fare"][0] == pytest.approx(23.425595019157, 1e-2)
        assert result1["fare"][1] == pytest.approx(16.1080039795446, 1e-2)
        assert result1["fare"][2] == 0.0

        # testing vDataFrame.regr (method = 'beta')
        result2 = titanic_vd.regr(columns=["survived", "age", "fare"], method="beta")
        plt.close("all")
        assert result2["survived"][0] == 1.0
        assert result2["survived"][1] == pytest.approx(-0.00142952871080426, 1e-2)
        assert result2["survived"][2] == pytest.approx(0.00241649261591561, 1e-2)
        assert result2["age"][0] == pytest.approx(-1.2483889156179, 1e-2)
        assert result2["age"][1] == 1.0
        assert result2["age"][2] == pytest.approx(0.0456059549185254, 1e-2)
        assert result2["fare"][0] == pytest.approx(28.8746643141762, 1e-2)
        assert result2["fare"][1] == pytest.approx(0.69923081967147, 1e-2)
        assert result2["fare"][2] == 1.0

        # testing vDataFrame.regr (method = 'r2')
        result3 = titanic_vd.regr(columns=["survived", "age", "fare"], method="r2",)
        plt.close("all")
        assert result3["survived"][0] == 1.0
        assert result3["survived"][1] == pytest.approx(0.00178460779712559, 1e-2)
        assert result3["survived"][2] == pytest.approx(0.0697754131022489, 1e-2)
        assert result3["age"][0] == pytest.approx(0.00178460779712559, 1e-2)
        assert result3["age"][1] == 1.0
        assert result3["age"][2] == pytest.approx(0.0318890892395806, 1e-2)
        assert result3["fare"][0] == pytest.approx(0.0697754131022489, 1e-2)
        assert result3["fare"][1] == pytest.approx(0.0318890892395806, 1e-2)
        assert result3["fare"][2] == 1.0
