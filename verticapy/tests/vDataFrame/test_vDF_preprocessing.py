# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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

import pytest
from verticapy import vDataFrame, drop_table, errors


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    drop_table(name="public.titanic", cursor=base.cursor, print_info=False)


@pytest.fixture(scope="module")
def iris_vd(base):
    from verticapy.learn.datasets import load_iris

    iris = load_iris(cursor=base.cursor)
    yield iris
    drop_table(name="public.iris", cursor=base.cursor)


@pytest.fixture(scope="module")
def market_vd(base):
    from verticapy.learn.datasets import load_market

    market = load_market(cursor=base.cursor)
    yield market
    drop_table(name="public.market", cursor=base.cursor, print_info=False)


class TestvDFPreprocessing:
    def test_vDF_decode(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["sex"].decode(values = {"female": 1, "male": 0}, others = 2)

        assert titanic_copy["sex"].distinct() == [0, 1]

    @pytest.mark.xfail(reason = "The temp view created in the smart method is not properly cleaned up"
                       " in case of failuer")
    def test_vDF_discretize(self, titanic_vd):
        ### method = "same_width"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].discretize(method = "same_width", h = 10)
        assert titanic_copy["age"].distinct() == ['[0;10]', '[10;20]', '[20;30]', '[30;40]',
                                                  '[40;50]', '[50;60]', '[60;70]', '[70;80]', '[80;90]']

        ### method = "same_freq"
        titanic_copy = titanic_vd.copy()

        # expected exception
        with pytest.raises(errors.ParameterError) as exception_info:
            titanic_copy["age"].discretize(method = "same_freq", bins = 1)
        # checking the error message
        assert exception_info.match("Parameter 'bins' must be greater or equals to 2 in case "
                                    "of discretization using the method 'same_freq'")

        titanic_copy["age"].discretize(method = "same_freq", bins = 5)
        assert titanic_copy["age"].distinct() == ['[0.330;21.000]', '[21.000;28.000]',
                                                  '[28.000;39.000]', '[39.000;80.000]']

        ### method = "smart"
        titanic_copy = titanic_vd.copy()

        # expected exception
        with pytest.raises(AssertionError) as exception_info:
            titanic_copy["age"].discretize(method = "smart", response = "survived", bins = 6, min_bin_size = 0)
        # checking the error message
        assert exception_info.match("Incorrect parameter 'min_samples_leaf'")

        titanic_copy["age"].discretize(method = "smart", response = "survived", bins = 6, min_bin_size = 25)
        # because randomness in the method the distinct values may not stay the same
        assert len(titanic_copy["age"].distinct()) == 6

        ### method = "topk"
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_extract(' ([A-Za-z])+\.')

        # expected exception
        with pytest.raises(errors.ParameterError) as exception_info:
            titanic_copy["name"].discretize(method = "topk", k = 0, new_category = "rare")
        # checking the error message
        assert exception_info.match("Parameter 'k' must be greater or equals to 2 in case of "
                                    "discretization using the method 'topk'")

        titanic_copy["name"].discretize(method = "topk", k = 5, new_category = "rare")
        assert titanic_copy["name"].distinct() == [' Dr.', ' Master.', ' Miss.', ' Mr.', ' Mrs.', 'rare']

        ### method = "auto" for numerical vcolumn
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].discretize()
        assert titanic_copy["age"].distinct() == ['[0.00;7.24]', '[14.48;21.72]', '[21.72;28.96]',
                                                  '[28.96;36.20]', '[36.20;43.44]', '[43.44;50.68]',
                                                  '[50.68;57.92]', '[57.92;65.16]', '[65.16;72.40]',
                                                  '[7.24;14.48]', '[72.40;79.64]', '[79.64;86.88]']

    def test_vDF_get_dummies(self, iris_vd):
        ### Testing vDataFrame.get_dummies
        # use_numbers_as_suffix = False
        iris_copy = iris_vd.copy()
        iris_copy.get_dummies()
        assert iris_copy.get_columns() == ['"SepalLengthCm"', '"SepalWidthCm"', '"PetalLengthCm"',
                                           '"PetalWidthCm"', '"Species"', '"Species_Iris-setosa"',
                                           '"Species_Iris-versicolor"']

        # use_numbers_as_suffix = True
        iris_copy = iris_vd.copy()
        iris_copy.get_dummies(use_numbers_as_suffix = True)
        assert iris_copy.get_columns() == ['"SepalLengthCm"', '"SepalWidthCm"', '"PetalLengthCm"',
                                           '"PetalWidthCm"', '"Species"', '"Species_0"', '"Species_1"']

        ### Testing vDataFrame.get_dummies
        # use_numbers_as_suffix = False
        iris_copy = iris_vd.copy()
        iris_copy["Species"].get_dummies(prefix = "D", prefix_sep = "--")
        assert iris_copy.get_columns() == ['"SepalLengthCm"', '"SepalWidthCm"', '"PetalLengthCm"',
                                           '"PetalWidthCm"', '"Species"', '"D--Iris-setosa"',
                                           '"D--Iris-versicolor"']

        # use_numbers_as_suffix = True
        iris_copy = iris_vd.copy()
        iris_copy["Species"].get_dummies(use_numbers_as_suffix = True)
        assert iris_copy.get_columns() == ['"SepalLengthCm"', '"SepalWidthCm"', '"PetalLengthCm"',
                                           '"PetalWidthCm"', '"Species"', '"Species_0"', '"Species_1"']

    def test_vDF_lable_encode(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["embarked"].label_encode()

        assert titanic_copy["embarked"].distinct() == [0, 1, 2, 3]

    def test_vDF_mean_encode(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["embarked"].mean_encode(response_column = "survived")
        result = titanic_copy["embarked"].distinct()

        result[0] == pytest.approx(0.2735849)
        result[1] == pytest.approx(0.3241695)
        result[2] == pytest.approx(0.5375494)
        result[3] == pytest.approx(1.0)

    def test_vDF_dropna(self, titanic_vd):
        # Testing vDataFrame.dropna
        titanic_copy = titanic_vd.copy()
        titanic_copy.dropna(columns = ["fare", "embarked", "age"], print_info = False)
        result = titanic_copy.count(columns = ["fare", "embarked", "age"])
        
        assert result["count"][0] == 994
        assert result["count"][1] == 994
        assert result["count"][2] == 994

        # Testing vDataFrame[].dropna
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].dropna(print_info = False)
        assert titanic_copy.count(["age"])["count"][0] == 997

    def test_vDF_fillna(self, titanic_vd):
        # Testing vDataFrame.fillna
        titanic_copy = titanic_vd.copy()
        titanic_copy.fillna(val = {"boat": "No boat"}, method = {"age": "mean", "embarked": "mode",
                                                                 "fare": "median", "body" : "0ifnull",
                                                                 "cabin" : "auto"}, print_info = False)

        result = titanic_copy.count(["age", "fare", "embarked", "boat", "cabin", "body"])

        assert result["percent"][0] == 100
        assert result["percent"][1] == 100
        assert result["percent"][2] == 100
        assert result["percent"][3] == 100
        assert result["percent"][4] == 100
        assert result["percent"][5] == 100

        # Testing vDataFrame[].fillna
        titanic_copy = titanic_vd.copy()

        assert titanic_copy["age"].count() == 997
        titanic_copy["age"].fillna(method = "mean", by = ["pclass", "sex"], print_info = False)
        assert titanic_copy["age"].count() == 1234

        assert titanic_copy["embarked"].count() == 1232
        titanic_copy["embarked"].fillna(method = "mode", print_info = False)
        assert titanic_copy["embarked"].count() == 1234

        assert titanic_copy["fare"].count() == 1233
        titanic_copy["fare"].fillna(method = "median", by = ["pclass", "sex"], print_info = False)
        assert titanic_copy["fare"].count() == 1234

        assert titanic_copy["cabin"].count() == 286
        titanic_copy["cabin"].fillna(method = "bfill", order_by = ["pclass", "cabin"], print_info = False)
        assert titanic_copy["cabin"].count() == 1234

        assert titanic_copy["home.dest"].count() == 706
        titanic_copy["home.dest"].fillna(method = "ffill", order_by = ["ticket"], print_info = False)
        assert titanic_copy["home.dest"].count() == 1234

        assert titanic_copy["body"].count() == 118
        assert titanic_copy["body"].mean() == pytest.approx(164.1440677)
        titanic_copy["body"].fillna(method = "auto", print_info = False)
        assert titanic_copy["body"].count() == 1234
        assert titanic_copy["body"].mean() == pytest.approx(164.1440677)

        assert titanic_copy["boat"].count() == 439
        assert titanic_copy["boat"].mode(dropna = True) == '13'
        titanic_copy["boat"].fillna(print_info = False)
        assert titanic_copy["boat"].count() == 1234
        assert titanic_copy["boat"].mode() == '13'

    def test_vDF_clip(self, market_vd):
        market_copy = market_vd.copy()
        market_copy["Price"].clip(lower = 1.0, upper = 4.0)

        assert market_copy["Price"].mean() == pytest.approx(1.95852599)

    @pytest.mark.xfail(reason = "method mean fails")
    def test_vDF_fill_outliers(self, market_vd):
        # method = "null"
        market_copy = market_vd.copy()
        assert market_copy["Price"].count() == 314
        market_copy["Price"].fill_outliers(method = "null", threshold = 1.5, use_threshold = True)
        assert market_copy["Price"].count() == 286

        # method = "winsorize", use_threshold = True
        market_copy = market_vd.copy()
        assert market_copy["Price"].mean == pytest.approx(2.077510986)
        market_copy["Price"].fill_outliers(method = "winsorize", threshold = 1.5, use_threshold = True)
        assert market_copy["Price"].mean() == pytest.approx(1.942654626)

        # method = "winsorize", use_threshold = False
        market_copy = market_vd.copy()
        market_copy["Price"].fill_outliers(method = "winsorize", alpha = 0.2, use_threshold = False)
        assert market_copy["Price"].mean() == pytest.approx(1.8211456)

        # method = "mean"
        market_copy = market_vd.copy()
        market_copy["Price"].fill_outliers(method = "mean", threshold = 1.5)
        assert market_copy["Price"].mean() == pytest.approx(0)

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_normalize(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_outliers(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_astype(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_bool_to_int(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_rename(self):
        pass
