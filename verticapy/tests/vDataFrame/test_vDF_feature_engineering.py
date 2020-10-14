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
import datetime
from verticapy import vDataFrame, drop_table, errors


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    amazon.set_display_parameters(print_info=False)
    yield amazon
    drop_table(
        name="public.amazon", cursor=base.cursor,
    )


@pytest.fixture(scope="module")
def iris_vd(base):
    from verticapy.learn.datasets import load_iris

    iris = load_iris(cursor=base.cursor)
    yield iris
    drop_table(name="public.iris", cursor=base.cursor)


@pytest.fixture(scope="module")
def smart_meters_vd(base):
    from verticapy.learn.datasets import load_smart_meters

    smart_meters = load_smart_meters(cursor=base.cursor)
    yield smart_meters
    drop_table(name="public.smart_meters", cursor=base.cursor)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    drop_table(name="public.titanic", cursor=base.cursor)


class TestvDFFeatureEngineering:
    def test_vDF_cummax(self, amazon_vd):
        amazon_copy = amazon_vd.copy()
        amazon_copy.cummax(column = "number", by = ["state"], order_by = ["date"], name = "cummax_number")

        assert amazon_copy["cummax_number"].max() == 25963.0

    def test_vDF_cummin(self, amazon_vd):
        amazon_copy = amazon_vd.copy()
        amazon_copy.cummin(column = "number", by = ["state"], order_by = ["date"], name = "cummin_number")

        assert amazon_copy["cummin_number"].min() == 0.0

    def test_vDF_cumprod(self, iris_vd):
        iris_copy = iris_vd.copy()
        iris_copy.cumprod(column = "PetalWidthCm", by = ["Species"],
                          order_by = ["PetalLengthCm"], name = "cumprod_number")

        assert iris_copy["cumprod_number"].max() == 1347985569095150.0

    def test_vDF_cumsum(self, amazon_vd):
        amazon_copy = amazon_vd.copy()
        amazon_copy.cumsum(column = "number", by = ["state"], order_by = ["date"], name = "cumsum_number")

        assert amazon_copy["cumsum_number"].max() == pytest.approx(723629.0)

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_rolling(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_analytic(self):
        pass

    @pytest.mark.xfail(reason = "the answers change time-to-time")
    def test_vDF_asfreq(self, smart_meters_vd):
        # bfill method
        result1 = smart_meters_vd.asfreq(ts = "time", rule = "1 hour",
                                         method = {"val": "bfill"}, by = ["id"])
        result1.sort({"time" : "asc"})

        assert result1.shape() == (148189, 3)
        assert result1["time"][2] == datetime.datetime(2014, 1, 1, 3, 0)
        assert result1["id"][2] == 2
        assert result1["val"][2] == pytest.approx(0.037)

        # ffill
        result2 = smart_meters_vd.asfreq(ts = "time", rule = "1 hour",
                                         method = {"val": "ffill"}, by = ["id"])
        result2.sort({"time" : "asc"})

        assert result2.shape() == (148189, 3)
        assert result2["time"][2] == datetime.datetime(2014, 1, 1, 3, 0)
        assert result2["id"][2] == 2
        assert result2["val"][2] == pytest.approx(0.037)

        # linear method
        result3 = smart_meters_vd.asfreq(ts = "time", rule = "1 hour",
                                         method = {"val": "linear"}, by = ["id"])
        result3.sort({"time" : "asc"})

        assert result3.shape() == (148189, 3)
        assert result3["time"][2] == datetime.datetime(2014, 1, 1, 3, 0)
        assert result3["id"][2] == 2
        assert result3["val"][2] == pytest.approx(0.06116129032)

    @pytest.mark.xfail(reason = "the answers change time-to-time")
    def test_vDF_sessionize(self, smart_meters_vd):
        smart_meters_copy = smart_meters_vd.copy()

        # expected exception
        with pytest.raises(errors.QueryError) as exception_info:
            smart_meters_copy.sessionize(ts = "time", by = ["id"], session_threshold = "1 time", name = "slot")
        # checking the error message
        assert exception_info.match("seems to be incorrect")

        smart_meters_copy.sessionize(ts = "time", by = ["id"], session_threshold = "1 hour", name = "slot")
        smart_meters_copy.sort({"time" : "asc"})

        assert smart_meters_copy.shape() == (11844, 4)
        assert smart_meters_copy["time"][2] == datetime.datetime(2014, 1, 1, 15, 30)
        assert smart_meters_copy["val"][2] == 0.235
        assert smart_meters_copy["id"][2] == 9
        assert smart_meters_copy["slot"][2] == 1

    def test_vDF_case_when(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy.case_when(name = "age_category",
                               conditions = {"age < 12": "children",
                                             "age < 18": "teenagers",
                                             "age > 60": "seniors",
                                             "age < 25": "young adults"},
                               others = "adults")

        assert titanic_copy["age_category"].distinct() == \
            ['adults', 'children', 'seniors', 'teenagers', 'young adults']

    def test_vDF_eval(self, titanic_vd):
        # new feature creation
        titanic_copy = titanic_vd.copy()
        titanic_copy.eval(name = "family_size", expr = "parch + sibsp + 1")

        assert titanic_copy["family_size"].max() == 11

        # Customized SQL code evaluation
        titanic_copy = titanic_vd.copy()
        titanic_copy.eval(name = "has_life_boat", expr = "CASE WHEN boat IS NULL THEN 0 ELSE 1 END")

        assert titanic_copy["boat"].count() == titanic_copy["has_life_boat"].sum()

    def test_vDF_abs(self, titanic_vd):
        # Testing vDataFrame.abs
        titanic_copy = titanic_vd.copy()

        titanic_copy.normalize(["fare", "age"])
        assert titanic_copy["fare"].min() == pytest.approx(-0.64513441)
        assert titanic_copy["age"].min() == pytest.approx(-2.0659389)

        titanic_copy.abs(["fare", "age"])
        assert titanic_copy["fare"].min() == pytest.approx(0.001082821)
        assert titanic_copy["age"].min() == pytest.approx(0.010561423)

        # Testing vDataFrame[].abs
        titanic_copy = titanic_vd.copy()
        titanic_copy["fare"].normalize()
        titanic_copy["fare"].abs()

        assert titanic_copy["fare"].min() == pytest.approx(0.001082821)

    def test_vDF_apply(self, titanic_vd):
        ### Testing vDataFrame.apply
        titanic_copy = titanic_vd.copy()

        titanic_copy.apply(func = {"boat": "DECODE({}, NULL, 0, 1)",
                                   "age" : "COALESCE(age, AVG({}) OVER (PARTITION BY pclass, sex))",
                                   "name": "REGEXP_SUBSTR({}, ' ([A-Za-z])+\.')"})

        assert titanic_copy["boat"].sum() == titanic_vd["boat"].count()
        assert titanic_copy["age"].std() == pytest.approx(13.234162542)
        assert len(titanic_copy["name"].distinct()) == 16

        ### Testing vDataFrame[].apply
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply(func = "POWER({}, 2)")
        assert titanic_copy["age"].min() == pytest.approx(0.1089)

        # expected exception
        with pytest.raises(errors.QueryError) as exception_info:
            titanic_copy["age"].apply(func = "POWER({}, 2)", copy = True)
        # checking the error message
        assert exception_info.match("The parameter 'name' must not be empty")

        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply(func = "POWER({}, 2)", copy = True, copy_name = "age_pow_2")
        assert titanic_copy["age_pow_2"].min() == pytest.approx(0.1089)

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_apply_fun(self):
        pass

    def test_vDF_applymap(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy.applymap(func = "COALESCE({}, 0)", numeric_only = True)

        assert titanic_copy["age"].count() == 1234

    def test_vDF_data_part(self, smart_meters_vd):
        smart_meters_copy = smart_meters_vd.copy()
        smart_meters_copy["time"].date_part("hour")

        assert len(smart_meters_copy["time"].distinct()) == 24

    def test_vDF_round(self, smart_meters_vd):
        smart_meters_copy = smart_meters_vd.copy()
        smart_meters_copy["val"].round(n = 1)

        assert smart_meters_copy["val"].mode() == '0.1000000'

    def test_vDF_slice(self, smart_meters_vd):
        # start = True
        smart_meters_copy = smart_meters_vd.copy()
        smart_meters_copy["time"].slice(length = 1, unit = "hour")

        assert smart_meters_copy["time"].min() == datetime.datetime(2014, 1, 1, 1, 0)

        # start = False
        smart_meters_copy = smart_meters_vd.copy()
        smart_meters_copy["time"].slice(length = 1, unit = "hour", start = False)

        assert smart_meters_copy["time"].min() == datetime.datetime(2014, 1, 1, 2, 0)

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_regexp(self):
        pass

    def test_vDF_str_contains(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_contains(pat = " ([A-Za-z])+\.")

        assert titanic_copy["name"].dtype() == 'boolean'

    def test_vDF_count(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_count(pat = " ([A-Za-z])+\.")
        
        assert titanic_copy["name"].distinct() == [1, 2]

    def test_vDF_str_extract(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_extract(pat = " ([A-Za-z])+\.")
        
        assert len(titanic_copy["name"].distinct()) == 16

    def test_vDF_str_replace(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_replace(to_replace = " ([A-Za-z])+\.", value = "VERTICAPY")
        
        assert 'VERTICAPY' in titanic_copy["name"][0]

    def test_vDF_str_slice(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_slice(start = 0, step = 3)
        
        assert len(titanic_copy["name"].distinct()) == 165

    def test_vDF_add(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].add(2)

        assert titanic_copy["age"].mean() == pytest.approx(titanic_vd["age"].mean() + 2)

    def test_vDF_div(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].div(2)

        assert titanic_copy["age"].mean() == pytest.approx(titanic_vd["age"].mean() / 2)

    def test_vDF_mul(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].mul(2)

        assert titanic_copy["age"].mean() == pytest.approx(titanic_vd["age"].mean() * 2)

    def test_vDF_sub(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].sub(2)

        assert titanic_copy["age"].mean() == pytest.approx(titanic_vd["age"].mean() - 2)

    def test_vDF_add_copy(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].add_copy(name = "copy_age")

        assert titanic_copy["copy_age"].mean() == titanic_copy["age"].mean()

    def test_vDF_copy(self, titanic_vd):
        titanic_copy = titanic_vd.copy()

        assert titanic_copy.get_columns() == titanic_vd.get_columns()

