"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
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
# Standard Python Modules
import sqlite3

# Versioning
from packaging.version import Version

# Pytest
import pytest

# Other Modules
import numpy as np

# VerticaPy
from verticapy import drop, set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_winequality
from verticapy.machine_learning.memmodel.decomposition import PCA as MemPCA
from verticapy.machine_learning.vertica.decomposition import MCA, PCA
from verticapy.machine_learning.vertica.model_management import load_model

# Matplotlib skip
import matplotlib

# Get the current matplotlib version
matplotlib_version = Version(matplotlib.__version__)
skip_plt = pytest.mark.skipif(
    matplotlib_version > Version("3.5.2"),
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd():
    winequality = load_winequality()
    yield winequality
    drop(
        name="public.winequality",
    )


@pytest.fixture(scope="module")
def model(winequality_vd):
    model_class = PCA(
        "pca_model_test",
    )
    model_class.drop()
    model_class.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
    yield model_class
    model_class.drop()


class TestPCA:
    def test_repr(self, model):
        assert model.__repr__() == "<PCA>"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_PCA("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'pca_model_test\', match_by_pos = \'true\', cutoff = 1)'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_deployInverseSQL(self, model):
        expected_sql = 'APPLY_INVERSE_PCA("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'pca_model_test\', match_by_pos = \'true\')'
        result_sql = model.deployInverseSQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS pca_model_test_drop")
        model_test = PCA(
            "pca_model_test_drop",
        )
        model_test.fit("public.winequality", ["alcohol", "quality"])

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'pca_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "pca_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'pca_model_test_drop'"
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
            "index, PC1, PC2, PC3",
            "counter_name, counter_value",
            "call_string",
        ]
        assert m_att["#_of_rows"] == [3, 3, 3, 3, 1]

        m_att_details = model.get_vertica_attributes(attr_name="principal_components")

        assert m_att_details["PC1"][0] == pytest.approx(0.00430584055130197, abs=1e-6)
        assert m_att_details["PC1"][1] == pytest.approx(0.995483456627961, abs=1e-6)
        assert m_att_details["PC1"][2] == pytest.approx(-0.0948374784417728, abs=1e-6)
        assert m_att_details["PC2"][0] == pytest.approx(0.00623540848865928, abs=1e-6)
        assert m_att_details["PC2"][1] == pytest.approx(0.0948097859779201, abs=1e-6)
        assert m_att_details["PC2"][2] == pytest.approx(0.995475878243064, abs=1e-6)
        assert m_att_details["PC3"][0] == pytest.approx(0.999971289296911, abs=1e-6)
        assert m_att_details["PC3"][1] == pytest.approx(-0.0048777108225008, abs=1e-6)
        assert m_att_details["PC3"][2] == pytest.approx(-0.00579901017465387, abs=1e-6)

    def test_get_params(self, model):
        assert model.get_params() == {
            "method": "lapack",
            "n_components": 0,
            "scale": False,
        }

    @skip_plt
    def test_plot(self, model, load_matplotlib):
        result = model.plot()
        assert len(result.get_default_bbox_extra_artists()) == 8
        result = model.plot(dimensions=(2, 3))
        assert len(result.get_default_bbox_extra_artists()) == 8

    @skip_plt
    def test_plot_scree(self, model, load_matplotlib):
        result = model.plot_scree()
        assert len(result.get_default_bbox_extra_artists()) == 15

    @skip_plt
    def test_plot_circle(self, model, load_matplotlib):
        result = model.plot_circle()
        assert len(result.get_default_bbox_extra_artists()) == 16
        result = model.plot_circle(dimensions=(2, 3))
        assert len(result.get_default_bbox_extra_artists()) == 16

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT APPLY_PCA(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction == pytest.approx(model.to_python()([[3.0, 11.0, 93.0]])[0])

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT APPLY_PCA(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.model_name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                ", ".join(model.to_sql())
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)

    def test_to_memmodel(self, model):
        current_cursor().execute(
            "SELECT APPLY_PCA(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.model_name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                ", ".join(
                    model.to_memmodel().transform_sql(
                        ["citric_acid", "residual_sugar", "alcohol"]
                    )
                )
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)
        prediction3 = model.to_memmodel().transform([[3.0, 11.0, 93.0]])
        assert prediction == pytest.approx(list(prediction3[0]))

    def test_get_transform(self, winequality_vd, model):
        winequality_trans = model.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["col1"].mean() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["col2"].mean() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["col3"].mean() == pytest.approx(0.0, abs=1e-6)

    def test_get_inverse_transform(self, winequality_vd, model):
        winequality_trans = model.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        winequality_trans = model.inverse_transform(
            winequality_trans, X=["col1", "col2", "col3"]
        )
        assert winequality_trans["citric_acid"].mean() == pytest.approx(
            winequality_vd["citric_acid"].mean(), abs=1e-6
        )
        assert winequality_trans["residual_sugar"].mean() == pytest.approx(
            winequality_vd["residual_sugar"].mean(), abs=1e-6
        )
        assert winequality_trans["alcohol"].mean() == pytest.approx(
            winequality_vd["alcohol"].mean(), abs=1e-6
        )

    def test_pca_score(self, model):
        result = model.score()
        assert result["Score"][0] == pytest.approx(0.0, abs=1e-6)
        assert result["Score"][1] == pytest.approx(0.0, abs=1e-6)
        assert result["Score"][2] == pytest.approx(0.0, abs=1e-6)

    def test_set_params(self, model):
        model.set_params({"n_components": 3})
        assert model.get_params()["n_components"] == 3

    def test_model_from_vDF(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS pca_vDF")
        model_test = PCA(
            "pca_vDF",
        )
        model_test.fit(winequality_vd, ["alcohol", "quality"])
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'pca_vDF'"
        )
        assert current_cursor().fetchone()[0] == "pca_vDF"
        model_test.drop()

    def test_optional_name(self):
        model = PCA()
        assert model.model_name is not None


ROTATION_X = ["x1", "x2", "x3", "x4"]


def _sqlite_select(sql, data, columns):
    conn = sqlite3.connect(":memory:")
    col_defs = ", ".join(f'"{col}" REAL' for col in columns)
    conn.execute(f"CREATE TABLE t ({col_defs})")
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(f"INSERT INTO t VALUES ({placeholders})", data)
    return conn.execute(f"SELECT {sql} FROM t ORDER BY rowid").fetchall()


@pytest.fixture(params=[False, True], ids=["scale_false", "scale_true"])
def offline_pca(request, monkeypatch):
    """
    Fitted PCA built without a database: the attributes
    mimic what Vertica returns for the generated data.
    """
    scale = request.param
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2)) @ rng.normal(size=(2, 4))
    data = (data + 0.3 * rng.normal(size=(100, 4))) * [1, 10, 0.1, 5]
    data = data + [3, -20, 0.5, 100]
    mean, sd = data.mean(axis=0), data.std(axis=0, ddof=1)
    Z = (data - mean) / sd if scale else data - mean
    eigval, eigvec = np.linalg.eigh(np.cov(Z, rowvar=False))
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    model = PCA.__new__(PCA)
    model.model_name = "pca_rotation_offline"
    model.parameters = {"n_components": 2, "scale": scale, "method": "lapack"}
    model.X = [f'"{x}"' for x in ROTATION_X]
    model.principal_components_ = eigvec[:, :2]
    model.mean_ = mean
    model.explained_variance_ = eigval / eigval.sum()
    model.cos2_ = eigvec[:, :2] ** 2 / np.sum(eigvec[:, :2] ** 2, axis=1)[:, None]
    model._unrotated_attributes = None
    # mimics the 'columns' model attribute (index, name, mean, sd)
    columns = {"name": list(ROTATION_X), "mean": list(mean), "sd": list(sd)}
    monkeypatch.setattr(model, "get_vertica_attributes", lambda attr_name=None: columns)
    return model, data, Z


class TestPCARotation:
    """
    Rotation tests which do not require a database.
    """

    @pytest.mark.parametrize("gamma", [1.0, 0.0])
    def test_rotate_matches_memmodel(self, offline_pca, gamma):
        model, _, _ = offline_pca
        V0 = model.principal_components_.copy()
        expected = MemPCA(V0, model.mean_)
        expected.rotate(gamma=gamma)
        model.rotate(gamma=gamma)
        np.testing.assert_allclose(
            model.principal_components_, expected.principal_components_, atol=1e-12
        )
        # rotated components stay orthonormal and span the same space
        V = model.principal_components_
        np.testing.assert_allclose(V.T @ V, np.eye(2), atol=1e-10)
        R = V0.T @ V
        np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(V0 @ R, V, atol=1e-10)
        assert not np.allclose(V, V0)
        assert model._is_rotated

    def test_rotated_to_memmodel_to_sql_to_python(self, offline_pca):
        model, data, Z = offline_pca
        model.rotate()
        if model.parameters["scale"]:
            for method in (model.to_memmodel, model.to_python, model.to_sql):
                with pytest.raises(NotImplementedError, match="scale=True"):
                    method()
            return
        expected = Z @ model.principal_components_
        np.testing.assert_allclose(
            model.to_memmodel().principal_components_, model.principal_components_
        )
        np.testing.assert_allclose(model.to_python()(data), expected, rtol=1e-9)
        result = _sqlite_select(", ".join(model.to_sql()), data.tolist(), ROTATION_X)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-9)

    def test_rotate_quartimax(self, offline_pca):
        model, _, _ = offline_pca
        V0 = model.principal_components_.copy()
        quartimax = MemPCA.matrix_rotation(V0, gamma=0.0, tol=np.float32(1e-6))
        # NumPy scalars are accepted
        model.rotate(gamma=np.float32(0.0), tol=np.float32(1e-6))
        np.testing.assert_allclose(model.principal_components_, quartimax, atol=1e-12)

    def test_rotate_all_components(self, offline_pca):
        # model trained with n_components = 0 (all the components)
        model, data, Z = offline_pca
        eigval, eigvec = np.linalg.eigh(np.cov(Z, rowvar=False))
        model.parameters["n_components"] = 0
        model.principal_components_ = eigvec[:, ::-1]
        model.explained_variance_ = eigval[::-1] / eigval.sum()
        model.rotate()
        assert model.principal_components_.shape == (4, 4)
        assert model.explained_variance_.sum() == pytest.approx(1.0)
        sql = model.deploySQL()
        assert sql.count(" AS col") == 4
        result = np.array(_sqlite_select(sql, data.tolist(), ROTATION_X))
        np.testing.assert_allclose(result, Z @ model.principal_components_, rtol=1e-9)

    def test_rotate_attributes(self, offline_pca):
        model, _, Z = offline_pca
        ev0 = model.explained_variance_.copy()
        mean0 = model.mean_.copy()
        model.rotate()
        scores = Z @ model.principal_components_
        total_variance = np.trace(np.cov(Z, rowvar=False))
        np.testing.assert_allclose(
            model.explained_variance_[:2],
            np.var(scores, axis=0, ddof=1) / total_variance,
            rtol=1e-9,
        )
        np.testing.assert_allclose(model.explained_variance_[2:], ev0[2:])
        # the retained variance is redistributed, not changed
        assert model.explained_variance_[:2].sum() == pytest.approx(ev0[:2].sum())
        assert model.explained_variance_.sum() == pytest.approx(ev0.sum())
        np.testing.assert_allclose(model.cos2_.sum(axis=1), np.ones(4))
        np.testing.assert_allclose(
            model.cos2_[:, 0],
            model.principal_components_[:, 0] ** 2
            / np.sum(model.principal_components_**2, axis=1),
        )
        np.testing.assert_array_equal(model.mean_, mean0)

    def test_rotate_repeated(self, offline_pca):
        model, _, Z = offline_pca
        V0 = model.principal_components_.copy()
        expected = MemPCA(V0, model.mean_)
        expected.rotate(gamma=0.0)
        expected.rotate(gamma=1.0)
        model.rotate(gamma=0.0)
        model.rotate(gamma=1.0)
        np.testing.assert_allclose(
            model.principal_components_, expected.principal_components_, atol=1e-12
        )
        np.testing.assert_array_equal(
            model._unrotated_attributes["principal_components"], V0
        )
        scores = Z @ model.principal_components_
        np.testing.assert_allclose(
            model.explained_variance_[:2],
            np.var(scores, axis=0, ddof=1) / np.trace(np.cov(Z, rowvar=False)),
            rtol=1e-9,
        )

    def test_rotated_deploysql_matches_numpy(self, offline_pca):
        model, data, Z = offline_pca
        model.rotate()
        sql = model.deploySQL()
        assert "APPLY_PCA" not in sql
        assert sql.count(" AS col") == 2
        result = np.array(_sqlite_select(sql, data.tolist(), ROTATION_X))
        np.testing.assert_allclose(
            result, Z @ model.principal_components_, rtol=1e-9, atol=1e-9
        )
        if not model.parameters["scale"]:
            np.testing.assert_allclose(
                result, model.to_memmodel().transform(data), rtol=1e-9, atol=1e-9
            )

    def test_rotated_deploysql_options(self, offline_pca):
        model, data, Z = offline_pca
        model.rotate()
        columns = ["id"] + ROTATION_X
        rows = [[i] + row for i, row in enumerate(data.tolist())]
        rows[0][2] = None
        # same arguments as Decomposition.transform
        sql = model.deploySQL(
            X=columns, n_components=1, key_columns=["id"], exclude_columns=["id"]
        )
        assert sql.startswith('"id", ')
        assert " AS col1" in sql and " AS col2" not in sql
        result = _sqlite_select(sql, rows, columns)
        assert [r[0] for r in result] == list(range(100))
        assert result[0][1] is None
        expected = Z @ model.principal_components_[:, :1]
        np.testing.assert_allclose(
            np.array([r[1] for r in result[1:]]), expected[1:, 0], rtol=1e-9
        )
        # exclusion is case insensitive and quoted names are accepted
        sql_2 = model.deploySQL(
            X=['"ID"'] + ROTATION_X, n_components=1, exclude_columns=["id"]
        )
        assert sql_2 == sql.replace('"id", ', "", 1)
        assert model.deploySQL(n_components=2) == model.deploySQL()
        # mixed-case identifiers are quoted and mapped by position
        mixed = ['"Key Col"', '"X1"', "X2", '"x3"', "x4"]
        sql_3 = model.deploySQL(
            X=mixed, key_columns=["Key Col"], exclude_columns=['"KEY COL"']
        )
        assert sql_3.startswith('"Key Col", ') and '("X1" - ' in sql_3
        assert '("X2" - ' in sql_3
        result = _sqlite_select(sql_3, rows, ["Key Col"] + ROTATION_X)
        np.testing.assert_allclose(
            np.array([r[1:] for r in result[1:]]),
            (Z @ model.principal_components_)[1:],
            rtol=1e-9,
        )

    def test_rotated_deploysql_errors(self, offline_pca):
        model, _, _ = offline_pca
        model.rotate()
        with pytest.raises(ValueError, match="cutoff"):
            model.deploySQL(cutoff=0.9)
        with pytest.raises(ValueError, match="n_components"):
            model.deploySQL(n_components=3)
        for n_components in (-1, 1.5, True):
            with pytest.raises(ValueError, match="n_components"):
                model.deploySQL(n_components=n_components)
        with pytest.raises(ValueError, match="number of input columns"):
            model.deploySQL(X=ROTATION_X[:3])
        with pytest.raises(NotImplementedError):
            model.deployInverseSQL()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gamma": 1.5},
            {"gamma": -0.1},
            {"gamma": "varimax"},
            {"q": 0},
            {"q": 2.5},
            {"tol": -1e-6},
            {"tol": float("nan")},
        ],
    )
    def test_rotate_invalid_parameters(self, offline_pca, kwargs):
        model, _, _ = offline_pca
        V0 = model.principal_components_.copy()
        with pytest.raises((TypeError, ValueError)):
            model.rotate(**kwargs)
        assert not model._is_rotated
        np.testing.assert_array_equal(model.principal_components_, V0)

    def test_unrotated_deploysql_unchanged(self, offline_pca, monkeypatch):
        model, _, _ = offline_pca

        def fail(*args, **kwargs):
            raise AssertionError("No database call expected.")

        monkeypatch.setattr(model, "get_vertica_attributes", fail)
        assert not model._is_rotated
        assert model.deploySQL() == (
            'APPLY_PCA("x1", "x2", "x3", "x4" USING PARAMETERS model_name = '
            "'pca_rotation_offline', match_by_pos = 'true', cutoff = 1)"
        )
        assert model.deploySQL(n_components=1, cutoff=0.5).endswith(
            "num_components = 1)"
        )
        assert model.deployInverseSQL().startswith("APPLY_INVERSE_PCA(")
        np.testing.assert_array_equal(
            model.to_memmodel().principal_components_, model.principal_components_
        )

    def test_rotate_failure_keeps_model_unrotated(self, offline_pca, monkeypatch):
        model, _, _ = offline_pca
        V0 = model.principal_components_.copy()

        def fail(*args, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

        monkeypatch.setattr(MemPCA, "rotate", fail)
        with pytest.raises(np.linalg.LinAlgError):
            model.rotate()
        assert not model._is_rotated
        np.testing.assert_array_equal(model.principal_components_, V0)
        assert model.deploySQL().startswith("APPLY_PCA(")

    @pytest.mark.parametrize("offline_pca", [True], indirect=True)
    @pytest.mark.parametrize(
        "field, value",
        [
            ("sd", 0.0),
            ("sd", None),
            ("sd", float("nan")),
            ("sd", float("inf")),
            ("name", "x4"),
        ],
    )
    def test_rotate_invalid_sd(self, offline_pca, field, value):
        model, _, _ = offline_pca
        columns = model.get_vertica_attributes("columns")
        columns[field][0] = value
        with pytest.raises(ValueError):
            model.rotate()
        assert not model._is_rotated

    @pytest.mark.parametrize("offline_pca", [True], indirect=True)
    def test_rotate_sd_mixed_case_names(self, offline_pca):
        model, _, _ = offline_pca
        model.X = ['"X1"', "x2", '"X3"', "X4"]
        model.rotate()
        np.testing.assert_allclose(
            model._unrotated_attributes["sd"],
            model.get_vertica_attributes("columns")["sd"],
        )

    def test_rotate_not_fitted(self):
        model = PCA.__new__(PCA)
        model.parameters = {"n_components": 2, "scale": False, "method": "lapack"}
        with pytest.raises(AttributeError):
            model.rotate()

    def test_rotate_mca_not_supported(self):
        model = MCA.__new__(MCA)
        with pytest.raises(NotImplementedError):
            model.rotate()


ROTATION_DB_X = ["citric_acid", "residual_sugar", "alcohol"]


@pytest.fixture(params=[False, True], ids=["scale_false", "scale_true"])
def rotation_model(request, winequality_vd):
    model_class = PCA(
        f"pca_rotation_test_{str(request.param).lower()}",
        n_components=2,
        scale=request.param,
    )
    model_class.drop()
    model_class.fit("public.winequality", ROTATION_DB_X)
    yield model_class
    model_class.drop()


class TestPCARotationVertica:
    """
    Rotation tests executed against Vertica.
    """

    @pytest.mark.parametrize("gamma", [1.0, 0.0])
    def test_rotate_matches_memmodel(self, rotation_model, gamma):
        V0 = rotation_model.get_vertica_attributes("principal_components").to_numpy()
        expected = MemPCA(V0, rotation_model.mean_)
        expected.rotate(gamma=gamma)
        rotation_model.rotate(gamma=gamma)
        np.testing.assert_allclose(
            rotation_model.principal_components_,
            expected.principal_components_,
            atol=1e-10,
        )

    def test_rotated_sql_matches_apply_pca(self, rotation_model):
        V0 = rotation_model.principal_components_.copy()
        rotation_model.rotate()
        R = V0.T @ rotation_model.principal_components_
        X = ", ".join(f'"{x}"' for x in ROTATION_DB_X)
        query = f"""
            SELECT p1, p2, {rotation_model.deploySQL()}
            FROM (
                SELECT col1 AS p1, col2 AS p2, {X}
                FROM (
                    SELECT APPLY_PCA({X} USING PARAMETERS
                        model_name = '{rotation_model.model_name}',
                        key_columns = '{X}',
                        num_components = 2) OVER ()
                    FROM public.winequality) a) b"""
        current_cursor().execute(query)
        result = np.array(current_cursor().fetchall(), dtype=float)
        assert len(result) > 0
        np.testing.assert_allclose(result[:, 2:], result[:, :2] @ R, atol=1e-8)

    def test_rotated_transform(self, rotation_model, winequality_vd):
        rotation_model.rotate()
        result = rotation_model.transform(winequality_vd, X=ROTATION_DB_X)
        assert "col1" in result.get_columns()[-2]
        assert "col2" in result.get_columns()[-1]
        assert len(result.get_columns()) == len(winequality_vd.get_columns()) - 1
        assert result["col1"].avg() == pytest.approx(0.0, abs=1e-6)
        assert result["col2"].avg() == pytest.approx(0.0, abs=1e-6)
        ev = rotation_model.explained_variance_
        assert result["col1"].var() / result["col2"].var() == pytest.approx(
            ev[0] / ev[1], rel=1e-6
        )
        result_1 = rotation_model.transform(
            winequality_vd, X=ROTATION_DB_X, n_components=1
        )
        assert "col2" not in "".join(result_1.get_columns())

    def test_score_unchanged_after_rotation(self, rotation_model, winequality_vd):
        score = rotation_model.score()["Score"]
        rotation_model.rotate()
        assert rotation_model.score()["Score"] == pytest.approx(score)
        transformed = rotation_model.transform(winequality_vd, X=ROTATION_DB_X)
        with pytest.raises(NotImplementedError):
            rotation_model.inverse_transform(transformed)

    def test_rotated_to_sql(self, rotation_model):
        rotation_model.rotate()
        if rotation_model.parameters["scale"]:
            with pytest.raises(NotImplementedError):
                rotation_model.to_sql()
            return
        query = f"""
            SELECT {rotation_model.deploySQL()}, {', '.join(rotation_model.to_sql())}
            FROM public.winequality"""
        current_cursor().execute(query)
        result = np.array(current_cursor().fetchall(), dtype=float)
        np.testing.assert_allclose(result[:, :2], result[:, 2:])

    def test_refit_and_reload_reset_rotation(self, rotation_model):
        V0 = rotation_model.principal_components_.copy()
        rotation_model.rotate()
        # refitting an existing model requires overwrite_model = True;
        # a failed refit keeps the current rotation.
        with pytest.raises(NameError):
            rotation_model.fit("public.winequality", ROTATION_DB_X)
        assert rotation_model._is_rotated
        rotation_model.overwrite_model = True
        rotation_model.fit("public.winequality", ROTATION_DB_X)
        assert not rotation_model._is_rotated
        np.testing.assert_allclose(rotation_model.principal_components_, V0)
        assert rotation_model.deploySQL().startswith("APPLY_PCA(")
        rotation_model.rotate()
        loaded = load_model(rotation_model.model_name)
        assert not loaded._is_rotated
        np.testing.assert_allclose(loaded.principal_components_, V0)
