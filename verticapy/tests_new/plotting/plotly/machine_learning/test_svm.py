# Pytest
import pytest

# Standard Python Modules


# Other Modules


# Verticapy
from verticapy.learn.svm import LinearSVC

# Testing variables
col_name_1 = "X"
col_name_2 = "Y"
col_name_3 = "Z"
by_col = "Category"


@pytest.fixture(scope="class")
def plot_result(dummy_pred_data_vd):
    model = LinearSVC(name="public.SVC")
    model.fit(dummy_pred_data_vd, [col_name_1], by_col)
    return model.plot()


@pytest.fixture(scope="class")
def plot_result_2d(load_plotly, dummy_pred_data_vd):
    model = LinearSVC(name="public.SVC")
    model.fit(dummy_pred_data_vd, [col_name_1, col_name_2], by_col)
    return model.plot()


@pytest.fixture(scope="class")
def plot_result_3d(load_plotly, dummy_pred_data_vd):
    model = LinearSVC(name="public.SVC")
    model.fit(
        dummy_pred_data_vd,
        [col_name_1, col_name_2, col_name_3],
        by_col,
    )
    return model.plot()


class TestMachineLearningSVMClassifierPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result, plot_result_2d, plot_result_3d):
        self.result = plot_result
        self.result_2d = plot_result_2d
        self.result_3d = plot_result_3d

    def test_properties_output_type_for_1d(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotly_figure_object, "Wrong object crated"

    def test_properties_output_typefor_2d(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result_2d) == plotly_figure_object, "Wrong object crated"

    def test_properties_output_type_for_3d(self, plotly_figure_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result_3d) == plotly_figure_object, "Wrong object crated"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_no_of_elements(self):
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some elements missing"

    def test_properties_no_of_elements_for_2d(self):
        # Arrange
        total_items = 3
        # Act
        # Assert
        assert len(self.result_2d.data) == total_items, "Some elements missing"

    def test_properties_no_of_elements_for_3d(self):
        # Arrange
        total_items = 3
        # Act
        # Assert
        assert len(self.result_3d.data) == total_items, "Some elements missing"

    def test_additional_options_custom_height(self, dummy_pred_data_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        model = LinearSVC(name="public.SVC")
        model.fit(dummy_pred_data_vd, [col_name_1], by_col)
        # Act
        result = model.plot(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"

    def test_additional_options_custom_height_for_2d(
        self, load_plotly, dummy_pred_data_vd
    ):
        # rrange
        custom_height = 650
        custom_width = 700
        model = LinearSVC(name="public.SVC")
        model.fit(dummy_pred_data_vd, [col_name_1, col_name_2], by_col)
        # Act
        result = model.plot(width=custom_width, height=custom_height)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
