# Pytest
import pytest

# Standard Python Modules


# Other Modules
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# VerticaPy
from vertica_highcharts.highcharts.highcharts import Highchart


def get_xaxis_label(obj):
    if isinstance(obj, plt.Axes):
        return obj.get_xlabel()
    elif isinstance(obj, go.Figure):
        return obj.layout["xaxis"]["title"]["text"]
    elif isinstance(obj, Highchart):
        return obj.options['xAxis'].title.text
    else:
        return None


def get_yaxis_label(obj):
    if isinstance(obj, plt.Axes):
        return obj.get_ylabel()
    elif isinstance(obj, go.Figure):
        return obj.layout["yaxis"]["title"]["text"]
    elif isinstance(obj, Highchart):
        return obj.options['yAxis'].title.text
    else:
        return None