import datetime
from tqdm import tqdm
from datetime import date
import pandas as pd
import numpy as np
from plotly import __version__

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import iplot

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


import random

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    roc_curve,
)
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

tqdm.pandas()

from collections import defaultdict

# Installing specific version of plotly to avoid Invalid property for color error in recent version which needs change in layout
import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt
from matplotlib import pyplot
import plotly.graph_objs as go
import chart_studio.plotly as py


def configure_plotly_browser_state():
    import IPython

    display(
        IPython.core.display.HTML(
            """
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        """
        )
    )


def detect_classify_anomalies(df, pred_name="WL_hat_autoencoder_lstm", window=7):
    df["error"] = df["Value"] - df[pred_name]
    df["percentage_change"] = ((df["Value"] - df[pred_name]) / df["Value"]) * 100
    df["meanval"] = df["error"].rolling(window=window).mean()
    df["deviation"] = df["error"].rolling(window=window).std()
    df["-3s"] = df["meanval"] - (2 * df["deviation"])
    df["3s"] = df["meanval"] + (2 * df["deviation"])
    df["-2s"] = df["meanval"] - (1.75 * df["deviation"])
    df["2s"] = df["meanval"] + (1.75 * df["deviation"])
    df["-1s"] = df["meanval"] - (1.5 * df["deviation"])
    df["1s"] = df["meanval"] + (1.5 * df["deviation"])
    cut_list = df[["error", "-3s", "-2s", "-1s", "meanval", "1s", "2s", "3s"]]
    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    df["impact"] = [
        (lambda x: np.where(cut_sort == df["error"][x])[1][0])(x)
        for x in range(len(df["error"]))
    ]
    severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
    region = {
        0: "NEGATIVE",
        1: "NEGATIVE",
        2: "NEGATIVE",
        3: "NEGATIVE",
        4: "POSITIVE",
        5: "POSITIVE",
        6: "POSITIVE",
        7: "POSITIVE",
    }
    df["color"] = df["impact"].map(severity)
    df["region"] = df["impact"].map(region)
    df["anomaly_points"] = np.where(df["color"] == 3, df["error"], np.nan)

    return df


def plot_anomaly(df, metric_name, pred_name):

    dates = df.index

    bool_array = abs(df["anomaly_points"]) > 0
    # And a subplot of the Actual Values.
    actuals = df["Value"][-len(bool_array) :]
    anomaly_points = bool_array * actuals
    anomaly_points[anomaly_points == 0] = np.nan
    # Order_results['meanval']=meanval
    # Order_results['deviation']=deviation
    color_map = {0: "'rgba(228, 222, 249, 0.65)'", 1: "yellow", 2: "orange", 3: "red"}
    table = go.Table(
        domain=dict(x=[0, 1], y=[0, 0.3]),
        columnwidth=[1, 2],
        # columnorder=[0, 1, 2,],
        header=dict(
            height=20,
            values=[
                ["<b>Date</b>"],
                ["<b>Actual Values </b>"],
                ["<b>Predicted</b>"],
                ["<b>% Difference</b>"],
                ["<b>Severity (0-3)</b>"],
            ],
            font=dict(color=["rgb(45, 45, 45)"] * 5, size=14),
            fill=dict(color="#d562be"),
        ),
        cells=dict(
            values=[
                df.round(3)[k].tolist()
                for k in ["Date", "Value", pred_name, "percentage_change", "color"]
            ],
            line=dict(color="#506784"),
            align=["center"] * 5,
            font=dict(color=["rgb(40, 40, 40)"] * 5, size=12),
            suffix=[None] + [""] + [""] + ["%"] + [""],
            height=27,
        ),
    )
    # df['ano'] = np.where(df['color']==3, df['error'], np.nan)
    anomalies = go.Scatter(
        name="Anomaly",
        x=dates,
        xaxis="x1",
        yaxis="y1",
        y=df["anomaly_points"],
        mode="markers",
        marker=dict(color="red", size=11, line=dict(color="red", width=2)),
    )
    upper_bound = go.Scatter(
        hoverinfo="skip",
        x=dates,
        showlegend=False,
        xaxis="x1",
        yaxis="y1",
        y=df["3s"],
        marker=dict(color="#444"),
        line=dict(color=("rgb(23, 96, 167)"), width=2, dash="dash"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )
    lower_bound = go.Scatter(
        name="Confidence Interval",
        x=dates,
        xaxis="x1",
        yaxis="y1",
        y=df["-3s"],
        marker=dict(color="#444"),
        line=dict(color=("rgb(23, 96, 167)"), width=2, dash="dash"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )
    Actuals = go.Scatter(
        name="Actuals",
        x=dates,
        y=df["Value"],
        xaxis="x2",
        yaxis="y2",
        marker=dict(size=12, color="blue"),
    )
    Predicted = go.Scatter(
        name="Predicted",
        x=dates,
        y=df[pred_name],
        xaxis="x2",
        yaxis="y2",
        marker=dict(size=12, line=dict(width=1), color="orange"),
    )
    # create plot for error...
    Error = go.Scatter(
        name="Error",
        x=dates,
        y=df["error"],
        xaxis="x1",
        yaxis="y1",
        marker=dict(size=12, line=dict(width=1), color="red"),
        text="Error",
    )
    anomalies_map = go.Scatter(
        name="anomaly actual",
        showlegend=False,
        x=dates,
        y=anomaly_points,
        mode="markers",
        xaxis="x2",
        yaxis="y2",
        marker=dict(color="red", size=11, line=dict(color="red", width=2)),
    )
    Mvingavrg = go.Scatter(
        name="Moving Average",
        x=dates,
        y=df["meanval"],
        xaxis="x1",
        yaxis="y1",
        marker=dict(size=12, line=dict(width=1), color="green"),
        text="Moving average",
    )
    axis = dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        ticklen=4,
        gridcolor="#ffffff",
        tickfont=dict(size=10),
    )
    layout = dict(
        width=1000,
        height=865,
        autosize=False,
        title=metric_name,
        margin=dict(t=75),
        showlegend=True,
        xaxis1=dict(axis, **dict(domain=[0, 1], anchor="y1", showticklabels=True)),
        xaxis2=dict(axis, **dict(domain=[0, 1], anchor="y2", showticklabels=True)),
        yaxis1=dict(
            axis,
            **dict(domain=[2 * 0.21 + 0.20 + 0.09, 1], anchor="x1", hoverformat=".2f")
        ),
        yaxis2=dict(
            axis,
            **dict(
                domain=[0.21 + 0.12, 2 * 0.31 + 0.02], anchor="x2", hoverformat=".2f"
            )
        ),
    )
    fig = go.Figure(
        data=[
            table,
            anomalies,
            anomalies_map,
            upper_bound,
            lower_bound,
            Actuals,
            Predicted,
            Mvingavrg,
            Error,
        ],
        layout=layout,
    )
    iplot(fig)
    pyplot.show()
    plt.show()
