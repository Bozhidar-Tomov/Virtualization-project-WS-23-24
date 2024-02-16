from flask import Flask, request, render_template
from flask_cors import CORS

import requests
import pandas as pd
import numpy as np

from plot import (
    plot_lda,
    plot_pca,
    elbow_plot,
    plot_kmeans,
    plot_monte_carlo,
    plot_regression,
)

import plotly.graph_objs as go  # TODO

PORT = 5000
HOST = "0.0.0.0"

app = Flask(__name__)
CORS(app)


# READY
@app.route("/")
def index():
    return render_template("index.html")


# READY
@app.route("/linregression", methods=["GET", "POST"])
def linregression_route():
    if request.method == "POST":
        dataset_choice = request.form["dataset"]

        response = requests.post(
            "http://math_operations_c:8001/linregression",
            data={"dataset": dataset_choice},
        )
        data = response.json()

        # Convert lists back to numpy arrays for plot_regression
        X = np.array(data["X"])
        y = np.array(data["y"])
        predictions = np.array(data["predictions"])

        plot_html = plot_regression(
            X,
            y,
            predictions,
            "Linear Regression Result",
            "Predictor Feature",
            "Target Feature",
        )

        return render_template(
            "linregression_results.html",
            coefficients=data["coefficients"],
            intercept=data["intercept"],
            mse=data["mse"],
            r2=data["r2"],
            plot_html=plot_html,
        )

    return render_template("linregression_form.html")


# READY
@app.route("/interpolation", methods=["GET", "POST"])
def interpolation_route():
    if request.method == "POST":
        data = request.get_json()

        response = requests.post(
            "http://math_operations_c:8001/interpolation", json=data
        )
        result = response.json()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=result["x_axis"],
                y=result["y_axis"],
                mode="lines",
                name="Interpolation",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["nodes_x"], y=data["values_y"], mode="markers", name="Points"
            )
        )
        fig.update_layout(title="Lagrange Interpolation Polynomial")

        fig.show()

        return render_template("interpolation_result.html", plot=fig)
    else:
        return render_template("interpolation_form.html")


# READY
@app.route("/ridge", methods=["GET", "POST"])
def ridge_route():
    if request.method == "POST":
        dataset_choice = request.form["dataset"]
        alpha = request.form.get("alpha", type=float, default=1.0)
        response = requests.post(
            "http://math_operations_c:8001/ridge",
            data={"dataset": dataset_choice, "alpha": alpha},
        )
        data = response.json()

        X = np.array(data["X"]).flatten()
        plot_html = plot_regression(
            X,
            data["y"],
            data["predictions"],
            "Ridge Regression Result",
            "Feature",
            "Target",
        )
        return render_template(
            "ridge_results.html",
            coefficients=data["coefficients"],
            intercept=data["intercept"],
            mse=data["mse"],
            r2=data["r2"],
            plot_html=plot_html,
        )
    return render_template("ridge_form.html")


# READY
@app.route("/lasso", methods=["GET", "POST"])
def lasso_route():
    if request.method == "POST":
        dataset_choice = request.form["dataset"]
        alpha = request.form.get("alpha", type=float, default=1.0)
        response = requests.post(
            "http://math_operations_c:8001/lasso",
            data={"dataset": dataset_choice, "alpha": alpha},
        )
        data = response.json()
        X = [item for sublist in data["X"] for item in sublist]  # Flatten the list
        plot_html = plot_regression(
            X,
            data["y"],
            data["predictions"],
            "Lasso Regression Result",
            "Feature",
            "Target",
        )
        return render_template(
            "lasso_results.html",
            coefficients=data["coefficients"],
            intercept=data["intercept"],
            mse=data["mse"],
            r2=data["r2"],
            plot_html=plot_html,
        )
    return render_template("lasso_form.html")


@app.route("/pca", methods=["GET", "POST"])
def pca_route():
    if request.method == "POST":
        dataset_choice = request.form["dataset"]
        n_components = request.form.get("n_components", type=int, default=2)

        response = requests.post(
            "http://math_operations_c:8001/pca",
            data={"dataset": dataset_choice, "n_components": n_components},
        )
        pca_df = response.json()

        plot_html = plot_pca(pca_df, title="PCA Result")
        elbow_plot_html = elbow_plot(pca_df["X"])

        return render_template(
            "pca_results.html",
            n_components=n_components,
            plot_html=plot_html,
            elbow_plot_html=elbow_plot_html,
        )

    return render_template("pca_form.html")


@app.route("/lda", methods=["GET", "POST"])
def lda_route():
    if request.method == "POST":
        dataset_choice = request.form["dataset"]
        n_components = request.form.get("n_components", type=int, default=2)

        response = requests.post(
            "http://math_operations_c:8001/lda",
            data={"dataset": dataset_choice, "n_components": n_components},
        )
        lda_json = response.json()
        lda_df = pd.DataFrame(lda_json)

        plot_html = plot_lda(lda_df, title="LDA Result")

        return render_template(
            "lda_results.html", n_components=n_components, plot_html=plot_html
        )

    return render_template("lda_form.html")


# READY
@app.route("/kmeans", methods=["GET", "POST"])
def kmeans_route():
    if request.method == "POST":
        dataset_choice = request.form["dataset"]
        n_clusters = request.form.get("n_clusters", type=int, default=3)

        response = requests.post(
            "http://math_operations_c:8001/kmeans",
            data={"dataset": dataset_choice, "n_clusters": n_clusters},
        )
        kmeans_json = response.json()
        kmeans_df = pd.DataFrame(kmeans_json)

        plot_html = plot_kmeans(kmeans_df, title="KMeans Clustering Result")

        return render_template(
            "kmeans_results.html", n_clusters=n_clusters, plot_html=plot_html
        )
    else:
        return render_template("kmeans_form.html")


# READY
@app.route("/montecarlo", methods=["GET", "POST"])
def montecarlo_route():
    if request.method == "POST":
        dataset_choice = request.form["dataset"]
        n_simulations = request.form.get("n_simulations", type=int, default=100)

        response = requests.post(
            "http://math_operations_c:8001/montecarlo",
            data={"dataset": dataset_choice, "n_simulations": n_simulations},
        )
        data = response.json()

        plot_html = plot_monte_carlo(data["accuracies"], n_simulations)

        return render_template(
            "montecarlo_results.html",
            accuracies=data["accuracies"],
            plot_html=plot_html,
        )

    return render_template("montecarlo_form.html")


import json


# READY
@app.route("/matrix", methods=["GET", "POST"])
def matrix_route():
    result = None
    if request.method == "POST":
        operation = request.form.get("operation")
        matrix1 = json.loads(request.form.get("matrix1"))
        matrix2 = json.loads(request.form.get("matrix2"))
        data = {"matrix1": matrix1, "matrix2": matrix2}
        response = requests.post(
            f"http://matrix_operations_c:8000/matrix/{operation}", json=data
        )
        if response.status_code == 200:
            result = response.json()
        else:
            result = {"error": response.text}
        return render_template("matrix_results.html", result=result)
    else:
        return render_template("matrix_form.html", result=result)


if __name__ == "__main__":
    app.run(port=PORT, host=HOST, debug=True)
