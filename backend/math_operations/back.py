from flask import Flask, request
from flask_cors import CORS

from sklearn.linear_model import LinearRegression
import numpy as np

from algorithms import monte_carlo_simulation, fit_regression
from algorithms import perform_lda, kmeans_clustering, Interpolation, perform_pca
from datasets import preprocess_digits, preprocess_iris, preprocess_wine

PORT = 8001
HOST = "0.0.0.0"

app = Flask(__name__)
CORS(app)


# READY
@app.route("/linregression", methods=["POST"])
def linregression_route():
    dataset_choice = request.form["dataset"]

    X, y = (
        preprocess_iris("linear_regression")
        if dataset_choice == "iris"
        else (
            preprocess_digits("linear_regression")
            if dataset_choice == "digits"
            else preprocess_wine("linear_regression")
        )
    )

    model = LinearRegression()

    coefficients, intercept, mse, r2, predictions = fit_regression(model, X, y)

    X = X.tolist() if isinstance(X, np.ndarray) else X
    y = y.tolist() if isinstance(y, np.ndarray) else y
    coefficients = (
        coefficients.tolist() if isinstance(coefficients, np.ndarray) else coefficients
    )
    intercept = intercept.tolist() if isinstance(intercept, np.ndarray) else intercept
    mse = mse.tolist() if isinstance(mse, np.ndarray) else mse
    r2 = r2.tolist() if isinstance(r2, np.ndarray) else r2
    predictions = (
        predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
    )

    return {
        "X": X,
        "y": y,
        "coefficients": coefficients,
        "intercept": intercept,
        "mse": mse,
        "r2": r2,
        "predictions": predictions,
    }


# READY
@app.route("/interpolation", methods=["POST"])
def interpolation_route():
    data = request.get_json()
    nodes = np.array(data["nodes_x"])
    values = np.array(data["values_y"])

    interpolator = Interpolation(nodes, values)

    x_axis = np.linspace(min(nodes), max(nodes), num=1000)
    y_axis = [interpolator.lagrange_poly(x) for x in x_axis]

    return {"x_axis": x_axis.tolist(), "y_axis": y_axis}


from sklearn.linear_model import Ridge


# READY
@app.route("/ridge", methods=["POST"])
def ridge_route():
    dataset_choice = request.form["dataset"]
    X, y = (
        preprocess_iris("ridge")
        if dataset_choice == "iris"
        else (
            preprocess_digits("ridge")
            if dataset_choice == "digits"
            else preprocess_wine("ridge")
        )
    )
    alpha = request.form.get("alpha", type=float, default=1.0)
    model = Ridge(alpha=alpha)

    X = X.tolist() if isinstance(X, np.ndarray) else X
    y = y.tolist() if isinstance(y, np.ndarray) else y

    coefficients, intercept, mse, r2, predictions = fit_regression(model, X, y)
    # Convert numpy arrays to lists before returning
    return {
        "X": X,
        "y": y,
        "coefficients": coefficients.tolist(),
        "intercept": intercept.tolist(),
        "mse": mse.tolist(),
        "r2": r2.tolist(),
        "predictions": predictions.tolist(),
    }


from sklearn.linear_model import Lasso


# READY
@app.route("/lasso", methods=["POST"])
def lasso_route():
    dataset_choice = request.form["dataset"]
    X, y = (
        preprocess_iris("lasso")
        if dataset_choice == "iris"
        else (
            preprocess_digits("lasso")
            if dataset_choice == "digits"
            else preprocess_wine("lasso")
        )
    )
    alpha = request.form.get("alpha", type=float, default=1.0)
    model = Lasso(alpha=alpha)
    X = X.tolist() if isinstance(X, np.ndarray) else X
    y = y.tolist() if isinstance(y, np.ndarray) else y
    coefficients, intercept, mse, r2, predictions = fit_regression(model, X, y)
    coefficients = (
        coefficients.tolist() if isinstance(coefficients, np.ndarray) else coefficients
    )
    intercept = [intercept] if isinstance(intercept, np.float64) else intercept
    mse = [mse] if isinstance(mse, np.float64) else mse
    r2 = [r2] if isinstance(r2, np.float64) else r2
    predictions = (
        predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
    )
    return {
        "X": X,
        "y": y,
        "coefficients": coefficients,
        "intercept": intercept,
        "mse": mse,
        "r2": r2,
        "predictions": predictions,
    }


@app.route("/pca", methods=["POST"])
def pca_route():
    dataset_choice = request.form["dataset"]
    n_components = request.form.get("n_components", type=int, default=2)

    X, y = (
        preprocess_iris("pca")
        if dataset_choice == "iris"
        else (
            preprocess_digits("pca")
            if dataset_choice == "digits"
            else preprocess_wine("pca")
        )
    )

    pca_df = perform_pca(X, n_components=n_components)
    pca_df["target"] = y.tolist()  # Add the target column here

    return pca_df


@app.route("/lda", methods=["POST"])
def lda_route():
    dataset_choice = request.form["dataset"]
    n_components = request.form.get("n_components", type=int, default=2)

    X, y = (
        preprocess_iris("lda")
        if dataset_choice == "iris"
        else (
            preprocess_digits("lda")
            if dataset_choice == "digits"
            else preprocess_wine("lda")
        )
    )

    lda_df = perform_lda(X, y, n_components=n_components)

    return lda_df


# READY
@app.route("/kmeans", methods=["POST"])
def kmeans_route():
    dataset_choice = request.form["dataset"]
    n_clusters = request.form.get("n_clusters", type=int, default=3)

    X, _ = (
        preprocess_iris("kmeans")
        if dataset_choice == "iris"
        else (
            preprocess_digits("kmeans")
            if dataset_choice == "digits"
            else preprocess_wine("kmeans")
        )
    )

    kmeans_df = kmeans_clustering(X, n_clusters=n_clusters)

    return kmeans_df.to_json()


from sklearn.ensemble import RandomForestClassifier


# READY
@app.route("/montecarlo", methods=["POST"])
def montecarlo_route():
    dataset_choice = request.form["dataset"]
    n_simulations = request.form.get("n_simulations", type=int, default=100)

    X, y = (
        preprocess_iris("classification")
        if dataset_choice == "iris"
        else (
            preprocess_digits("classification")
            if dataset_choice == "digits"
            else preprocess_wine("classification")
        )
    )

    model = RandomForestClassifier()

    accuracies = monte_carlo_simulation(X, y, model, n_simulations)

    return {"accuracies": accuracies}


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
