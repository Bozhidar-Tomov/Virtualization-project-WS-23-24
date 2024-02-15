from sklearn.decomposition import PCA  # TODO
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px


def plot_monte_carlo(accuracies, n_simulations):
    hist_data = go.Figure(data=[go.Histogram(x=accuracies)])
    hist_data.update_layout(
        title=f"Monte Carlo Simulation Results (n={n_simulations})",
        xaxis_title="Accuracy",
        yaxis_title="Frequency",
        bargap=0.2,
    )
    hist_data.show()


def plot_regression(X, Y, predictions, title, xaxis_title, yaxis_title):
    scatter = go.Scatter(x=X, y=Y, mode="markers", name="Data Points")

    line = go.Scatter(x=X, y=predictions, mode="lines", name="Regression Line")

    layout = go.Layout(
        title=title, xaxis=dict(title=xaxis_title), yaxis=dict(title=yaxis_title)
    )

    fig = go.Figure(data=[scatter, line], layout=layout)
    fig.show()


def plot_lda(df, title="Visualization", color_sequence=None):
    fig = px.scatter(
        df,
        x=df.columns[0],
        y=df.columns[1],
        color="target",
        title=title,
        color_discrete_sequence=color_sequence,
    )
    fig.show()


def plot_pca(pca_df, target=None, title="PCA Visualization"):
    if target is not None:
        pca_df["target"] = target

    fig = px.scatter(pca_df, x="PCA1", y="PCA2", color="target", title=title)
    fig.show()


def elbow_plot(data):
    pca = PCA()
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
    plt.title("Elbow Plot for PCA")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.grid()
    plt.show()


def plot_kmeans(df, title="K-Means Clustering", color_sequence=None):
    fig = px.scatter(
        df,
        x=df.columns[0],
        y=df.columns[1],
        color="Cluster",
        title=title,
        color_discrete_sequence=color_sequence,
    )
    fig.show()


def plot_interpolation():
    pass
