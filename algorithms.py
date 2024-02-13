from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

def monte_carlo_simulation(X, y, model, n_simulations):
    accuracies = []
    for _ in range(n_simulations):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Train the model
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return accuracies

def plot_monte_carlo(accuracies, n_simulations):
    hist_data = go.Figure(data=[go.Histogram(x=accuracies)])
    hist_data.update_layout(
        title=f'Monte Carlo Simulation Results (n={n_simulations})',
        xaxis_title='Accuracy',
        yaxis_title='Frequency',
        bargap=0.2
    )
    hist_data.show()



from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go

def fit_regression(model, X, Y):
    model.fit(X, Y)
    predictions = model.predict(X)
    mse = mean_squared_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    return model.coef_, model.intercept_, mse, r2, predictions

def plot_regression(X, Y, predictions, title, xaxis_title, yaxis_title):
    scatter = go.Scatter(x=X.flatten(), y=Y, mode='markers', name='Data Points')
    line = go.Scatter(x=X.flatten(), y=predictions, mode='lines', name='Regression Line')
    
    layout = go.Layout(title=title,
                       xaxis=dict(title=xaxis_title),
                       yaxis=dict(title=yaxis_title))
    
    fig = go.Figure(data=[scatter, line], layout=layout)
    fig.show()



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import plotly.express as px

def perform_lda(data, target, n_components=2):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_components = lda.fit_transform(data, target)

    lda_df = pd.DataFrame(data=lda_components, 
                          columns=[f'LDA{i+1}' for i in range(n_components)])
    lda_df['target'] = target

    return lda_df

def plot_lda(df, title="Visualization", color_sequence=None):
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], 
                     color='target', 
                     title=title,
                     color_discrete_sequence=color_sequence)
    fig.show()


from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)

    pca_df = pd.DataFrame(data=components, 
                          columns=[f'PCA{i+1}' for i in range(n_components)])
    
    return pca_df

def plot_pca(pca_df, target=None, title="PCA Visualization"):
    if target is not None:
        pca_df['target'] = target
    
    fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='target', title=title)
    fig.show()

def elbow_plot(data):
    pca = PCA()
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
    plt.title('Elbow Plot for PCA')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.grid()
    plt.show()



from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px

def kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)

    kmeans_df = pd.DataFrame(data=data, columns=[f'Feature{i+1}' for i in range(data.shape[1])])
    kmeans_df['Cluster'] = kmeans_labels

    return kmeans_df

def plot_kmeans(df, title="K-Means Clustering", color_sequence=None):
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], 
                     color='Cluster', 
                     title=title,
                     color_discrete_sequence=color_sequence)
    fig.show()










