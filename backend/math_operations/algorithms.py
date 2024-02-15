from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


from sklearn.metrics import mean_squared_error, r2_score

def fit_regression(model, X, Y):
    model.fit(X, Y)
    predictions = model.predict(X)
    mse = mean_squared_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    return model.coef_, model.intercept_, mse, r2, predictions


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

def perform_lda(data, target, n_components=2):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_components = lda.fit_transform(data, target)

    lda_df = pd.DataFrame(data=lda_components, 
                          columns=[f'LDA{i+1}' for i in range(n_components)])
    lda_df['target'] = target

    return lda_df


from sklearn.decomposition import PCA
import pandas as pd

def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)

    pca_df = pd.DataFrame(data=components, 
                          columns=[f'PCA{i+1}' for i in range(n_components)])
    
    return pca_df


from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px

def kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)

    kmeans_df = pd.DataFrame(data=data, columns=[f'Feature{i+1}' for i in range(data.shape[1])])
    kmeans_df['Cluster'] = kmeans_labels

    return kmeans_df

class Interpolation:
    def __init__(self, nodes, values):
        self.nodes = nodes
        self.values = values
        
    def base(self, i, x):
        product = 1
        for j in range(len(self.nodes)):
            if i == j: continue
            product *= (x - self.nodes[j]) / (self.nodes[i] - self.nodes[j])
        return product
    
    def lagrange_poly(self, x):
        result = 0
        for i in range(len(self.nodes)):
            result += self.values[i] * self.base(i, x)
        return result
