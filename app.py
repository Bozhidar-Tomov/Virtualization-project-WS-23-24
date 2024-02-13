from flask import Flask, render_template, request
from algorithms import monte_carlo_simulation, plot_monte_carlo, fit_regression, plot_regression
from algorithms import perform_lda, plot_lda, perform_pca, plot_pca, elbow_plot, kmeans_clustering, plot_kmeans
from datasets import load_dataset, preprocess_digits, preprocess_iris, preprocess_wine

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


from sklearn.linear_model import LinearRegression 
@app.route('/linregression', methods=['GET', 'POST'])
def linregression_route():
    if request.method == 'POST':
        dataset_choice = request.form['dataset']

        X, y = preprocess_iris('linear_regression') if dataset_choice == 'iris' \
                else preprocess_digits('linear_regression') if dataset_choice == 'digits' \
                else preprocess_wine('linear_regression')

        model = LinearRegression()

        coefficients, intercept, mse, r2, predictions = fit_regression(model, X, y)

        plot_html = plot_regression(X, y, predictions, "Linear Regression Result", "Predictor Feature", "Target Feature")


        return render_template('linregression_results.html', coefficients=coefficients, intercept=intercept, mse=mse, r2=r2, plot_html=plot_html)

    return render_template('linregression_form.html')



from sklearn.linear_model import Ridge

@app.route('/ridge', methods=['GET', 'POST'])
def ridge_route():
    if request.method == 'POST':
        dataset_choice = request.form['dataset']

        X, y = preprocess_iris('ridge') if dataset_choice == 'iris' \
                else preprocess_digits('ridge') if dataset_choice == 'digits' \
                else preprocess_wine('ridge')

        alpha = request.form.get('alpha', type=float, default=1.0)  # Example: getting alpha from form, with a default value
        model = Ridge(alpha=alpha)

        coefficients, intercept, mse, r2, predictions = fit_regression(model, X, y)

        plot_html = plot_regression(X[:, 0], y, predictions, "Ridge Regression Result", "Feature", "Target")

        return render_template('ridge_results.html', coefficients=coefficients, intercept=intercept, mse=mse, r2=r2, plot_html=plot_html)

    return render_template('ridge_form.html')



from sklearn.linear_model import Lasso
@app.route('/lasso', methods=['GET', 'POST'])
def lasso_route():
    if request.method == 'POST':
        dataset_choice = request.form['dataset']

        X, y = preprocess_iris('lasso') if dataset_choice == 'iris' \
                else preprocess_digits('lasso') if dataset_choice == 'digits' \
                else preprocess_wine('lasso')

        alpha = request.form.get('alpha', type=float, default=1.0)  
        model = Lasso(alpha=alpha)

        coefficients, intercept, mse, r2, predictions = fit_regression(model, X, y)
        plot_html = plot_regression(X[:, 0], y, predictions, "Lasso Regression Result", "Feature", "Target")

        return render_template('lasso_results.html', coefficients=coefficients, intercept=intercept, mse=mse, r2=r2, plot_html=plot_html)
        
    return render_template('lasso_form.html')





from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt

@app.route('/pca', methods=['GET', 'POST'])
def pca_route():
    if request.method == 'POST':
        dataset_choice = request.form['dataset']
        n_components = request.form.get('n_components', type=int, default=2)

        X, y = preprocess_iris('pca') if dataset_choice == 'iris' \
                else preprocess_digits('pca') if dataset_choice == 'digits' \
                else preprocess_wine('pca')

        pca_df = perform_pca(X, n_components=n_components)
        pca_df['target'] = y  # Add the target column here

        plot_html = plot_pca(pca_df, title="PCA Result")

        elbow_plot_html = elbow_plot(X)

        return render_template('pca_results.html', n_components=n_components, plot_html=plot_html, elbow_plot_html=elbow_plot_html)

    return render_template('pca_form.html')



    
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

@app.route('/lda', methods=['GET', 'POST'])
def lda_route():
    if request.method == 'POST':
        dataset_choice = request.form['dataset']
        n_components = request.form.get('n_components', type=int, default=2)

        X, y = preprocess_iris('lda') if dataset_choice == 'iris' \
                else preprocess_digits('lda') if dataset_choice == 'digits' \
                else preprocess_wine('lda')

        lda_df = perform_lda(X, y, n_components=n_components)

        plot_html = plot_lda(lda_df, title="LDA Result")

        return render_template('lda_results.html', n_components=n_components, plot_html=plot_html)

    return render_template('lda_form.html')

@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_route():
    if request.method == 'POST':
        dataset_choice = request.form['dataset']
        n_clusters = request.form.get('n_clusters', type=int, default=3)

        X, _ = preprocess_iris('kmeans') if dataset_choice == 'iris' \
                else preprocess_digits('kmeans') if dataset_choice == 'digits' \
                else preprocess_wine('kmeans')

        kmeans_df = kmeans_clustering(X, n_clusters=n_clusters)

        plot_html = plot_kmeans(kmeans_df, title="KMeans Clustering Result")

        return render_template('kmeans_results.html', n_clusters=n_clusters, plot_html=plot_html)

    return render_template('kmeans_form.html')



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

@app.route('/montecarlo', methods=['GET', 'POST'])
def montecarlo_route():
    if request.method == 'POST':
        dataset_choice = request.form['dataset']
        n_simulations = request.form.get('n_simulations', type=int, default=100)

        X, y = preprocess_iris('classification') if dataset_choice == 'iris' \
                else preprocess_digits('classification') if dataset_choice == 'digits' \
                else preprocess_wine('classification')

        model = RandomForestClassifier()  

        accuracies = monte_carlo_simulation(X, y, model, n_simulations)

        plot_html = plot_monte_carlo(accuracies, n_simulations)
        return render_template('montecarlo_results.html', accuracies=accuracies, plot_html=plot_html)

    return render_template('montecarlo_form.html')





if __name__ == '__main__':
    app.run(debug=True)

@app.route('/ask_ai_form')
def show_ask_ai_form():
    return render_template('askai.html')

