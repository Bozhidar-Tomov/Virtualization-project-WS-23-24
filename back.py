from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)
PORT = 8000

@app.route('/matrix/add', methods=['POST'])
def add_matrices():
    try:
        data = request.get_json()
        matrix1 = np.array(data['matrix1'])
        matrix2 = np.array(data['matrix2'])
        if matrix1.shape != matrix2.shape:
            return "The matrices must have the same dimensions.", 400
        result = np.add(matrix1, matrix2).tolist()
        return jsonify(result)
    except Exception as e:
        return str(e), 400

@app.route('/matrix/subtract', methods=['POST'])
def subtract_matrices():
    try:
        data = request.get_json()
        matrix1 = np.array(data['matrix1'])
        matrix2 = np.array(data['matrix2'])
        if matrix1.shape != matrix2.shape:
            return "The matrices must have the same dimensions.", 400
        result = np.subtract(matrix1, matrix2).tolist()
        return jsonify(result)
    except Exception as e:
        return str(e), 400

@app.route('/matrix/multiply', methods=['POST'])
def multiply_matrices():
    try:
        data = request.get_json()
        matrix1 = np.array(data['matrix1'])
        matrix2 = np.array(data['matrix2'])
        if matrix1.shape[1] != matrix2.shape[0]:
            return "The number of columns in the first matrix must be equal to the number of rows in the second matrix.", 400
        result = np.dot(matrix1, matrix2).tolist()
        return jsonify(result)
    except Exception as e:
        return str(e), 400

@app.route('/matrix/inverse', methods=['POST'])
def inverse_matrix():
    try:
        data = request.get_json()
        matrix = np.array(data['matrix'])
        if np.linalg.det(matrix) == 0:
            return "The matrix is not invertible.", 400
        result = np.linalg.inv(matrix).tolist()
        return jsonify(result)
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
