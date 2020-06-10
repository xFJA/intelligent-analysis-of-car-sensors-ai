from flask import Flask, request, jsonify 
from pca import pca
from io import StringIO
import os

app = Flask(__name__)

# Routes
@app.route('/pca', methods=['POST'])
def pca_request():
    
    two_first_components_plot, components_and_features_plot, explained_variance_ratio = pca(request.files.get("csv"))
    
    return jsonify(
        twoFirstComponentsPlot = two_first_components_plot,
        componentsAndFeaturesPlot = components_and_features_plot, 
        explainedVarianceRatio = explained_variance_ratio
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)