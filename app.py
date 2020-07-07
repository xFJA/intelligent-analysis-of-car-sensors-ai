from flask import Flask, request, jsonify
from pca import pca
from io import StringIO
import os

app = Flask(__name__)

# Routes


@app.route('/pca', methods=['POST'])
def pca_request():

    two_first_components_plot, components_and_features_plot, wcss_plot, cumulative_explained_variance_ratio_plot,  explained_variance_ratio, cluster_list = pca(
        request.files.get("csv"))

    return jsonify(
        twoFirstComponentsPlot=two_first_components_plot,
        componentsAndFeaturesPlot=components_and_features_plot,
        wcssPlot=wcss_plot,
        cumulativeExplainedVarianceRatioPlot=cumulative_explained_variance_ratio_plot,
        explainedVarianceRatio=explained_variance_ratio,
        clusterList=cluster_list
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
