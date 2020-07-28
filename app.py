from flask import Flask, request, jsonify
from pca import pca
from io import StringIO
import os
import json

app = Flask(__name__)

# Routes


@app.route('/pca', methods=['POST'])
def pca_request():

    two_first_components_plot, components_and_features_plot, wcss_plot, cumulative_explained_variance_ratio_plot,  explained_variance_ratio, cluster_list, more_important_features = pca(
        request.files.get("csv"), int(request.args.get('components-number')), int(request.args.get('clusters-number')))

    return jsonify(
        twoFirstComponentsPlot=two_first_components_plot,
        componentsAndFeaturesPlot=components_and_features_plot,
        wcssPlot=wcss_plot,
        cumulativeExplainedVarianceRatioPlot=cumulative_explained_variance_ratio_plot,
        explainedVarianceRatio=explained_variance_ratio,
        clusterList=cluster_list,
        moreImportantFeatures=json.dumps(more_important_features)
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
