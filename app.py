from flask import Flask, request, jsonify
from ai import k_means as km, svm, predict
from io import StringIO
import os
import json

app = Flask(__name__)


@app.route('/pca', methods=['POST'])
def pca_request():

    # Apply k-means to the CSV
    two_first_components_plot, components_and_features_plot, wcss_plot, cumulative_explained_variance_ratio_plot,  explained_variance_ratio, cluster_list, more_important_features, svm_params = km.start(
        request.files.get("csv"), int(request.args.get('components-number')), int(request.args.get('clusters-number')))

    # Apply SVM to the data labelled by k-means
    svm_plot = svm.start(
        svm_params['df'], svm_params['x_scaled_reduced'], 6)

    return jsonify(
        twoFirstComponentsPlot=two_first_components_plot,
        componentsAndFeaturesPlot=components_and_features_plot,
        wcssPlot=wcss_plot,
        cumulativeExplainedVarianceRatioPlot=cumulative_explained_variance_ratio_plot,
        explainedVarianceRatio=explained_variance_ratio,
        clusterList=cluster_list,
        moreImportantFeatures=json.dumps(more_important_features),
        svmPlot=svm_plot
    )


@app.route('/svm', methods=['POST'])
def svm_classification_request():
    classification_list = svm.classify_svm(request.files.get("csv"), int(
        request.args.get('dataset-rows-number')))
    return jsonify(
        classificationList=classification_list
    )


@app.route('/predict', methods=['POST'])
def predict_request():
    learning_curve_plot, prediction_plot, rmse, time = predict.predict(
        request.files.get("csv"), request.args.get('feature'), request.args.get('epochs'), request.args.get('predictions-feature-type'), int(request.args.get('principal-components-number')))

    return jsonify(
        learningCurvePlot=learning_curve_plot,
        predictionPlot=prediction_plot,
        rmse=str(rmse),
        time=str(time)
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
