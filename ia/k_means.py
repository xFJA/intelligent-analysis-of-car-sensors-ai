import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random
import base64
import io
import json
import operator
from kneed import KneeLocator
from media import image

COLOR_LIST = ['b', 'g', 'r', 'c', 'm', 'y']


def generate_pc_columns_names(number):
    res = []

    for i in range(number):
        res.append("pc"+str(i+1))

    return res


def generate_cluster_labels(number):
    res = []

    for i in range(number):
        res.append("Type "+str(i))

    return res


def generate_cluster_map(number):
    res = {}

    cluster_labels = generate_cluster_labels(number)

    for i in range(number):
        res[i] = cluster_labels[i]

    return res


def get_pca_more_important_features(df, features, pca, components_number):
    # Get identity matrix
    i = np.identity(df.shape[1])

    # Pass identity matrix to transform method to get weights (coeficients)
    coef = pca.transform(i)

    more_important_features = {}

    for j in range(components_number):
        pca_coef = {}
        cont = 0
        for pc in coef:
            pca_coef[features[cont]] = abs(pc[j])
            cont += 1

        # Calculate the percetange of each coeficient (percentage(i) = (lamba(i) * 100) / sum(lambas))
        percetage_sum = sum(pca_coef.values())
        pca_coef_percetage = {}
        for key in pca_coef:
            pca_coef_percetage[key] = (pca_coef[key] * 100) / percetage_sum

        # Order each feature coeficient percentage by it value
        pca_coef_percetage = sorted(pca_coef_percetage.items(),
                                    key=operator.itemgetter(1), reverse=True)

        # Get most important features to cover the 90% of participation
        percentage_participation = 0
        pca_most_important_features = {}

        for per in pca_coef_percetage:
            if (percentage_participation >= 90):
                break

            value = per[1]
            pca_most_important_features[per[0]] = value
            percentage_participation += value

        more_important_features[j] = pca_most_important_features

    return more_important_features


def start(csv_file, components_number, clusters_number):
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Get features from the first row
    features = df.columns.values

    # Remove features labels
    x = df.loc[:, features].values

    # Standardizing data (mean = 0 , variance = 1)
    x = StandardScaler().fit_transform(x)

    # Generate cumulative explanined variance ratio plot
    pca = PCA().fit(x)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Components number')
    plt.ylabel('Cumulative explained variance')
    cumulative_explained_variance_ratio_plot = image.get_base64(plt)
    plt.clf()

    # Create pca
    pca = PCA(n_components=components_number)

    # Fit model with the number of components selected
    pca.fit_transform(x)

    # Get PCA scores (create clusters based on the components scores)
    pca_scores = pca.transform(x)

    # Get the principal features for each principal component
    more_important_features = get_pca_more_important_features(
        df, features, pca, components_number)

    # Fit kmeans using the data from PCA
    wcss = []
    for i in range(1, 21):
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_pca.fit(pca_scores)
        wcss.append(kmeans_pca.inertia_)
    

    # Find elbow
    kneedle = KneeLocator(range(1, 21), wcss, S=1.0,
                          curve='convex', direction='decreasing')

    # Plot wcss
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, 21), wcss, marker='o', linestyle='--')
    plt.vlines(kneedle.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='m', label='Elbow')
    plt.legend()
    plt.xlabel("Clusters number")
    plt.ylabel("WCSS")
    wcss_plot = image.get_base64(plt)
    plt.clf()

    # Run k-means with the number of cluster chosen
    kmeans_pca = KMeans(n_clusters=clusters_number,
                        init='k-means++', random_state=42)

    # Fit data with the k-means pca model
    kmeans_pca.fit(pca_scores)

    # Create dataset with results from PCA and the cluster column
    df_kmeans_pca = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(pca_scores)], axis=1)
    df_kmeans_pca.columns.values[-components_number:] = generate_pc_columns_names(
        components_number)
    df_kmeans_pca['Kmeans value'] = kmeans_pca.labels_

    # Add cluster column with a label associated to each kmeans value
    df_kmeans_pca['Cluster'] = df_kmeans_pca['Kmeans value'].map(
        generate_cluster_map(clusters_number))

    # Create columns labels for each component
    pc_colums_names = generate_pc_columns_names(components_number)

    # Plot two first compontens
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal component 1', fontsize=16)
    ax.set_ylabel('Principal component 2', fontsize=16)
    ax.set_title('PCA', fontsize=22)
    targets = generate_cluster_labels(clusters_number)
    colors = COLOR_LIST
    for target, color in zip(targets, colors):
        indicesToKeep = df_kmeans_pca['Cluster'] == target
        ax.scatter(df_kmeans_pca.loc[indicesToKeep, 'pc1'],
                   df_kmeans_pca.loc[indicesToKeep, 'pc2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    two_first_components_plot = image.get_base64(plt)

    # Print the amount of data that holds the components
    explained_variance_ratio = pca.explained_variance_ratio_

    # Create a plot for the porcetage of participation of each feature in each component
    plt.matshow(pca.components_, cmap='Blues')
    plt.yticks([0, 1, 2], pc_colums_names, fontsize=10)
    plt.colorbar()
    plt.xticks(range(len(features)), features, rotation=90, ha='right')
    components_and_features_plot = image.get_base64(plt, 'tight')
    plt.clf()

    return (two_first_components_plot.decode('ascii'), components_and_features_plot.decode('ascii'),
            wcss_plot.decode('ascii'), cumulative_explained_variance_ratio_plot.decode(
                'ascii'), pd.Series(explained_variance_ratio).to_json(orient='values'),
            pd.Series(kmeans_pca.labels_).to_json(orient='values'), more_important_features)
