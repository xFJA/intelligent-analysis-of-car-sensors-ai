import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random
import base64
import io

NUMBER_COMPONENTS = 3

def generate_pc_columns_names(number):
    res = []

    for i in range(number):
        res.append("pc"+str(i))

    return res


def pca(csv_file):

    # Read file
    df = pd.read_csv(csv_file)

    # Get features from the first row
    features = df.columns.values

    # Remove features
    x = df.loc[:, features].values

    # Standardizing data (mean = 0 , variance = 1)
    x = StandardScaler().fit_transform(x)

    # Cumulative explanined variance ratio plot
    pca = PCA().fit(x)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Components number')
    plt.ylabel('Cumulative explained variance')
    #plt.savefig("cumulative_explained_variance.png", bbox_inches='tight')

    # Create Base 64 string of the plot
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    cumulative_explained_variance_ratio_plot = base64.b64encode(
        pic_IObytes.read())
    plt.clf()

    # Create pca
    pca = PCA(n_components=NUMBER_COMPONENTS)

    # Fit model with the number of components selected
    pca.fit_transform(x)

    # Get PCA scores (create clusters based on the components scores)
    pca_scores = pca.transform(x)

    # Fit kmeans using the data from PCA
    wcss = []
    for i in range(1, 21):
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_pca.fit(pca_scores)
        wcss.append(kmeans_pca.inertia_)

    # Plot wcss
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, 21), wcss, marker='o', linestyle='--')
    plt.xlabel("Clusters number")
    plt.ylabel("WCSS")
    # plt.savefig('wcss.png')

    # Create Base 64 string of the plot
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    wcss_plot = base64.b64encode(pic_IObytes.read())

    # TODO: Get cluster nubmer from request
    # Run k-means with the number of cluster chosen
    kmeans_pca = KMeans(n_clusters=5, init='k-means++', random_state=42)

    # Fit data with the k-means pca model
    kmeans_pca.fit(pca_scores)

    # Create dataset with results from PCA and the cluster column
    df_kmeans_pca = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(pca_scores)], axis=1)
    df_kmeans_pca.columns.values[-3:] = generate_pc_columns_names(
        NUMBER_COMPONENTS)
    df_kmeans_pca['Kmeans value'] = kmeans_pca.labels_

    # Add cluster column with a label associated to each kmeans value
    df_kmeans_pca['Cluster'] = df_kmeans_pca['Kmeans value'].map(
        {0: 'Type 1', 1: 'Type 2', 2: 'Type 3', 3: 'Type 4', 4: 'Type 5'})

    # Create columns labels for each component
    pc_colums_names = generate_pc_columns_names(NUMBER_COMPONENTS)

    # Plot two first compontens
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal component 1', fontsize=16)
    ax.set_ylabel('Principal component 2', fontsize=16)
    ax.set_title('PCA', fontsize=22)
    targets = ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5']
    colors = ['#64b5f6', '#1e88e5', '#0d47a1', '#1a237e', '#3949ab']
    for target, color in zip(targets, colors):
        indicesToKeep = df_kmeans_pca['Cluster'] == target
        ax.scatter(df_kmeans_pca.loc[indicesToKeep, 'pc1'],
                   df_kmeans_pca.loc[indicesToKeep, 'pc2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    # plt.savefig("two_first_componets_plot.png")

    # Create Base 64 string of the plot
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    two_first_components_plot = base64.b64encode(pic_IObytes.read())

    # Print the amount of data that holds the components
    explained_variance_ratio = pca.explained_variance_ratio_

    # Create a plot for the porcetage of participation of each feature in each component
    plt.matshow(pca.components_, cmap='Blues')
    plt.yticks([0, 1, 2], pc_colums_names, fontsize=10)
    plt.colorbar()
    plt.xticks(range(len(features)), features, rotation=90, ha='right')
    #plt.savefig("pca_and_features_participation.png", bbox_inches='tight')

    # Create Base 64 string of the plot
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png', bbox_inches='tight')
    pic_IObytes.seek(0)
    components_and_features_plot = base64.b64encode(pic_IObytes.read())
    plt.clf()

    return two_first_components_plot.decode('ascii'), components_and_features_plot.decode('ascii'),  wcss_plot.decode('ascii'), cumulative_explained_variance_ratio_plot.decode('ascii'), pd.Series(explained_variance_ratio).to_json(orient='values')
