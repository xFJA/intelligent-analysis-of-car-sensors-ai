import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random
import base64
import io

# TODO: find a new way to group the dataset
CLUSTER_TYPES = ["Road", "Highway", "Dirty road"]
NUMBER_COMPONENTS = 3


def get_cluster_column(cont):
    res = []

    cluster_size = len(CLUSTER_TYPES)

    i = 0

    while i < cont:
        cluster = CLUSTER_TYPES[random.randint(0, cluster_size) - 1]
        times = random.randint(80, 269)

        res += [cluster] * times
        i += times

    return res


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

    # Standardizing (mean = 0 , variance = 1)
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

    # Create the principal components
    principal_components = pca.fit_transform(x)

    # Get PCA scores
    pca_scores = pca.transform(x)

    # Fit k means using the data from PCA
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

    # Create columns labels for each component
    pc_colums_names = generate_pc_columns_names(NUMBER_COMPONENTS)

    principal_components_df = pd.DataFrame(
        data=principal_components, columns=pc_colums_names)

    # Add the cluster column
    cluster_column = get_cluster_column(len(df.index)-1)
    df_complete = pd.concat(
        [principal_components_df, pd.Series(cluster_column)], axis=1)
    df_complete.rename(columns={0: 'cluster'}, inplace=True)

    # Plot two first compontens
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal component 1', fontsize=16)
    ax.set_ylabel('Principal component 2', fontsize=16)
    ax.set_title('PCA', fontsize=22)
    targets = CLUSTER_TYPES
    colors = ['#64b5f6', '#1e88e5', '#0d47a1']
    for target, color in zip(targets, colors):
        indicesToKeep = df_complete['cluster'] == target
        ax.scatter(df_complete.loc[indicesToKeep, 'pc1'],
                   df_complete.loc[indicesToKeep, 'pc2'], c=color, s=50)
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
