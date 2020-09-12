from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
# pylint: disable=relative-beyond-top-level
from .common import generate_cluster_labels, get_base64, COLOR_LIST

def generate_svm_color_map(number):
    res = {}

    for i in range(number):
        res[i] = COLOR_LIST[i]

    return res

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape((xx.shape))
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def start(df, x_scaled_reduced, clusters_number):
    # pipelilne
    steps = [('SVM', SVC())]
    pipeline = Pipeline(steps)

    cluster_data = df.drop(['cluster'], axis=1)

    # Split dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        cluster_data, df.cluster, test_size=0.25, stratify=df.cluster, random_state=30)

    # Scale the data for PCA
    scaler1 = StandardScaler()
    scaler1.fit(x_test)
    x_test_scaled = scaler1.transform(x_test)

    # Apply PCA
    pca1 = PCA(n_components=2) 
    x_test_scaled_reduced = pca1.fit_transform(x_test_scaled)

    parameters = {'SVM__C': (0.001, 0.1, 10,
                             100, 10e5), 'SVM__gamma': (0.1, 0.01)}

    create_grid = GridSearchCV(
        pipeline, param_grid=parameters, cv=clusters_number-1)  # check
    create_grid.fit(x_train, y_train)

    svm = SVC(kernel='rbf', C=float(create_grid.best_params_[
              'SVM__C']), gamma=float(create_grid.best_params_['SVM__gamma']))

    classify = svm.fit(x_test_scaled_reduced, y_test)

    x0, x1 = x_test_scaled_reduced[:, 0], x_test_scaled_reduced[:, 1]
    xx, yy = make_meshgrid(x0, x1)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    cdictl = generate_svm_color_map(clusters_number)

    y_tar_list = y_test.tolist()
    yl1 = [int(target1) for target1 in y_tar_list]
    labels1 = yl1

    labll = generate_cluster_labels(clusters_number)

    for l1 in np.unique(labels1):
        ix1 = np.where(labels1 == l1)
        ax.scatter(x0[ix1], x1[ix1], c=cdictl[l1], label=labll[l1])

    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[
               :, 1], s=40, facecolors='none', edgecolors='navy', label='Support vectors')

    plot_contours(ax, classify, xx, yy, cmap='seismic', alpha=0.4)
    plt.legend(fontsize=15)

    plt.xlabel('pc1')
    plt.ylabel('pc2')

    svm_plot = get_base64(plt)
    plt.clf()

    return svm_plot.decode('ascii')
