import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

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


# load dataset
df = pd.read_csv("live1.csv")

# TODO: Get the features from the first row in the csv
# features
features = ['ENGINE_RUN_TIME ()', 'ENGINE_RPM ()', 'VEHICLE_SPEED ()', 'THROTTLE ()', 'ENGINE_LOAD ()',
            'COOLANT_TEMPERATURE ()', 'LONG_TERM_FUEL_TRIM_BANK_1 ()', 'SHORT_TERM_FUEL_TRIM_BANK_1 ()', 'INTAKE_MANIFOLD_PRESSURE ()',
            'FUEL_TANK ()', 'ABSOLUTE_THROTTLE_B ()', 'PEDAL_D ()', 'PEDAL_E ()', 'COMMANDED_THROTTLE_ACTUATOR ()', 'FUEL_AIR_COMMANDED_EQUIV_RATIO ()',
            'ABSOLUTE_BAROMETRIC_PRESSURE ()', 'RELATIVE_THROTTLE_POSITION ()', 'INTAKE_AIR_TEMP ()', 'TIMING_ADVANCE ()', 'CATALYST_TEMPERATURE_BANK1_SENSOR1 ()',
            'CATALYST_TEMPERATURE_BANK1_SENSOR2 ()', 'CONTROL_MODULE_VOLTAGE ()', 'COMMANDED_EVAPORATIVE_PURGE ()', 'TIME_RUN_WITH_MIL_ON ()',
            'TIME_SINCE_TROUBLE_CODES_CLEARED ()', 'DISTANCE_TRAVELED_WITH_MIL_ON ()', 'WARM_UPS_SINCE_CODES_CLEARED ()']

# Remove features
x = df.loc[:, features].values

# Standardizing (mean = 0 , variance = 1)
x = StandardScaler().fit_transform(x)

# Create pca
pca = PCA(n_components=NUMBER_COMPONENTS)

# Create the principal components
principal_components = pca.fit_transform(x)

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
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal component 1', fontsize=16)
ax.set_ylabel('Principal component 2', fontsize=16)
ax.set_title('PCA', fontsize=22)
targets = CLUSTER_TYPES
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = df_complete['cluster'] == target
    ax.scatter(df_complete.loc[indicesToKeep, 'pc1'],
               df_complete.loc[indicesToKeep, 'pc2'], c=color, s=50)
ax.legend(targets)
ax.grid()
plt.savefig("two_first_componets_plot.png")

# Print the amount of data that holds the components
print(pca.explained_variance_ratio_)

# Create a plot for the porcetage of participation of each feature in each component
plt.matshow(pca.components_, cmap='magma')
plt.yticks([0, 1, 2], pc_colums_names, fontsize=10)
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=90, ha='right')
plt.savefig("pca_and_features_participation.png", bbox_inches='tight')
