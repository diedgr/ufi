#!/usr/bin/env python
# coding: utf-8

# # Underserved Farmers Index (UFI)
# **Version 1.0.0 (updated December 23, 2024)**

# In[ ]:


# imports
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import missingno as msno
import plotly.figure_factory as ff
import plotly.express as px
import shapely
import geopandas
from shapely.geometry import MultiPolygon, Polygon
import requests
import json


# ## Data Preprocessing

# In[ ]:


# loading data
data = pd.read_csv('ufi_v1.0.0.csv') # select correct version


# In[ ]:


# filter rows where COUNTY is populated 
filtered_data = data[data['COUNTY'].notna()]

# group by STATE and calculate the average of netcash_producer_by_operation
AVG_NETCASH_BY_STATE = filtered_data.groupby('STATE')['NETCASH_PRD_BY_OPS'].mean()

# map the average values back to the original df as a new column
data['AVG_NETCASH_BY_STATE'] = data['STATE'].map(AVG_NETCASH_BY_STATE)

# create binary variable: 1 if below average, 0 if above or equal to average
data['BELOWAVG_NETCASH'] = (data['NETCASH_PRD_BY_OPS'] < data['AVG_NETCASH_BY_STATE']).astype(int)


# In[ ]:


# checking for missing values
df_cleaned = data[data['COUNTY'].notna() & (data['COUNTY'] != '')]
nan_per_column = df_cleaned.isna().sum()
print(nan_per_column)


# In[ ]:


# replacing NaN with zero
df_cleaned['NETCASH_PRD_BY_OPS'] = df_cleaned['NETCASH_PRD_BY_OPS'].fillna(0)


# In[ ]:


# summary statistics
mean_values = df_cleaned.mean()
std_values = df_cleaned.std()

summary_stats = pd.DataFrame({
    'Mean': mean_values,
    'Standard Deviation': std_values
})

print(summary_stats)


# In[ ]:


# list of variables to visualize
variables = [
    'PCT_NONWHITE',
    'PCT_HISPANIC',
    'PCT_FEMALE',
    'PCT_MILITARY',
    'PCT_BELOW35',
    'PCT_FARMYEAR11',
    'PCT_OPS_LOSS',
    'BELOWAVG_NETCASH'
]

# creating histograms
plt.figure(figsize=(15, 10))

for i, variable in enumerate(variables):
    plt.subplot(3, 3, i + 1)  
    sns.histplot(df_cleaned[variable], bins=30, kde=True)  
    plt.title(f'Distribution of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[ ]:


# correlation matrix
correlation_matrix = df_cleaned[variables].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

print(correlation_matrix)


# ## PCA 

# In[ ]:


# isolating revelant variables
data_for_pca = df_cleaned[variables]

# standardizing the data with Z-scores
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data_for_pca)

# PCA
pca = PCA()
principal_components = pca.fit_transform(standardized_data)


# In[ ]:


# creating df for principal components
pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# determining how many principal components to retain
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(explained_variance_ratio >= 0.90) + 1  # e.g., 90% of variance
print(n_components)


# In[ ]:


# absolute values of the loadings across retained components
retained_loadings = np.abs(pca.components_[:n_components].T)  # Absolute loadings for retained components
average_loadings = np.mean(retained_loadings, axis=1)  # Average of absolute loadings

# normalizing the average loadings to sum to 1
weights_normalized = average_loadings / np.sum(average_loadings)

# creating df to store the variable names and their positive normalized weights
weights_df = pd.DataFrame({
    'Variable': variables,
    'Weight': weights_normalized
})


# In[ ]:


# plotting normalized positive weights as a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Variable', y='Weight', data=weights_df, palette='dark:salmon')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Weight')
plt.xlabel('Variable')
plt.tight_layout()
plt.show()


# In[ ]:


# priting normalized weights
for var, weight in zip(variables, weights_normalized):
    print(f"Weight for {var}: {weight}")


# ## Index Development

# In[ ]:


# creating index as a weighted sum of the standardized variables using the calculated weights
index = np.dot(standardized_data, weights_normalized)

# adding index to the original dataframe
df_cleaned['UFI'] = index


# In[ ]:


# rank transformation
df_cleaned['UFI_Ranked'] = df_cleaned['UFI'].rank(method='average') 
df_cleaned['UFI_Ranked'] = (df_cleaned['UFI_Ranked'] - df_cleaned['UFI_Ranked'].min()) / \
                                               (df_cleaned['UFI_Ranked'].max() - df_cleaned['UFI_Ranked'].min())


# In[ ]:


# defining disadvantaged cut-off (90 percentile)
percentile_90 = df_cleaned['UFI_Ranked'].quantile(0.90)
df_cleaned['Disadvantaged'] = df_cleaned['UFI_Ranked'] >= percentile_90


# ## Counts

# In[ ]:


# disadvantaged counties per state
disadvantaged_count_by_state = df_cleaned[df_cleaned['Disadvantaged']].groupby('STATE').size()
print(disadvantaged_count_by_state)

total_counties_by_state = df_cleaned.groupby('STATE').size()
disadvantaged_count_by_state = df_cleaned[df_cleaned['Disadvantaged']].groupby('STATE').size()
percent_disadvantaged = (disadvantaged_count_by_state / total_counties_by_state) * 100
percent_disadvantaged_df = percent_disadvantaged.reset_index(name='Percent_Disadvantaged').fillna(0)
print(percent_disadvantaged_df)


# In[ ]:


# counties classified as disadvantaged
total_counties = len(df_cleaned)
disadvantaged_count = df_cleaned[df_cleaned['Disadvantaged']].shape[0]
disadvantaged_percentage = (disadvantaged_count / total_counties) * 100
print(f"Total counties: {total_counties}")
print(f"Disadvantaged counties: {disadvantaged_count}")
print(f"Percentage of disadvantaged counties: {disadvantaged_percentage:.2f}%")


# In[ ]:


# mapping states to U.S. Census Bureau regions and divisions
us_census_regions_and_divisions = {
    'Northeast': {
        'New England': ['CONNECTICUT', 'MAINE', 'MASSACHUSETTS', 'NEW HAMPSHIRE', 'RHODE ISLAND', 'VERMONT'],
        'Middle Atlantic': ['NEW JERSEY', 'NEW YORK', 'PENNSYLVANIA']
    },
    'Midwest': {
        'East North Central': ['ILLINOIS', 'INDIANA', 'IOWA', 'MICHIGAN', 'MINNESOTA', 'OHIO', 'WISCONSIN'],
        'West North Central': ['KANSAS', 'MISSOURI', 'NEBRASKA', 'NORTH DAKOTA', 'SOUTH DAKOTA']
    },
    'South': {
        'South Atlantic': ['DELAWARE', 'FLORIDA', 'GEORGIA', 'MARYLAND', 'NORTH CAROLINA', 'SOUTH CAROLINA', 'VIRGINIA', 'WEST VIRGINIA'],
        'East South Central': ['ALABAMA', 'KENTUCKY', 'MISSISSIPPI', 'TENNESSEE'],
        'West South Central': ['ARKANSAS', 'LOUISIANA', 'OKLAHOMA', 'TEXAS']
    },
    'West': {
        'Mountain': ['ARIZONA', 'COLORADO', 'IDAHO', 'MONTANA', 'NEVADA', 'NEW MEXICO', 'UTAH', 'WYOMING'],
        'Pacific': ['ALASKA', 'CALIFORNIA', 'HAWAII', 'OREGON', 'WASHINGTON']
    }
}

def get_region_and_division(state):
    state = state.upper()  # Convert the state name to uppercase
    for region, divisions in us_census_regions_and_divisions.items():
        for division, states in divisions.items():
            if state in states:
                return region, division
    return 'Unknown', 'Unknown'

# applying the region and division mapping
df_cleaned[['Region', 'Division']] = df_cleaned['STATE'].apply(lambda state: pd.Series(get_region_and_division(state)))

# calculating the count and percentage of disadvantaged counties
grouped = df_cleaned.groupby(['Region', 'Division'])

# counting total counties and disadvantaged counties in each group
region_division_summary = grouped.agg(
    Total_Counties=('STATE', 'size'),
    Disadvantaged_Counties=('Disadvantaged', lambda x: x.sum())
)

# calculating the percentage of disadvantaged counties
region_division_summary['Percent_Disadvantaged'] = (region_division_summary['Disadvantaged_Counties'] / region_division_summary['Total_Counties']) * 100
print(region_division_summary)


# ## Visualization

# In[ ]:


# loading GeoJSON file
geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
response = requests.get(geojson_url)
geojson_data = response.json()

# extracting the FIPS codes from the GeoJSON
geojson_fips = [feature['id'] for feature in geojson_data['features']]

# converting to a set for fast comparison
geojson_fips_set = set(geojson_fips)


# In[ ]:


# ensuring FIPS codes are strings 
df_cleaned['FIPS'] = df_cleaned['FIPS'].astype(str).str.split('.').str[0].str.zfill(5)

# creating choropleth map of UFI scores
fig = px.choropleth(
    df_cleaned,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="FIPS", 
    color="UFI_Ranked",  
    color_continuous_scale="geyser",
    scope="usa",  
    title="USA by UFI Ranked",
    labels={"UFI_Ranked": "UFI Value"}
)

fig.update_geos(fitbounds="locations", visible=True)  # Make sure all counties are included

fig.show()


# In[ ]:


# creating binary map. of disadvantagegd status
color_map = {0: "#D3D3D3", 1: "#ca562c"}  

fig = px.choropleth(
    df_cleaned,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="FIPS",
    color="Disadvantaged",  
    color_discrete_map=color_map,
    scope="usa", 
    title="USA by UFI Ranked",
    labels={"UFI_Ranked": "UFI Value"}
)

fig.update_geos(fitbounds="locations", visible=True)  

fig.show()


# ## Sensitivity Analysis

# In[ ]:


# number of Monte Carlo simulations
n_simulations = 100000

# calculating the mean and standard deviation of each variable
means = data_for_pca.mean()
std_devs = data_for_pca.std()


# In[ ]:


# generating random samples for each variable using normal distribution
simulated_data = np.random.normal(loc=means, scale=std_devs, size=(n_simulations, len(variables)))


# In[ ]:


# using the normalized weights obtained from PCA to calculate the index
simulated_index = np.dot(simulated_data, weights_normalized)
print(simulated_index)


# In[ ]:


# ranking the simulated index values
simulated_index_ranked = pd.Series(simulated_index).rank(method='average')
simulated_index_ranked = (simulated_index_ranked - simulated_index_ranked.min()) / (simulated_index_ranked.max() - simulated_index_ranked.min())


# In[ ]:


# basic statistics of the simulated UFI_Ranked values
simulated_index_ranked.describe()


# In[ ]:


# ensuring no NaNs or infinite values in the simulated data
simulated_data = np.random.normal(loc=means, scale=std_devs, size=(n_simulations, len(variables)))


# In[ ]:


# Initialize a list to store sensitivity values
sensitivity = []

# performing sensitivity analysis for each variable
for i in range(len(variables)):
    # two scenarios: increasing and decreasing the i-th variable by 1 standard deviation
    high_impact = np.copy(simulated_data)
    low_impact = np.copy(simulated_data)
    
    high_impact[:, i] += std_devs[i]
    low_impact[:, i] -= std_devs[i]
    
    high_index = np.dot(high_impact, weights_normalized)
    low_index = np.dot(low_impact, weights_normalized)
    
    # calculating the absolute difference in the UFI values 
    sensitivity_range = np.mean(np.abs(high_index - low_index))
    
    # storing the sensitivity range and variable name
    sensitivity.append((variables[i], sensitivity_range))

# sorting the sensitivity by the largest impact
sensitivity_sorted = sorted(sensitivity, key=lambda x: x[1], reverse=True)

# unzipping the sorted sensitivity data into labels and values
labels, sensitivity_values = zip(*sensitivity_sorted)

# results and plot
print("Sensitivity Analysis for UFI Index (Raw Scale):")
for label, value in zip(labels, sensitivity_values):
    print(f"{label}: Sensitivity = {value:.4f}")

if sensitivity_values:
    plt.figure(figsize=(10, 6))
    plt.barh(labels, sensitivity_values, color='#B8390E')
    plt.xlabel('Sensitivity')
    plt.tight_layout()
    plt.show()
else:
    print("No sensitivity values to plot.")


# In[ ]:


# grouping variables by their classes
sdf_vars = ['PCT_FEMALE', 'PCT_HISPANIC', 'PCT_NONWHITE']
lrfr_vars = ['PCT_OPS_LOSS', 'BELOWAVG_NETCASH']
vfr_vars = ['PCT_MILITARY']
bfr_vars = ['PCT_FARMYEAR11']

# calculating total sensitivity for each group
sdf_sensitivity = sum(sensitivity_values[labels.index(var)] for var in sdf_vars if var in labels)
lrfr_sensitivity = sum(sensitivity_values[labels.index(var)] for var in lrfr_vars if var in labels)
vfr_sensitivity = sum(sensitivity_values[labels.index(var)] for var in vfr_vars if var in labels)
bfr_sensitivity = sum(sensitivity_values[labels.index(var)] for var in bfr_vars if var in labels)

# results and plot
print("\nSensitivity by Variable Class:")
print(f"SDFR Sensitivity: {sdf_sensitivity:.4f}")
print(f"LRFR Sensitivity: {lrfr_sensitivity:.4f}")
print(f"VFR Sensitivity: {vfr_sensitivity:.4f}")
print(f"BFR Sensitivity: {bfr_sensitivity:.4f}")

class_labels = ['SDFR', 'LRFR', 'VFR', 'BFR']
class_sensitivity = [sdf_sensitivity, lrfr_sensitivity, vfr_sensitivity, bfr_sensitivity]

plt.figure(figsize=(10, 6))
plt.barh(class_labels, class_sensitivity, color='#CC5500')
plt.xlabel('Sensitivity')
plt.tight_layout()
plt.show()


# ## Stability Test

# In[ ]:


# features and targets
X = df_cleaned[['PCT_NONWHITE', 'PCT_HISPANIC', 'PCT_FEMALE', 'PCT_MILITARY', 
                'PCT_BELOW35', 'PCT_FARMYEAR11', 'PCT_OPS_LOSS', 'BELOWAVG_NETCASH']]
y = df_cleaned['UFI_Ranked']


# In[ ]:


# kfold with 10 splits
kf = KFold(n_splits=10, shuffle=True, random_state=42)


# In[ ]:


# list to store UFI differences for each fold
UFI_diff = []

# 10-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # simple linear regression 
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # predicting UFI for the test set
    y_pred = model.predict(X_test)
    
    # calculating UFI difference (absolute error)
    fold_diff = mean_absolute_error(y_test, y_pred)
    
    # appending the fold difference (UFI_diff) to the list
    UFI_diff.append(fold_diff)


# In[ ]:


# results and plot
print("UFI Differences across folds:")
for i, diff in enumerate(UFI_diff, 1):
    print(f"Fold {i}: {diff:.4f}")
    
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), UFI_diff, marker='o', color='darkorange', linestyle='-', markersize=6, label='UFI Difference')
plt.xlabel('Fold Number')
plt.ylabel('UFI Difference (Mean Absolute Error)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Data Export

# In[ ]:


# safe to CSV
df_cleaned.to_csv('ufi_v1.0.0.csv', index=False)

