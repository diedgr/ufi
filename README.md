![UFI logo](https://static.wixstatic.com/media/e8d0f3_4c2f029650d14abaa8576961ef13e985~mv2.png/v1/fill/w_167,h_167,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/UFI%20logo.png) 

# Underserved Farmers Index (UFI)
**Version 1.0.0 (updated December 23, 2024)**

## About

Historically underserved farmers and ranchers—including those who are beginning, socially disadvantaged, veterans, and limited resource farmers—have long faced barriers to accessing resources. While there is a growing body of research and federal initiatives aimed at addressing these inequities, no tool has been developed to quantify or assess the geographic concentration and reach of underserved farmers and ranchers at the county level.

The Underserved Farmers Index (UFI) seeks to fill this gap by creating a comprehensive, nationally applicable index that uses principal component analysis to analyze socio-demographic, farm operation, and financial characteristics data from the [2022 Census of Agriculture](https://quickstats.nass.usda.gov). By aggregating these factors, the UFI provides an in-depth look at where underserved farmers and ranchers are located.

The UFI is an essential tool for policymakers, researchers, and community leaders seeking to understand and address the disparities in agricultural access across the United States. By providing a clearer picture of underserved agricultural communities, the UFI helps guide targeted interventions and support sustainable development and agricultural policies.

## Installation

To get started with the UFI, clone the repository and install the required dependencies listed in the ``requirements.txt`` file. Create a virtual environment (optional but recommended) and use the command ``pip install -r requirements.txt`` to install the necessary packages, which include ``geopandas``, ``matplotlib``, ``numpy``, ``pandas``, and others. The dataset used in the analysis is ``ufi_v1.0.0.csv``, and the Python script that demonstrates how the index was developed, along with sensitivity analysis and stability tests, is ``ufi_v1.0.0.py``. After setting up the environment and installing dependencies, you can run the script with ``python ufi_v1.0.0.py`` to generate the UFI based on 2022 Census of Agriculture [data](https://quickstats.nass.usda.gov).
