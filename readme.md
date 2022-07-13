# Abstract
Wildfires are increasingly common in the Southwestern region of the United States due to climate change. The purpose of this project is to create an interactive application where a user can see the likelihood of a wildfire based on their inputs.

# Data

This project utilized two datasets:
- [Dataset #1](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2013-0009.5) from the US Department of Agriculture: It provides information on all of the wildfires from 1992-2018. For this project, a subset of data from the years 2000-2005 yielded 490,092 rows, with each row containing data about a single wildfire.

- [Dataset #2](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00861/html) from the National Centers of Environmental Information (National Oceanic and Atmospheric Administration): This dataset provides an aggregation of historical climate data that is updated daily. For this project, I looked solely at data from the US Historical Climate Network from the years 2000-2005. This resulted in 1218 files (one for each station in the HCN) ranging in length from 210-427 rows of data, where each row contains a month's worth of data about a given weather measurement (e.g., precipitation levels).

# Data Pipeline and Tools:
### Data Ingestion:
Data was initially downloaded directly from the server. Weather data can be updated daily through a python file that downloads from the server.

### Data Storage: 
Data is stored on Google Cloud in SQL.

### Processing:
All processing of data (including EDA and modeling) was done with PySpark on Google Cloud.

### Deployment:
Streamlit was used to create two interactive apps:
- an app to explore the data
- an app to predict the likelihood of a wildfire

# Algorithms
For this project, I trained a simple logistic regression model to predict the size of a wildfire, given the following parameters: latitude, longitude, maximum daily temperature, minimum daily temperature, and precipitation level.

# Communication
See slides
The Streamlit app can be accessed [here](https://ysancia-engineering-fires-app-8sp987.streamlitapp.com/)