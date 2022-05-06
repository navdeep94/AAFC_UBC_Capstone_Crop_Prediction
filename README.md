# Exploring optimal machine learning models for predicting crop yield at township level

- Your title can change over time.
- Here you should add the problem your project is trying to solve

## Team Members

- Navdeep Singh Saini: Data Science Enthusiast with the technical and business expertise along with the Analytical acumen to translate real world customer problems into data-driven decisions
- Person 2: one sentence about you!
- Person 3: one sentence about you!
- Person 4: one sentence about you!

## Describe your topic/interest in about 150-200 words

Crop yield and production are crucial to meet the demands of millions in this country. Apart from that, one of the most important aspects is to increase exports, and to achieve that, prior information regarding the yield is highly significant, especially for the decision makers who can further formulate the export policies based on the results generated related to the crop yield forecast. Due to crops being one of the 4 key sectors managed by Agri-Food Canada, it is in our best interest to make sure that this sector is performing to its highest capacity.  Not to mention that there are several millions of individuals counting on its sustained performance, we are tasked with ensuring that we’re able to keep up with the demand of them in the coming years and the future. As AAFC works to grow exports, while providing leadership in the expansion and development of a competitive, innovative, and sustainable Canadian agriculture and agri-food sector, the project is quite essential to come up with data driven planning with regards to agricultural production

## About this Project

We seek to test Machine Learning (ML) methods for predicting crop yield at township level. Currently, crop yield prediction and forecasting is accomplished using the Canadian Crop Yield Forecaster where weekly low resolution (~1km) NDVI values and daily agrometeorological variables from unevenly distributed climate stations are used in a blended model which uses statistical and biophysical modules (Newlands et al. 2014). A number of issues arise when statistical based models are used to predict crop yield such as multicollinearity among predictor variables and lack of plausible scientific explanation for  some predictor variables that are selected in some experiments (inference).Studies with ML algorithms have shown that predictions can be made without making any prior assumptions about the relationship between the response and predictor variables (You et al., 2017, Cai et al., 2019, Yang et al., 2019). Using the Canadian Prairie domain as the study area, we seek to test the following algorithms (or better ones) in the prediction of canola, spring wheat and barley; XGBoost and Lasso and compare results with observed values. We propose to use  cloud-based resources such as Google Earth Engine for processing large amounts of data without physically downloading the data. This will allow the use of complicated procedures such as Deep Learning libraries for training the models. 

## Describe your dataset in about 150-200 words

### Crop yield data:###

AAFC has access to crop yield data provided by provincial governments of Alberta, Saskatchewan and Manitoba via an memorandum of understanding (MOU). Originally at quarter section  level, these data sets have been rescaled to township level using geostatistical techniques. These data sets will be used as observations for training the models.

### Earth Observation predictor variables:###
AAFC scientists and partners have assembled ground and satellite derived predictor variables for improving the crop yield prediction skill at finer scale. Examples include the Normalized Difference Vegetation Index (NDVI) from Synthetic Aperture Radar (SAR) and optical sensors, surface soil moisture from active and passive microwave sensors, the evaporative stress index from thermal-optical data, leaf area index from Sentinel-2 and high resolution modelled weather data sets from the Canadian Meteorological Centre at 2.5km and 10km. These data sets have been aggregated at township level for at least 20 years. Other data sets include value added variables derived from intermediate models such as the Versatile Soil Moisture Budget (VSMM) from which soil moisture and crop stress index are calculated and used as input in the crop yield prediction models. Heat related indices such as Growing degree days and stress days are equally available. All data sets are organized as tables and tagged to townships by crop type (using crop density maps).


## Acknowledgements and references 

