# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
## Importing Packages and Modules and necessary Libraries
from lasso_aafc.LASSO import lasso_ecodistrict_model_validation_scoring
from lasso_aafc.LASSO import lasso_ecodistrict_model_fit
import pandas as pd

## Importing Wrangled and Clean Dataset
data=pd.read_csv("aafc_data.csv",index_col='Unnamed: 0')

## Instantiating the object of the Class by passing data and Ecodistrict ID
ecodistrict_id = 748
test=lasso_ecodistrict_model_validation_scoring(data,ecodistrict_id)

## Show Lasso Model Performance Metrics
print("The validation metrics for Lasso for eco district ID " + str(ecodistrict_id) + " are as follows: ")
test.validation_metrics()

## Show Predicted Crop Yield on Training Dataset
train_predicted_df = test.predicted_train_dataset()
train_predicted_df.to_csv('Outputs/train_predicted_df.csv')

## Show Predicted Crop Yield on Testing Dataset
test_predicted_df = test.predicted_test_dataset()
test_predicted_df.to_csv('Outputs/test_predicted_df.csv')

## Showing the Feature Importance
test.feature_importance()

## Importing New Test Set
data_to_score=pd.read_csv("scoring_test_df.csv",index_col='Unnamed: 0')

## Show Predicted Crop Yield on New Test Set
new_data_predicted_df = test.score(data_to_score)
new_data_predicted_df.to_csv('Outputs/new_data_predicted_df.csv')