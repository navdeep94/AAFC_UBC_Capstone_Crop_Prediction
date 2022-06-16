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
from pca_rf_aafc.PCA_RF import pca_rf_ecodistrict_model_validation_scoring
<<<<<<< HEAD
=======
from pca_rf_aafc.PCA_RF import pca_rf_ecodistrict_model_validation_scoring
>>>>>>> b44d785239e892b487afffaff1037c679bf4a503
import pandas as pd

## Importing Wrangled and Clean Dataset
data=pd.read_csv("aafc_data.csv",index_col='Unnamed: 0')

<<<<<<< HEAD
try:
    ## Storing list of unique ecodistricts in the list
    ecodistricts = data['ECODISTRICT_ID'].unique()
=======
## Instantiating the object of the Class by passing data and Ecodistrict ID
ecodistrict_id = 748
test=pca_rf_ecodistrict_model_validation_scoring(data,ecodistrict_id)
>>>>>>> b44d785239e892b487afffaff1037c679bf4a503

    print("Enter Valid Ecodistrict ID: ")

    ## Storing input into the variable
    ecodistrict_id = int(input())

    ## Checking if the Eco District ID entered by the user exists in the data
    if (ecodistrict_id in ecodistricts):

        ## Instantiating the object of the Class by passing data and Ecodistrict ID
        test=pca_rf_ecodistrict_model_validation_scoring(data,ecodistrict_id)

        ## Show PCA_RF Model Performance Metrics
        print("The validation metrics for PCA_RF for eco district ID " + str(ecodistrict_id) + " are as follows: ")
        test.validation_metrics()

        ## Show Predicted Crop Yield on Training Dataset
        train_predicted_df = test.predicted_train_dataset()
        train_predicted_df.to_csv('Outputs/train_predicted_df.csv')

<<<<<<< HEAD
        ## Show Predicted Crop Yield on Testing Dataset
        test_predicted_df = test.predicted_test_dataset()
        test_predicted_df.to_csv('Outputs/test_predicted_df.csv')

        ## Show Number of Principal Components
        test.number_principal_components()

        ## Show Explained Variance
        test.cummulative_explained_variance()

        ## Importing New Test Set
        data_to_score=pd.read_csv("scoring_test_df.csv",index_col='Unnamed: 0')

        ## Show Predicted Crop Yield on New Test Set
        new_data_predicted_df = test.score(data_to_score)
        new_data_predicted_df.to_csv('Outputs/new_data_predicted_df.csv')

    else:
        print("Ecodistrict not present")
except:
    print("Invalid Input")
=======
## Show Predicted Crop Yield on New Test Set
new_data_predicted_df = test.score(data_to_score)
new_data_predicted_df.to_csv('Outputs/new_data_predicted_df.csv')
>>>>>>> b44d785239e892b487afffaff1037c679bf4a503
