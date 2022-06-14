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

## Importing libraries
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class mlp_ecodistrict_model_fit():
    """
    A class representing Multi-Layer Perceptron Model having certain attributes and the methods

    ...

    Attributes
    ----------
    aafc_data : dataframe
        Clean and Wrangled Crop Yield Dataset
    ecodistrict : str
        specific Eco district region ID
    

    Methods
    -------
    model_train_test(additional=""):
        Fits the model after train test split
    """
    
    def __init__(self,aafc_data, ecodistrict):
        ## Instantiating the class with the required data

        self.aafc_data=aafc_data
        self.ecodistrict=ecodistrict
        
    def model_train_test(self):
        '''
        Returns key datasets required for performing model evaluation metrics and predictions.

        Parameters:
                self: Class Object

        Returns:
                returns scaler,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred
        '''

        ## Filtering the Wrangled Data with the required Ecodistrict
        data=self.aafc_data[self.aafc_data['ECODISTRICT_ID']==self.ecodistrict]

        ## Store the length or observations for that Ecodistrict ID
        records=len(data)

        ## Calculating unique townships for that Ecodistrict ID
        unique_twnships=data['TWP_ID'].nunique()

        ## Store labels or response for modelling
        labels = data['YieldKgAcre']

        ## Store Features or predictors for modelling
        features= data.drop(['YieldKgAcre'], axis = 1)
        
        ## Splits the data into training and testing into 80-20 ratio
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)

        ## storing the scaled training predictor data after removing some of the unnecessary columns like Township ID, Eco district ID and Year not required for modelling purposes
        scaler = scale(train_features.loc[:, ~train_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Storing the original training predictor data into another variable before scaling
        train_index=train_features

        ## Storing the original testing predictor data into another variable before scaling
        test_index=test_features
        
        ## storing the scaled training predictor data after removing some of the unnecessary columns like Township ID, Eco district ID and Year not required for modelling purposes
        train_features = scale(train_features.loc[:, ~train_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])

        ## storing the scaled testing predictor data after removing some of the unnecessary columns like Township ID, Eco district ID and Year not required for modelling purposes
        test_features = scale(test_features.loc[:, ~test_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Training the multi layer perceptron model on the trained dataset
        regr = MLPRegressor(random_state=1, max_iter=5000).fit(train_features, train_labels)

        ## Calculating predicted values of the crop yield based on training predictors
        pred_train = regr.predict(train_features)

        ## Calculating predicted values of the crop yield based on testing predictors
        pred = regr.predict(test_features)
        
        ## returns necessary values which is to be used for model evaluation and predictions
        return scaler,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred

    
class mlp_ecodistrict_model_validation_scoring(mlp_ecodistrict_model_fit):
    """
    A child class used to evaluate Multi-Layer Perceptron Model and make predictions having its certain attributes and the methods as well as attributes and methods of its parent class

    ...

    Attributes
    ----------
    aafc_data : dataframe
        Clean and Wrangled Crop Yield Dataset
    ecodistrict : str
        specific Ecodistrict region ID
    

    Methods
    -------
    validation_metrics(additional=""):
        Evaluates the Multi Layer Layer Perceptron in terms of MSE Train, Test, Mean Absolute Error and Accuracy
    
    predicted_train_dataset(additional=""):
        Predicts the Crop Yield Prediction Based on Training Predictors

    predicted_test_dataset(additional=""):
        Predicts the Crop Yield Prediction Based on Testing Predictors

    score(additional=""):
        Predicts the Crop Yield Prediction Based on any new data Predictors
    """
    
    def __init__(self,aafc_data, ecodistrict):
        ## Instantiating this class by instantiating the parent class

        mlp_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
        
    def validation_metrics(self):
        '''
        Evaluates Model Performance and prints MSE Train,Test,Mean Absolute Error and Accuracy

        Parameters:
                self: Class Object

        Returns:
                None
        '''

        ## using the data returned from the parent class function for model evaluation
        scaler,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred=mlp_ecodistrict_model_fit.model_train_test(self)
        
        ## Calculating the Mean Squared Error for Training Set
        mse_train = round(mean_squared_error(train_labels, pred_train,squared=False),2)

        ## Storing the training predicted values in the original training dataset
        train_index['Predicted_Yield']=pred_train
        
        ## Calculating the Mean Squared Error for Testing Set
        mse_test =round(mean_squared_error(test_labels, pred,squared=False),2)
        
        ## Calculate the absolute errors
        errors = abs(pred - test_labels)

        ## Print out the mean absolute error (mae)
        mae=round(np.mean(errors), 2)
        
        ## Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / test_labels)

        ## Calculate accuracy
        accuracy = round(100 - np.mean(mape),2)
        
        print("Mean Squared Error Train: ",mse_train)
        print("Mean Squared Error Test: ",mse_test)

        print("Mean Absolute Error: ",mae)
        print("Accuracy: ",accuracy)
        
        
    def predicted_train_dataset(self):
        '''
        Returns the Predicted Crop Yield in the Training Set

        Parameters:
                self: Class Object

        Returns:
                returns train_index i.e., dataset with predicted values of crop yield based on training set predictors
        '''

        ## using the data returned from the parent class function for train set prediction
        scaler,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred=mlp_ecodistrict_model_fit.model_train_test(self)

        ## Storing the training predicted values in the original training dataset
        train_index['Predicted_Yield']=pred_train
        
        return train_index

    def predicted_test_dataset(self):
        '''
        Returns the Predicted Crop Yield in the Testing Set

        Parameters:
                self: Class Object

        Returns:
                returns test_index i.e., dataset with predicted values of crop yield based on Testing set predictors
        '''
        ## using the data returned from the parent class function for test set prediction
        scalar,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred=mlp_ecodistrict_model_fit.model_train_test(self)

        ## Storing the testing predicted values in the original test dataset
        test_index['Predicted_Yield']=pred

        return test_index
    def score(self,data):
        '''
        Returns the Predicted Crop Yield in the new Data Set

        Parameters:
                self: Class Object

        Returns:
                returns data i.e., dataset with predicted values of crop yield based on New Dataset predictors
        '''
        
        ## using the data returned from the parent class function for any new data set prediction
        scaler,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred=mlp_ecodistrict_model_fit.model_train_test(self)

        ## scaled the new data set features or predictors
        features=scale(data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Calculating prediction via model for new data set predictors
        predicted_yield = regr.predict(features)

        ## Storing the predicted values in the new dataset
        data['Predicted_Yield']=predicted_yield

        return data
