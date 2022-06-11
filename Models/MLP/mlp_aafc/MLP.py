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
    
    def __init__(self,aafc_data, ecodistrict):
        self.aafc_data=aafc_data
        self.ecodistrict=ecodistrict
        
    def model_train_test(self):
        data=self.aafc_data[self.aafc_data['ECODISTRICT_ID']==self.ecodistrict]
        records=len(data)
        unique_twnships=data['TWP_ID'].nunique()
        labels = data['YieldKgAcre']
        features= data.drop(['YieldKgAcre'], axis = 1)
        
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)
        scaler = scale(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        train_index=train_features
        test_index=test_features
        
        
        train_features = scale(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        test_features = scale(test_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        regr = MLPRegressor(random_state=1, max_iter=5000).fit(train_features, train_labels)
        pred_train = regr.predict(train_features)
        pred = regr.predict(test_features)
        
        return scaler,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred

    
class mlp_ecodistrict_model_validation_scoring(mlp_ecodistrict_model_fit):
    
    def __init__(self,aafc_data, ecodistrict):
        mlp_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
        
    def validation_metrics(self):
        scaler,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred=mlp_ecodistrict_model_fit.model_train_test(self)
        
        # Training data
        mse_train = round(mean_squared_error(train_labels, pred_train,squared=False),2)
        train_index['Predicted_Yield']=pred_train
        
        # Test data
        
        
        mse_test =round(mean_squared_error(test_labels, pred,squared=False),2)
        
        
        #Calculate the absolute errors
        errors = abs(pred - test_labels)
        # Print out the mean absolute error (mae)
        mae=round(np.mean(errors), 2)
        
        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / test_labels)
        # Calculate and display accuracy
        accuracy = round(100 - np.mean(mape),2)
        
        print("Mean Squared Error Train: ",mse_train)
        print("Mean Squared Error Test: ",mse_test)

        print("Mean Absolute Error: ",mae)
        print("Accuracy: ",accuracy)
        
        
    def predicted_train_dataset(self):
        scaler,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred=mlp_ecodistrict_model_fit.model_train_test(self)
        train_index['Predicted_Yield']=pred_train
        return train_index

    def predicted_test_dataset(self):
        scalar,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred=mlp_ecodistrict_model_fit.model_train_test(self)
        test_index['Predicted_Yield']=pred
        return test_index
    def score(self,data):
        scaler,regr,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred=mlp_ecodistrict_model_fit.model_train_test(self)
        features=scale(data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        predicted_yield = regr.predict(features)
        data['Predicted_Yield']=predicted_yield
        return data