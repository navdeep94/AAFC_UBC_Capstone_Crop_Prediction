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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class lasso_ecodistrict_model_fit():
    """
    A class representing LASSO model having certain attributes and the methods
    ...
    Attributes
    ----------
    aafc_data : dataframe
        Clean and Wrangled Crop Yield Dataset
    ecodistrict : str
        specific Ecodistrict region ID
    
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
                returns scaler,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance
        '''
        
        ## Filtering the Wrangled Data with the required Ecodistrict
        data=self.aafc_data[self.aafc_data['ECODISTRICT_ID']==self.ecodistrict]
        
        # Store the length or observations for that Ecodistrict ID
        records=len(data)
        
        ## Calculating unique townships for that Ecodistrict ID
        unique_twnships=data['TWP_ID'].nunique()
        
        ## Store labels or response for modelling
        labels = data['YieldKgAcre']
        
        ## Store Features or predictors for modelling
        features= data.drop(['YieldKgAcre'], axis = 1)
        
        ##Storing the column names of the predictor variables
        feature_list = list(features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])].columns)
        
        ## Splits the data into training and testing into 80-20 ratio
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)
        
        ## Standardizing the features
        scaler = StandardScaler().fit(train_features.loc[:, ~train_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Storing the original training and test predictor data into another variable before scaling
        train_index=train_features
        test_index=test_features
        
        ## storing the scaled training and test predictor data after removing the columns like Township ID, Ecodistrict ID and Year which are not required for modelling purposes
        train_features = scaler.transform(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        test_features = scaler.transform(test_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Defining Lasso model with 5 fold cross-validation
        model = LassoCV(cv=5, random_state=0, max_iter=10000)
        
        ## Fitting the model with training data
        model.fit(train_features, train_labels)
        
        ## Choosing the best alpha value and fitting the model accordingly
        lasso_best = Lasso(alpha=model.alpha_)
        lasso_best.fit(train_features, train_labels)
        
        ## Predicting the yield for both training and the test data set 
        pred_train = lasso_best.predict(train_features)
        pred = lasso_best.predict(test_features)
        
        ## returning necessary data points for further model evaluation and predictions
        return scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list

    
class lasso_ecodistrict_model_validation_scoring(lasso_ecodistrict_model_fit):
    """
    A child class used to evaluate LASSO Model and make predictions having its own attributes and the methods, as well as attributes and methods of its parent class
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
        Evaluates LASSO in terms of MSE Train, Test, Mean Absolute Error and Accuracy
    
    predicted_train_dataset(additional=""):
        Predicts the Crop Yield Prediction Based on Training Predictors
    predicted_test_dataset(additional=""):
        Predicts the Crop Yield Prediction Based on Testing Predictors
    feature_importance(additional=""):
        Return the exact variance explained by the principal components selected
    score(additional=""):
        Predicts the Crop Yield  based on new predictor data
    """
    
    def __init__(self,aafc_data, ecodistrict):
        ## Instatiating this class by instatiating the parent class
        lasso_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
        
    def validation_metrics(self):
        '''
        Evaluates Model Performance and prints MSE Train,Test,Mean Absolute Error and Accuracy
        Parameters:
                self: Class Object
        Returns:
                None
        '''
        ## using the data returned from the parent class function for model evaluation
        scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        r2train=round(lasso_best.score(train_features, train_labels)*100, 2)
        r2test=round(lasso_best.score(test_features, test_labels)*100, 2)
        
        ## Calculating the MSE for training data
        mse_train = round(mean_squared_error(train_labels, pred_train,squared=False),2)
 
        ## Calculating the MSE for test data
        mse_test =round(mean_squared_error(test_labels, pred,squared=False),2)
        
        
        ## Calculating the absolute error value
        errors = abs(pred - test_labels)
        
        ## Calculating the MAE
        mae=round(np.mean(errors), 2)
        
        ## Calculating the mean absolute percentage error (MAPE)
        mape = 100 * (errors / test_labels)
        
        ## Calculating the accuracy
        accuracy = round(100 - np.mean(mape),2)
        
        ## Printing all the calculated performance metrics
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
        scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        
        ## Storing the training predicted values in the original training dataset
        train_index['Predicted_Yield']=pred_train
        return train_index
    
    def feature_importance(self):
        '''
        Returns the important features and the respective importance
        Parameters:
                self: Class Object
        Returns:
                returns a dataframe containing the important features and the importance values
        '''
        scalar,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        ## Calculating the importance of each variable
        coefficients=lasso_best.coef_
        importance = np.abs(coefficients)
        
        ## Obtaining the features and the respective importance for only the features having an importance greater than zero
        feature_lst=[]
        importance_lst=[]
        for i in range(0,len(importance)):
            if importance[i]>0:
                feature_lst.append(feature_list[i])
                importance_lst.append(importance[i])
        feature_importance_df=pd.DataFrame()
        feature_importance_df['features']= feature_lst
        feature_importance_df['importance']=importance_lst
        return feature_importance_df.sort_values(by=['importance'],ascending=False)

    def predicted_test_dataset(self):
        '''
        Returns the Predicted Crop Yield in the Testing Set
        Parameters:
                self: Class Object
        Returns:
                returns test_index i.e., dataset with predicted values of crop yield based on Testing set predictors
        '''
        
        ## using the data returned from the parent class function for test set prediction
        scalar,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        
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
        scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        
        ## Removing the columns not required for modelling purpose and scaling the same
        features=scaler.transform(data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Calculating predictions via model for new data set predictors
        predicted_yield = lasso_best.predict(features)
        
        ## Storing the predicted values in the new dataset
        data['Predicted_Yield']=predicted_yield
        return data
        
