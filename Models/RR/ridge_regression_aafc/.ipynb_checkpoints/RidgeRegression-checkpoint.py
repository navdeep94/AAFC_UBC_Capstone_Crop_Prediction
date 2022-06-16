# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# importing all of the required libraries. 
# various functions are imported from different libraries and packages. 
import pandas as pd    # pandas
import numpy as np     # numpy
from sklearn import model_selection   # from sklearn package imported model_selection
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

# to avoid warning messages (as they are consuming a lot of space, making it difficult to view this file)
import warnings
warnings.filterwarnings('ignore')



class ridge_regression_ecodistrict_model_fit():
    """
    A class representing Ridge Regression Model having certain attributes and the methods.
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
        
        ## Filtering the Wrangled Data with the required Eco district
        data=self.aafc_data[self.aafc_data['ECODISTRICT_ID']==self.ecodistrict]
        
        ## Store the length or observations for that Eco district ID
        records=len(data)
        
        ## Calculating unique townships for that Eco district ID
        unique_twnships=data['TWP_ID'].nunique()
        
        ## Store labels or response for modelling
        labels = data['YieldKgAcre']
        
        ## Store Features or predictors for modelling
        features= data.drop(['YieldKgAcre'], axis = 1)
        feature_list = list(features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])].columns)
        
        ## Splits the data into training and testing into 80-20 ratio
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)
        
        ## storing the scaled training predictor data after removing some of the unnecessary columns like Township ID, Eco district ID and Year not required for modelling purposes
        scaler = StandardScaler().fit(train_features.loc[:, ~train_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])       

        ## Storing the original training predictor data into another variable before scaling
        train_index=train_features
        
        ## Storing the original test predictor data into another variable before scaling
        test_index=test_features        
        
        ## storing the scaled training predictor data after removing some of the unnecessary columns like Township ID, Eco district ID and Year not required for modelling purposes
        train_features = scaler.transform(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
         ## storing the scaled testing predictor data after removing some of the unnecessary columns like Township ID, Eco district ID and Year not required for modelling purposes
        test_features = scaler.transform(test_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## applying cross validation
        model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=10)
        
        ## Training the Ridge Regression on the trained dataset (fitting the model)
        model.fit(train_features, train_labels)
        
        ## Linear least squares with L2 regularization (minimizes the objective function).
        ridge_best = Ridge(alpha=model.alpha_)
        
        ## Fitting the model again on training data
        ridge_best.fit(train_features, train_labels)
        
        ## Calculating predicted values of the crop yield based on training predictors
        pred_train = ridge_best.predict(train_features)
        
        ## Calculating predicted values of the crop yield based on testing predictors
        pred = ridge_best.predict(test_features)
        
        ## returns necessary values which is to be used for model evaluation and predictions
        return scaler,ridge_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list
    

class ridge_regression_ecodistrict_model_validation_scoring(ridge_regression_ecodistrict_model_fit):
    """
    A child class used to evaluate Ridge Regression Model and make predictions having its certain attributes and the methods as well as attributes and methods of its parent class
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
        
    feature_importance(additional=""):
        Evaluates the important features (predictors) and provides their importance scores in descending order.
    
    predicted_test_dataset(additional=""):
        Predicts the Crop Yield Prediction Based on Testing Predictors
    score(additional=""):
        Predicts the Crop Yield Prediction Based on any new data Predictors
    """
        
    
    def __init__(self,aafc_data, ecodistrict):
        
        ## Instatiating this class by instatiating the parent class        
        ridge_regression_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)   
        
    
    def validation_metrics(self):
        '''
        Evaluates Model Performance and prints MSE Train,Test,Mean Absolute Error and Accuracy
        Parameters:
                self: Class Object
        Returns:
                None
        '''
        
        
       ## using the data returened from the parent class function for model evaluation
        scaler,ridge_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=ridge_regression_ecodistrict_model_fit.model_train_test(self) 
        
        ## Calculating R squared for the training set
        r2train=round(ridge_best.score(train_features, train_labels)*100, 2)
        
        ## Calculating R squared for the test set
        r2test=round(ridge_best.score(test_features, test_labels)*100, 2)
        
        
        # Training data
        ## Calculating the Mean Squared Error for Training Set
        mse_train = round(mean_squared_error(train_labels, pred_train,squared=False),2)
        
        ## Storing the training predicted values in the original training dataset
        train_index['Predicted_Yield']=pred_train
        
        # Test data
        ## Calculating the Mean Squared Error for Test Set
        mse_test =round(mean_squared_error(test_labels, pred,squared=False),2)
        
        
        ## Calculate the absolute errors
        errors = abs(pred - test_labels)
        
        ## Print out the mean absolute error (mae)
        mae=round(np.mean(errors), 2)
        
        ## Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / test_labels)
        
        ## Calculate and display accuracy
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
        scaler,ridge_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=ridge_regression_ecodistrict_model_fit.model_train_test(self)
        
        ## Storing the training predicted values in the original training dataset
        train_index['Predicted_Yield']=pred_train
        
        return train_index
    

    def feature_importance(self):
        '''
        Returns the feature (predictors) importance 
        Parameters:
                self: Class Object
        Returns:
                returns a dataframe with two columns, one for feature names and another for their importance in descending order
        '''
        
        ## using the data returned from the parent class function for test set prediction        
        scalar,ridge_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=ridge_regression_ecodistrict_model_fit.model_train_test(self)
        
        ## getting the importance coefficients and storing them in "importance" variable
        coefficients=ridge_best.coef_
        importance = np.abs(coefficients)
        
        ## creating empty lists (for creating dataframe)
        feature_lst=[]
        importance_lst=[]
        
        ## For loop iterating through the values of "feature_list" and "importance" lists
        for i in range(0,len(importance)):
            if importance[i]>0:
                
                ## appending the feature names and the importance values in two new lists 
                feature_lst.append(feature_list[i])
                importance_lst.append(importance[i])
        
        ## creating a dataframe
        feature_importance_df=pd.DataFrame()
        
        ## adding the newly created lists as a column in the new dataframe
        feature_importance_df['features']= feature_lst
        feature_importance_df['importance']=importance_lst
        
        ## returns the dataframe, sorted by importance values in descending order
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
        scalar,ridge_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=ridge_regression_ecodistrict_model_fit.model_train_test(self)
        
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
        scaler,ridge_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=ridge_regression_ecodistrict_model_fit.model_train_test(self)
        
        ## scaled the new data set features or predictors
        features=scaler.transform(data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Calculating prediction via model for new data set predictors
        predicted_yield = ridge_best.predict(features)
        
        ## Storing the predicted values in the new dataset
        data['Predicted_Yield']=predicted_yield
        
        return data
        
    
    
    
# -


