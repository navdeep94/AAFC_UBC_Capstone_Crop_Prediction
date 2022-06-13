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
## Importing Libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold, cross_val_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class pcr_ecodistrict_model_fit():
    """
    A class representing Principal Components Regression Model having certain attributes and the methods

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
        ## Defining the instance of Principal Component Model
        pca = PCA()

        ## Filtering the Wrangled Data with the required Ecodistrict
        data=self.aafc_data[self.aafc_data['ECODISTRICT_ID']==self.ecodistrict]

        ## Strong the length or observations for that Ecodistrict ID
        records=len(data)

        ## Calculating unique townships for that Ecodistrict ID
        unique_twnships=data['TWP_ID'].nunique()

        ## Store labels or response for modelling
        labels = data['YieldKgAcre']

        ## Store Features or predictors for modelling
        features= data.drop(['YieldKgAcre'], axis = 1)
        
        ## Splits the data into training and testing into 80-20 ratio
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)

        ## storing the scaled training predictor data after removing some of the unnessary columns like Township ID, Ecodistrict ID and Year not required for modelling purposes
        scaler = scale(train_features.loc[:, ~train_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Storing the original training predictor data into another variable before scaling
        train_index=train_features

        ## Storing the original testing predictor data into another variable before scaling
        test_index=test_features
        
        ## storing the scaled training predictor data after removing some of the unnessary columns like Township ID, Ecodistrict ID and Year not required for modelling purposes
        X_train_scaled = scale(train_features.loc[:, ~train_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])

        ## storing the scaled testing predictor data after removing some of the unnessary columns like Township ID, Ecodistrict ID and Year not required for modelling purposes
        X_test_scaled = scale(test_features.loc[:, ~test_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## The training set features scaled and transformed into principal components
        X_train_pc = pca.fit_transform(X_train_scaled)
        
        ## List of cummulative variances by all the principal components in %
        result = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

        ## Code to check number of principal components explaining atleast 95 % of variance in original dataset predictors
        for j in range(0, len(result)):
            if result[j] > 95:
                index = j
                variance = result[j]  ## variance explained by those selected principal components (must be greater than or equal to 95 %)
                break
                
        # Find optimal number of principal components
        best_pc_num = index + 1
        
        ## Subset of training principal components set for only selected components
        train_features = X_train_pc[:,:best_pc_num]
        
        ## The test set features scaled and transformed into principal components
        X_test_pc = pca.transform(X_test_scaled)

         ## Subset of training principal components set for only selected components
        test_features = X_test_pc[:,:best_pc_num]
        
        # Train model on training set of selected principal components
        lin_reg_pc = LinearRegression().fit(train_features, train_labels)
        
        # Calculating predicted values of the crop yield based on training predictors
        pred_train = lin_reg_pc.predict(train_features)
        
        # Calculating predicted values of the crop yield based on testing predictors
        pred = lin_reg_pc.predict(test_features)

        ## return necessary data points for further model evaluation and predictions
        return scaler,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance
        
class pcr_ecodistrict_model_validation_scoring(pcr_ecodistrict_model_fit):
    """
    A child class used to evaluate Principal Components Regression Model and make predictions having its certain attributes and the methods as well as attributes and methods of its parent class

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
        Evaluates the PCR in terms of MSE Train, Test, Mean Absolute Error and Accuracy
    
    predicted_train_dataset(additional=""):
        Predicts the Crop Yield Prediction Based on Training Predictors

    predicted_test_dataset(additional=""):
        Predicts the Crop Yield Prediction Based on Testing Predictors

    number_principal_components(additional=""):
        Returns the optimal number of principal components explaining atleast 95% of variance

    cummulative_explained_variance(additional=""):
        Return the exact variance explained by the principal components selected

    score(additional=""):
        Predicts the Crop Yield Prediction Based on any new data Predictors
    """
    
    def __init__(self,aafc_data, ecodistrict):
        ## Instatiating this class by instatiating the parent class
        pcr_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
        
    def validation_metrics(self):
        '''
        Evaluates Model Performance and prints MSE Train,Test,Mean Absolute Error and Accuracy

        Parameters:
                self: Class Object

        Returns:
                None
        '''

        ## using the data returened from the parent class function for model evaluation
        scaler,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)

        ## Define cross-validation folds for doing Training MSE
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        ## Calculating the Mean Squared Error for Training Set
        mse_train = -1 * cross_val_score(lin_reg_pc, train_features, train_labels, cv=cv, scoring='neg_root_mean_squared_error').mean()

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
        scaler,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        
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
        scalar,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        
        ## Storing the testing predicted values in the original test dataset
        test_index['Predicted_Yield']=pred

        return test_index

    def number_principal_components(self):
        '''
        Prints the optimal number of principal components explanining atleast 95% of the variance

        Parameters:
                self: Class Object

        Returns:
                None
        '''
        ## using the data returned from the parent class function for optimal number of principal components
        scalar,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        
        print("Number of Principal Components is:",best_pc_num)

    def cummulative_explained_variance(self):
        '''
        Prints the exact variance from optimal number of principal components

        Parameters:
                self: Class Object

        Returns:
                None
        '''

        ## using the data returned from the parent class function for exact variance
        scalar,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        
        print("Cummulative Explained Variance is:",round(variance,2),"%")

    def score(self,data):
        '''
        Returns the Predicted Crop Yield in the new Data Set

        Parameters:
                self: Class Object

        Returns:
                returns data i.e., dataset with predicted values of crop yield based on New Dataset predictors
        '''

        ## Instation Principal Component Model
        pca = PCA()

        ## using the data returned from the parent class function for any new data set prediction
        scaler,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        
        ## Removing unnecessary columns for modelling purposes
        features=data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])]

        ## scaled the new data set features or predictors
        features_scaled = scale(features)

        ## The training set features scaled and transformed into principal components and best principal components are chosen from that
        features_pc = pca.fit_transform(features_scaled)[:,:best_pc_num]
        
        ## Calculating prediction via model for new data set predictors
        predicted_yield = lin_reg_pc.predict(features_pc)

        ## Storing the predicted values in the new dataset
        data['Predicted_Yield']=predicted_yield

        return data