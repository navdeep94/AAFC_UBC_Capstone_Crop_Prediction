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
from sklearn.ensemble import RandomForestRegressor


class pca_rf_ecodistrict_model_fit():
    """
    A class representing PCA+RF model having certain attributes and the methods
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
        
        ## Standardizing the features
        scaler = StandardScaler().fit(train_features.loc[:, ~train_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Storing the original training and test predictor data into another variable before scaling    
        train_index=train_features
        test_index=test_features
        
        ## storing the scaled training and test predictor data after removing the columns like Township ID, Ecodistrict ID and Year which are not required for modelling purposes
        X_train_scaled = scale(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        X_test_scaled = scale(test_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## The training set features scaled and transformed into principal components
        X_train_pc = pca.fit_transform(X_train_scaled)
        
        ## List of cummulative variances by all the principal components in %
        result = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100) 
        
        ## Code to check number of principal components explaining atleast 98 % of variance in original dataset predictors
        for j in range(0, len(result)):
            if result[j] > 98:
                index = j
                variance = result[j]
                break
                
        ## Find optimal number of principal components
        best_pc_num = index + 1
        
        ## Defining the random forest model 
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        
        ## Fitting the random forest model for just the PCs summing to 98% of total variance
        rf.fit(X_train_pc[:,:best_pc_num], train_labels);
        
        ## The test set features scaled and transformed into principal components
        X_test_pc = pca.transform(X_test_scaled)[:,:best_pc_num]
        
        ## Predicting the yield for train and test datasets
        pred_train = rf.predict(X_train_pc[:,:best_pc_num])
        pred_test = rf.predict(X_test_pc[:,:best_pc_num])
        
        ## return necessary data points for further model evaluation and predictions

        return pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance

class pca_rf_ecodistrict_model_validation_scoring(pca_rf_ecodistrict_model_fit):
    """
    A child class used to evaluate PCA+RF Model and make predictions having its own attributes and the methods, as well as attributes and methods of its parent class
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
        Evaluates PCA+RF in terms of MSE Train, Test, Mean Absolute Error and Accuracy
    
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
        pca_rf_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
    
    def validation_metrics(self):
        '''
        Evaluates Model Performance and prints MSE Train,Test,Mean Absolute Error and Accuracy
        Parameters:
                self: Class Object
        Returns:
                None
        '''
        ## using the data returned from the parent class function for model evaluation
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pca_rf_ecodistrict_model_fit.model_train_test(self)
        
        ## Calculating the MSE for training data
        mse_train = round(mean_squared_error(train_labels, pred_train,squared=False),2)
        ## Calculating the MSE for test data
        mse_test =round(mean_squared_error(test_labels, pred_test,squared=False),2)
        
        ## Calculating the absolute error value
        errors = abs(pred_test - test_labels)
        
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
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pca_rf_ecodistrict_model_fit.model_train_test(self)
        
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
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pca_rf_ecodistrict_model_fit.model_train_test(self)
        ## Storing the testing predicted values in the original test dataset
        test_index['Predicted_Yield']=pred_test
        return test_index
    
    def number_principal_components(self):
        '''
        Prints the optimal number of principal components explanining atleast 98% of the variance
        Parameters:
                self: Class Object
        Returns:
                None
        '''
        ## using the data returned from the parent class function for optimal number of principal components
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pca_rf_ecodistrict_model_fit.model_train_test(self)
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
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pca_rf_ecodistrict_model_fit.model_train_test(self)
        print("Cummulative Explained Variance is:",round(variance,2),"%")
    
    def score(self,data):
        '''
        Returns the Predicted Crop Yield in the new Data Set
        Parameters:
                self: Class Object
        Returns:
                returns data i.e., dataset with predicted values of crop yield based on New Dataset predictors
        '''
        ## using the data returned from the parent class function for any new data set prediction
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pca_rf_ecodistrict_model_fit.model_train_test(self)
        
        ## Removing the columns not required for modelling purpose and scaling the same
        features=scaler.transform(data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Performing PCA and retaining the number of PCs as derived from fitting the training data set capturing 98% of variance
        pca_for_scoring_df = pca.transform(features)[:,:best_pc_num]
        
        ##Feeding the output from PCA to Random Forest model and predicting the yield
        predicted_yield = rf.predict(pca_for_scoring_df)
        
        ## Storing the predicted values in the new dataset
        data['Predicted_Yield']=predicted_yield        
        return data

    
