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
# Import all needed libraries and sublibraries
import numpy as np 
import pandas as pd    
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class feed_forward_ecodistrict_model_fit():
    """
    A class representing Fully Connected Feedforward Neural Network model having certain attributes and the methods.
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
        
        ## Defines "deep" model and its structure
        ## 10 layers and 161 outputs
        model = Sequential()
        
        ## defining 1st layer and then other layers
        ## activation function ReLU
        model.add(Dense(161, input_shape=(161,), activation='relu'))
        model.add(Dense(161, activation='relu'))
        model.add(Dense(161, activation='relu'))
        model.add(Dense(161, activation='relu'))
        model.add(Dense(161, activation='relu'))
        model.add(Dense(161, activation='relu'))
        model.add(Dense(161, activation='relu'))
        model.add(Dense(161, activation='relu'))
        model.add(Dense(161, activation='relu'))    
        model.add(Dense(1,))
        
        ## Optimization using ADAM optimizer
        model.compile(Adam(lr=0.003), 'mean_squared_error')
    
        ##ADAM is the optimizer, Loss function = mean squared error, Learning rate
        ## Pass several parameters to 'EarlyStopping' function and assigns it to 'earlystopper'.
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    
        ## Early stopping is controling the number of epochs, till the model is performing well. 
        ## val_loss: Validation loss: loss or cost which we want to monitor. (error during validation)
        ## min delta is used to tell what's the minimum cost to terminate the number of epochs. (How much a cost should improve in an epoch before it terminates). 
        ## patience: allows the cost to not decrese for a certain number of epochs
        
        
        ## Fits model over 2000 iterations with 'earlystopper' callback, and assigns it to history.
        ## Verbose: controls whether or not certain things printed at each epoch.
        history = model.fit(train_features, train_labels, epochs = 2000, validation_split = 0.2,shuffle = True, verbose = 0, callbacks = [earlystopper])
                            
        ## storing model in another variable name    
        ff_best = model
        
        ## Calculating predicted values of the crop yield based on training predictors
        pred_train = ff_best.predict(train_features)
        
        ## Calculating predicted values of the crop yield based on testing predictors
        pred = ff_best.predict(test_features)
        
        ## returns necessary values which is to be used for model evaluation and predictions
        return scaler,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list

    
class feed_forward_ecodistrict_model_validation_scoring(feed_forward_ecodistrict_model_fit):
    """
    A child class used to evaluate Fully Connected Feedforward Neural Network Model and make predictions having its certain attributes and the methods as well as attributes and methods of its parent class
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
        feed_forward_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
        
    def validation_metrics(self):
        '''
        Evaluates Model Performance and prints MSE Train,Test,Mean Absolute Error and Accuracy
        Parameters:
                self: Class Object
        Returns:
                None
        '''
        
        
        ## using the data returened from the parent class function for model evaluation        
        scaler,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=feed_forward_ecodistrict_model_fit.model_train_test(self)

        
        # Training data
        ## Calculating the Mean Squared Error for Training Set
        mse_train = round(mean_squared_error(train_labels, pred_train,squared=False),2)
        
        ## Storing the training predicted values in the original training dataset
        train_index['Predicted_Yield']=pred_train
        
        # Test data
        ## Calculating the Mean Squared Error for Test Set
        mse_test =round(mean_squared_error(test_labels, pred,squared=False),2)
        
        #Calculate the absolute errors
        pred = np.squeeze(pred)
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
        '''
        Returns the Predicted Crop Yield in the Training Set
        Parameters:
                self: Class Object
        Returns:
                returns train_index i.e., dataset with predicted values of crop yield based on training set predictors
        '''
        
        
        ## using the data returned from the parent class function for train set prediction  
        scaler,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=feed_forward_ecodistrict_model_fit.model_train_test(self)        
        
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
        scalar,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=feed_forward_ecodistrict_model_fit.model_train_test(self)
        
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
        scaler,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=feed_forward_ecodistrict_model_fit.model_train_test(self)
        
        ## scaled the new data set features or predictors
        features=scaler.transform(data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        ## Calculating prediction via model for new data set predictors
        predicted_yield = ff_best.predict(features)
        
        ## Storing the predicted values in the new dataset
        data['Predicted_Yield']=predicted_yield
        
        return data
        
    
    
# -


