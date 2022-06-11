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
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class feed_forward_ecodistrict_model_fit():
    
    def __init__(self,aafc_data, ecodistrict):
        self.aafc_data=aafc_data
        self.ecodistrict=ecodistrict
        
    def model_train_test(self):
        data=self.aafc_data[self.aafc_data['ECODISTRICT_ID']==self.ecodistrict]
        records=len(data)
        unique_twnships=data['TWP_ID'].nunique()
        labels = data['YieldKgAcre']
        features= data.drop(['YieldKgAcre'], axis = 1)
        feature_list = list(features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])].columns)
        
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)
        scaler = StandardScaler().fit(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        train_index=train_features
        test_index=test_features
        
        
        train_features = scaler.transform(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        test_features = scaler.transform(test_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        # Defines "deep" model and its structure
        # 10 layers
        # 15 output 
        model = Sequential()
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
        model.compile(Adam(lr=0.003), 'mean_squared_error')
    
        #ADAM is the optimizer, Loss function = mean squared error, Learning rate

        # Pass several parameters to 'EarlyStopping' function and assigns it to 'earlystopper'.
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    
        # Early stopping is controling the number of epochs, till the model is performing well. 
        # val_loss: Validation loss: loss or cost which we want to monitor. (error during validation)
        # min delta is used to tell what's the minimum cost to terminate the number of epochs. (How much a cost should improve in an epoch before  
        # it terminates). 
        # patience: allows the cost to not decrese for a certain number of epochs

        # Fits model over 2000 iterations with 'earlystopper' callback, and assigns it to history.
        history = model.fit(train_features, train_labels, epochs = 2000, validation_split = 0.2,shuffle = True, verbose = 0, callbacks = [earlystopper])
        # Verbose: controls whether or not certain things printed at each epoch.                    
        ff_best = model
        pred_train = ff_best.predict(train_features)
        pred = ff_best.predict(test_features)
        return scaler,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list

    
class feed_forward_ecodistrict_model_validation_scoring(feed_forward_ecodistrict_model_fit):
    
    def __init__(self,aafc_data, ecodistrict):
        feed_forward_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
        
    def validation_metrics(self):
        scaler,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=feed_forward_ecodistrict_model_fit.model_train_test(self)

        
        # Training data
        
        mse_train = round(mean_squared_error(train_labels, pred_train,squared=False),2)
        train_index['Predicted_Yield']=pred_train
        
        # Test data
        
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
        scaler,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=feed_forward_ecodistrict_model_fit.model_train_test(self)
        train_index['Predicted_Yield']=pred_train
        return train_index
    

    def predicted_test_dataset(self):
        scalar,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=feed_forward_ecodistrict_model_fit.model_train_test(self)
        test_index['Predicted_Yield']=pred
        return test_index
    
    def score(self,data):
        scaler,ff_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=feed_forward_ecodistrict_model_fit.model_train_test(self)
        features=scaler.transform(data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        predicted_yield = ff_best.predict(features)
        data['Predicted_Yield']=predicted_yield
        return data
        
    
    
# -


