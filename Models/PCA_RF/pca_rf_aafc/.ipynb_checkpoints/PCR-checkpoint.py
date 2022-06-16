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
    
    def __init__(self,aafc_data, ecodistrict):
        self.aafc_data=aafc_data
        self.ecodistrict=ecodistrict
        
    def model_train_test(self):
        pca = PCA()
        data=self.aafc_data[self.aafc_data['ECODISTRICT_ID']==self.ecodistrict]
        records=len(data)
        unique_twnships=data['TWP_ID'].nunique()
        labels = data['YieldKgAcre']
        features= data.drop(['YieldKgAcre'], axis = 1)
        
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)
        scaler = scale(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        train_index=train_features
        test_index=test_features
        
        X_train_scaled = scale(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        X_test_scaled = scale(test_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        X_train_pc = pca.fit_transform(X_train_scaled)

        result = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100) ## List of cummulative variances by all the principal components in %
        for j in range(0, len(result)):
            if result[j] > 95:
                index = j
                variance = result[j]
                break
                
        # determine optimal number of principal components
        best_pc_num = index + 1
        
        train_features = X_train_pc[:,:best_pc_num]
        
        X_test_pc = pca.transform(X_test_scaled)
        test_features = X_test_pc[:,:best_pc_num]
        
        # Train model on training set
        lin_reg_pc = LinearRegression().fit(train_features, train_labels)
        
        # Predict on train data
        pred_train = lin_reg_pc.predict(train_features)
        
        # Predict on test data
        pred = lin_reg_pc.predict(test_features)
        return scaler,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance
        
class pcr_ecodistrict_model_validation_scoring(pcr_ecodistrict_model_fit):
    
    def __init__(self,aafc_data, ecodistrict):
        pcr_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
        
    def validation_metrics(self):
        scaler,lin_reg_pc,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)

        # Define cross-validation folds
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Training data
        mse_train = -1 * cross_val_score(lin_reg_pc, train_features, train_labels, cv=cv, scoring='neg_root_mean_squared_error').mean()
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
        scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        train_index['Predicted_Yield']=pred_train
        return train_index
        
    def predicted_test_dataset(self):
        scalar,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        test_index['Predicted_Yield']=pred
        return test_index

    def number_principal_components(self):
        scalar,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        print("Number of Principal Components is:",best_pc_num)

    def cummulative_explained_variance(self):
        scalar,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        print("Cummulative Explained Variance is:",round(variance,2),"%")

    def score(self,data):
        pca = PCA()
        scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,best_pc_num,variance=pcr_ecodistrict_model_fit.model_train_test(self)
        features=data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])]
        features_scaled = scale(features)
        features_pc = pca.fit_transform(features_scaled)[:,:best_pc_num]
        
        predicted_yield = lasso_best.predict(features_pc)
        data['Predicted_Yield']=predicted_yield
        return data