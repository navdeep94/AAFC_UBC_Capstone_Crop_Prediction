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
        
        model = LassoCV(cv=5, random_state=0, max_iter=10000)
        model.fit(train_features, train_labels)
        lasso_best = Lasso(alpha=model.alpha_)
        lasso_best.fit(train_features, train_labels)
        
        pred_train = lasso_best.predict(train_features)
        pred = lasso_best.predict(test_features)
        return scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list

    
class lasso_ecodistrict_model_validation_scoring(lasso_ecodistrict_model_fit):
    
    def __init__(self,aafc_data, ecodistrict):
        lasso_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
        
    def validation_metrics(self):
        scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        r2train=round(lasso_best.score(train_features, train_labels)*100, 2)
        r2test=round(lasso_best.score(test_features, test_labels)*100, 2)
        
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
        scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        train_index['Predicted_Yield']=pred_train
        return train_index
    def feature_importance(self):
        scalar,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        coefficients=lasso_best.coef_
        importance = np.abs(coefficients)
        
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
        scalar,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        test_index['Predicted_Yield']=pred
        return test_index
    def score(self,data):
        scaler,lasso_best,train_features,test_features,train_labels,test_labels,train_index,test_index,pred_train,pred,feature_list=lasso_ecodistrict_model_fit.model_train_test(self)
        features=scaler.transform(data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        predicted_yield = lasso_best.predict(features)
        data['Predicted_Yield']=predicted_yield
        return data
        
