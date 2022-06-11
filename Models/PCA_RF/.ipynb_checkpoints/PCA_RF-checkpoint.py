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


class pcr_rf_ecodistrict_model_fit():
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
        scaler = StandardScaler().fit(train_features.loc[:, ~train_features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])

        train_index=train_features
        test_index=test_features
        
        X_train_scaled = scale(train_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        X_test_scaled = scale(test_features.loc[:, ~features.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        
        X_train_pc = pca.fit_transform(X_train_scaled)

        result = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100) ## List of cummulative variances by all the principal components in %
        for j in range(0, len(result)):
            if result[j] > 98:
                index = j
                variance = result[j]
                break
                
        # determine optimal number of principal components
        best_pc_num = index + 1
        
        
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        rf.fit(X_train_pc[:,:best_pc_num], train_labels);
        X_test_pc = pca.transform(X_test_scaled)[:,:best_pc_num]
        pred_train = rf.predict(X_train_pc[:,:best_pc_num])
        pred_test = rf.predict(X_test_pc[:,:best_pc_num])

        return pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance

class pcr_rf_ecodistrict_model_validation_scoring(pcr_rf_ecodistrict_model_fit):
    
    def __init__(self,aafc_data, ecodistrict):
        pcr_rf_ecodistrict_model_fit.__init__(self,aafc_data, ecodistrict)
    
    def validation_metrics(self):
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pcr_rf_ecodistrict_model_fit.model_train_test(self)
        
        mse_train = round(mean_squared_error(train_labels, pred_train,squared=False),2)
        # Test data
        mse_test =round(mean_squared_error(test_labels, pred_test,squared=False),2)
        
        
 
        errors = abs(pred_test - test_labels)

        mae=round(np.mean(errors), 2)
        

        mape = 100 * (errors / test_labels)
        # Calculate and display accuracy
        accuracy = round(100 - np.mean(mape),2)
        
        print("Mean Squared Error Train: ",mse_train)
        print("Mean Squared Error Test: ",mse_test)

        print("Mean Absolute Error: ",mae)
        print("Accuracy: ",accuracy)
    
    def predicted_train_dataset(self):
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pcr_rf_ecodistrict_model_fit.model_train_test(self)
        train_index['Predicted_Yield']=pred_train
        return train_index
    
    def predicted_test_dataset(self):
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pcr_rf_ecodistrict_model_fit.model_train_test(self)
        test_index['Predicted_Yield']=pred_test
        return test_index
    
    def number_principal_components(self):
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pcr_rf_ecodistrict_model_fit.model_train_test(self)
        print("Number of Principal Components is:",best_pc_num)

    def cummulative_explained_variance(self):
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pcr_rf_ecodistrict_model_fit.model_train_test(self)
        print("Cummulative Explained Variance is:",round(variance,2),"%")
    
    def score(self,data):
        pca,scaler,rf,train_labels,test_labels,train_index,test_index,pred_train,pred_test,best_pc_num,variance=pcr_rf_ecodistrict_model_fit.model_train_test(self)
        features=scaler.transform(data.loc[:, ~data.columns.isin(['TWP_ID', 'ECODISTRICT_ID', 'YEAR'])])
        pca_for_scoring_df = pca.transform(features)[:,:best_pc_num]
        predicted_yield = rf.predict(pca_for_scoring_df)
        data['Predicted_Yield']=predicted_yield        
        return data

    
