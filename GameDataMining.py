#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2022
"""
import json
import math
import time
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score,recall_score

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display
import webbrowser


class Modeling():
    def __init__(self):
        # False True
        self.SMOTES = False
        self.RandomizedSearchCV = False
        self.featurized_data_file_path = 'data_featurization.csv'
        
        self.Scaler = False
        self.ScalerVars = ['max_points']
        ### RF LGB
        self.model = 'LGB'
        # False True
        self.plot_importan = False
        self.plot_permutat_importan = False
        
        self.RF_original_params = {
            #  'n_estimators':100,
            # 'criterion':'gini',
            # "max_depth":5,
            "min_samples_split":2,
            "min_samples_leaf":2,
            "min_weight_fraction_leaf":0.0,
            # "max_features":'auto',
            "max_leaf_nodes":None,
            "min_impurity_decrease":0.0,
            "min_impurity_split":None,
            "bootstrap":True,
            "oob_score":False,
            "n_jobs":-1,
            "verbose":0,
            "warm_start":False,
            # "class_weight":'balanced'
        }
        
        self.RF_search_params = { 
            'n_estimators': [500,1000,],
            'max_features': ['auto','sqrt','log2'],
            'max_depth' : [4,5,6,7,8],
            'criterion' :['gini', 'entropy'],
        }
        self.RF_params = {
            'n_estimators':1000,
            # 'criterion':'gini',
            "max_depth":8,
            # "min_samples_split":2,
            # "min_samples_leaf":2,
            # "min_weight_fraction_leaf":0.0,
            "max_features":'auto',
            # "max_leaf_nodes":32,
            # "min_impurity_decrease":0.0,
            # "min_impurity_split":None,
            # "bootstrap":True,
            "oob_score":True,
            "n_jobs":-1,
            # "verbose":0,
            # "warm_start":False,
            "class_weight":'balanced',
            "max_samples": 0.5
        }
        
        self.LGB_params = {  
            'boosting_type': 'gbdt',  
            'objective': 'binary',  
            'metric':  'auc',  
            'verbose':-1,
            # 'num_leaves': 16,  ### could change but be careful about overfitting
            'max_depth': 8,  ### could change but be careful about overfitting
            # 'min_data_in_leaf': 450,  
            'learning_rate': 0.01,  ### could change but be careful about local optimization
            'feature_fraction': 0.4,  ### like random forest for its features to sample
            'bagging_fraction': 0.4,  ### like random forest for its samples to sample
            'bagging_freq': 1000,  ### how many times for sample
            'lambda_l1': 0.01,    ### L1 norm (lead to more zero coeff)
            'lambda_l2': 0.01,    ### L2 norm
            # 'is_unbalance': True # Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
            }  
        
        self.LGB_original_params = {  
            'boosting_type': 'gbdt',  
            'objective': 'binary',  
            'metric':  'auc',  
            'verbose':-1,
            # 'num_leaves': 16,  ### could change but be careful about overfitting
            # 'max_depth': 8,  ### could change but be careful about overfitting
            # 'min_data_in_leaf': 450,  
            'learning_rate': 0.01,  ### could change but be careful about local optimization
            # 'feature_fraction': 0.6,  ### like random forest for its features to sample
            # 'bagging_fraction': 0.6,  ### like random forest for its samples to sample
            # 'bagging_freq': 200,  ### how many times for sample
            # 'lambda_l1': 0.01,    ### L1 norm (lead to more zero coeff)
            # 'lambda_l2': 0.01,    ### L2 norm
            'is_unbalance': False # Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
            }  
        
        self.LGB_search_params = {
            'bagging_freq': range(500, 1300, 200),
            'min_child_weight': range(3, 20, 2),
            'colsample_bytree': np.arange(0.4, 1.0),
            'max_depth': range(4, 16, 2),
            'num_leaves':range(16, 64, 4),
            # 'subsample': np.arange(0.5, 1.0, 0.1),
            'feature_fraction': np.arange(0.1, 0.6, 0.1),
            'bagging_fraction': np.arange(0.1, 0.6, 0.1),
            'lambda_l1': np.arange(0.01, 0.1, 0.01),
            'lambda_l2': np.arange(0.01, 0.1, 0.01),
            # 'min_child_samples': range(10, 30)
            }
        
    
    def ComputeROCAUC(self,X,y,clf,index):
        y_predict = clf.predict_proba(X.iloc[index])[:,1]
        precisions, recalls, thresholds = precision_recall_curve(y.iloc[index], y_predict)
        precision = average_precision_score(y.iloc[index], y_predict)
        predict_label = [1 if i >= 0.5 else 0 for i in y_predict]
        precision = precision_score(y.iloc[index],predict_label)
        recall = recall_score(y.iloc[index],predict_label)
        auc_score = roc_auc_score(y.iloc[index], y_predict)
        # plot_roc_curve(clf, X.iloc[index], y.iloc[index])
        # plt.show()
        return precision, recall, auc_score
    
    def Scaling(self,df):
        for ScalerVar in self.ScalerVars:
            array_2 = np.array(df[ScalerVar]).reshape(-1, 1)
            df[ScalerVar] = RobustScaler().fit(array_2).transform(array_2)
        return df
            
    
    def TrainModel(self):
        df = pd.read_csv(self.featurized_data_file_path)
        df_1 = df.copy()
        df_1['login_days'].fillna(0,inplace=True)
        # df['retain'] = [1-i for i in list(df['churn'])]
        y = df['churn']
        if self.model == 'LGB' and not self.SMOTES:
            df_1 = df_1.drop(['unique_id','churn','sessions_total'],axis=1)
        else:
            df_1 = df_1.drop(['unique_id','churn','login_interval','sessions_total'],axis=1)
        if self.Scaler:
            df_1 = self.Scaling(df_1)
        X = df_1
        
        cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        results = pd.DataFrame(columns=['training_score', 'test_score'])
        precisions, recalls, scores = [], [], []
        
        # if self.model == 'RF':
        #     clf = RandomForestClassifier()
        # elif self.model == 'LGB':
        #     clf = lgb.LGBMClassifier()
        
        importance_record_dict = {}
            
        for (train, test), i in zip(cv.split(X, y), range(5)):
            if self.SMOTES:
                sm = SMOTE()
                X_res, y_res = sm.fit_resample(X.iloc[train], y.iloc[train])
            else:
                X_res, y_res = X.iloc[train], y.iloc[train]
            if self.RandomizedSearchCV:
                if self.model == 'RF':
                    rf_cv = RandomizedSearchCV(RandomForestClassifier(**self.RF_original_params), self.RF_search_params, cv= 5)
                    params = self.RF_original_params.copy()
                elif self.model == 'LGB':
                    rf_cv = RandomizedSearchCV(lgb.LGBMClassifier(**self.LGB_original_params), self.LGB_search_params, cv= 5)
                    params = self.LGB_original_params.copy()
                rf_cv.fit(X_res,y_res)
                best_params_ = rf_cv.best_params_
                params.update(best_params_)
                # print (params)
                if self.model == 'RF':
                    clf = RandomForestClassifier(**params)
                    
                elif self.model == 'LGB':
                    clf = lgb.LGBMClassifier(**params)
            else:
                if self.model == 'RF':
                    clf = RandomForestClassifier(**self.RF_params)
                    
                elif self.model == 'LGB':
                    clf = lgb.LGBMClassifier(**self.LGB_params)
                
            clf.fit(X_res, y_res)
            importance_list = [[X.columns[i],clf.feature_importances_[i]] for i in range(len(X.columns))]
            for importances in importance_list:
                if importances[0] not in importance_record_dict.keys():
                    importance_record_dict[importances[0]] = []
                else:
                    importance_record_dict[importances[0]].append(importances[1])
            # importance_list= sorted(importance_list, key = lambda x:x[1],reverse = True)
            # print (importance_list)
            _, _, auc_score_train = self.ComputeROCAUC(X,y,clf,train)
            precision, recall, auc_score = self.ComputeROCAUC(X,y,clf,test)
            scores.append((auc_score_train, auc_score,precision,recall))
            precisions.append(precision)
            recalls.append(recall)
        # plot_roc_curve(clf, X.iloc[test], y.iloc[test])
        
        performance_table = pd.DataFrame(scores, columns=['AUC Train', 'AUC Test','Precision Test','Recall Test'])
        performance_mean = performance_table.mean()
        performance_medain = performance_table.median()
        print ('\n')
        print ('*'*10)
        print ('\n')
        print(self.featurized_data_file_path)
        print(self.model)
        print(performance_table)
        print('performance:')
        print ('mean: \n',performance_mean)
        # print('median: \n',performance_medain)
        
        for key_ in importance_record_dict.keys():
            importances = importance_record_dict[key_]
            importance_record_dict[key_] = np.median(importances)
        
        importance_record_dict= dict(sorted(importance_record_dict.items(),key=lambda x:x[1],reverse=True))
        print (importance_record_dict)
        
    
        if self.model == 'LGB' and self.plot_importan:
            if self.RandomizedSearchCV:
                clf = lgb.LGBMClassifier(**params)
            else:
                clf = lgb.LGBMClassifier(**self.LGB_params)
            if self.SMOTES:
                X_res, y_res = sm.fit_resample(X, y)
            else:
                X_res, y_res = X, y
            clf.fit(X_res, y_res)

            # clf.feature_importances_(importance_type='gain')
            feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X.columns)), columns=['Value','Feature'])

            plt.figure(figsize=(20, 10))
            sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
            plt.title('LightGBM Features importances')
            plt.tight_layout()
            plt.savefig('lgbm_importances_split.png')
            plt.show()
            
            
        
        elif self.model == 'RF' and self.plot_importan:
            if self.RandomizedSearchCV:
                clf = RandomForestClassifier(**params)
            else:
                clf = RandomForestClassifier(**self.RF_params)
            if self.SMOTES:
                X_res, y_res = sm.fit_resample(X, y)
            else:
                X_res, y_res = X, y
            feature_names = X_res.iloc[test].columns.tolist()
            clf.fit(X_res, y_res)
            importances = clf.feature_importances_
            std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
            forest_importances = pd.Series(importances, index=feature_names)
            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=std, ax=ax)
            ax.set_title("Feature importances using MDI")
            ax.set_ylabel("Mean decrease in impurity")
            fig.tight_layout()
            plt.savefig('RF_plot_MDI_importan.png')
            # plt.show()
            
        if self.model == 'RF' and self.plot_permutat_importan:
            if self.RandomizedSearchCV:
                clf = RandomForestClassifier(**params)
            else:
                clf = RandomForestClassifier(**self.RF_params)
            if self.SMOTES:
                X_res, y_res = sm.fit_resample(X.iloc[train], y.iloc[train])
            else:
                X_res, y_res = X.iloc[train], y.iloc[train]
            feature_names = X_res.columns.tolist()
            clf.fit(X_res, y_res)
            result = permutation_importance(clf, X.iloc[test], y.iloc[test], n_repeats=10, n_jobs=2)
            forest_importances = pd.Series(result.importances_mean, index=feature_names)
            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
            ax.set_title("Feature importances using permutation on full model")
            ax.set_ylabel("Mean accuracy decrease")
            fig.tight_layout()
            plt.savefig('RF_plot_permutat_importan.png')
            # plt.show()
                
            
        
        if self.model == 'LGB' and self.plot_permutat_importan:
            if self.RandomizedSearchCV:
                clf = lgb.LGBMClassifier(**params)
            else:
                clf = lgb.LGBMClassifier(**self.LGB_params)
            if self.SMOTES:
                X_res, y_res = sm.fit_resample(X.iloc[train], y.iloc[train])
            else:
                X_res, y_res = X.iloc[train], y.iloc[train]
            clf.fit(X_res, y_res)
            permutation_importan = PermutationImportance(clf)
            permutation_importan.fit(X.iloc[test], y.iloc[test])
            print(f"Permutation importance for fold 5")
            
            # eli5.format_as_dataframe(eli5.explain_weights(permutation_importan, feature_names=X.iloc[test].columns.tolist()))
            
            html_obj = eli5.show_weights(permutation_importan, feature_names = X.iloc[test].columns.tolist())
            # Write html object to a file (adjust file path; Windows path is used here)
            with open('permutation_importan.htm','wb') as f:
                f.write(html_obj.data.encode("UTF-8"))

            # Open the stored HTML file on the default browser
            url = r'permutation_importan.htm'
            webbrowser.open(url, new=2)
        

        
class FeatureEngineering():
    def __init__(self):
        self.cleaned_data_file_path = 'data_clean.csv'
        self.output_name = 'data_featurization.csv'
        # sum median mean maximum quantile
        self.TimeWindowYN = False
        self.TimeWindow = 20
        self.FeaturizationMethod = 'mean'
        self.quantile = 0.75
        self.nominals = ['unique_id','churn','registration_platform','marketing_source']
        self.OnehotColumns = ['registration_platform','marketing_source']
        # True False
        self.UOD = False
        self.UODTreshold = 5
        self.MOD = False
        # IF LOF
        self.UODMethod = 'IF'
        self.contamination = 'auto'
        self.n_neighbors = 10
        self.OFT = 5
        
    def Aggregate(self,feature_list):
        if self.TimeWindowYN:
            feature_list = feature_list[-self.TimeWindow:]
        if self.FeaturizationMethod == 'mean':
            return round(np.nanmean(feature_list),4)
        elif self.FeaturizationMethod == 'sum':
            return round(np.nansum(feature_list),4)
        elif self.FeaturizationMethod == 'median':
            return round(np.nanmedian(feature_list),4)
        elif self.FeaturizationMethod == 'maximum':
            return round(np.nanmax(feature_list),4)
        elif self.FeaturizationMethod == 'quantile':
            return round(np.nanquantile(feature_list,self.quantile),4)
    
    def DetectMatrix(self,df_final):
        df_2 = df_final.copy()
        array_2 = df_2.drop(self.nominals, axis=1).to_numpy()
        array_dect = RobustScaler().fit(array_2).transform(array_2)
        return df_2,array_dect
    
    def UnivariateOutlierProcess(self,df_final):
        df_2,array_dect = self.DetectMatrix(df_final)
        
        count_outlying = 0
        for i in range(array_dect.shape[0]):
            for j in range(array_dect.shape[1]):
                if abs(array_dect[i,j]) > self.UODTreshold:
                    df_2.iloc[i, j+len(self.nominals)] = np.nan    
                    count_outlying += 1
        print ("outlying cell num:", count_outlying)
        return df_2
    
    def MultivariateOutlierProcess(self,df_final):
        df_2,array_dect = self.DetectMatrix(df_final)
        if self.UODMethod == 'IF':
            clf = IsolationForest(contamination=self.contamination).fit(array_dect)
            y_pred = clf.predict(array_dect)
            pred_outliers = [i for i in range(len(y_pred)) if y_pred[i] == -1]
            
        elif self.UODMethod == 'LOF':
            clf = LocalOutlierFactor(n_neighbors=self.n_neighbors)
            y_pred_label = clf.fit_predict(array_dect)
            y_pred = clf.negative_outlier_factor_     
            pred_outliers = [i for i in range(len(y_pred)) if abs(y_pred[i]) > self.OFT]
            
        print (self.UODMethod, 'detect outlier num:',len(pred_outliers))
        if len(pred_outliers) >0:
            df_2 = df_2.drop(pred_outliers)
            df_2.reset_index(inplace=True, drop=True)
        
        return df_2
    
    def Featurization(self):
        original_df = pd.read_csv(self.cleaned_data_file_path)
        df_1 = original_df.copy()
        df_1['unique_id'] = df_1['unique_id'].apply(str)
        unique_ids = list(set(df_1['unique_id']))
        
        vars = ['transactions','Increase_points',\
                'eventcount_message','eventcount_fight','eventcount_trade','eventcount_build',\
                'eventcount_recruit','eventcount_ally','eventcount_research']
        
        final_vars = ['unique_id','churn','registration_platform','marketing_source','max_points','sessions_total',\
                    'start_interval','login_interval','recency', 'login_days','login_frequency', 'agg_session_duration','agg_eventcount_total']
        for var in vars:
            final_vars.append('rate_'+var)
            final_vars.append('agg_'+var)
        
        df_final = pd.DataFrame(columns=[final_vars])
        
        for unique_id_idx in tqdm(range(len(unique_ids))):
            unique_id = unique_ids[unique_id_idx]
            sub_df = df_1[df_1['unique_id'] == unique_id]
            # basic
            registration_platform = list(sub_df['registration_platform'])[0]
            marketing_source = list(sub_df['marketing_source'])[0]
            max_points = list(sub_df['max_points'])[0]
            sessions_total = list(sub_df['sessions_total'])[0]
            
            # label
            active_four_weeks = list(sub_df['active_four_weeks'])[0]
            if not active_four_weeks:
                churn = 1
            else:
                churn = 0
            # R
            date_registered = list(sub_df['date_registered'])[0]
            date_registered_timestamp = int(time.mktime(time.strptime(date_registered, "%Y-%m-%d %H:%M:%S")))
            first_dates = list(sub_df['first_date'])
            first_dates_day = [i[:10] for i in first_dates]
            login_day_list = list(set(first_dates_day))
            login_days = len(login_day_list)
            
            sub_counts = []
            sub_interval = []
            for login_day in login_day_list:
                sub_count = first_dates_day.count(login_day)
                sub_counts.append(sub_count)
            login_day_list.sort()
            for i in range(len(login_day_list)-1):
                day_1 = int(time.mktime(time.strptime(login_day_list[i], "%Y-%m-%d")))
                day_2 = int(time.mktime(time.strptime(login_day_list[i+1], "%Y-%m-%d")))
                sub_interval.append((day_2 - day_1)/86400)
            login_interval = round(np.mean(sub_interval),4)
            login_frequency = round(np.mean(sub_counts),4)
            recent_date = first_dates[-1]
            primary_date = first_dates[0]
            recent_date_timestamp = int(time.mktime(time.strptime(recent_date, "%Y-%m-%d %H:%M:%S")))
            primary_date_timestamp = int(time.mktime(time.strptime(primary_date, "%Y-%m-%d %H:%M:%S")))
            recency = round((date_registered_timestamp + 86400*28 - recent_date_timestamp)/(60*60*24),4)
            start_interval = round((primary_date_timestamp - date_registered_timestamp)/(60),4)
            # F and M
            agg_session_duration = self.Aggregate([round(i/60,4) for i in list(sub_df['session_duration'])])
            agg_eventcount_total = self.Aggregate(list(sub_df['eventcount_total']))
            values_list = [unique_id,churn,registration_platform,marketing_source,max_points,sessions_total,\
                            start_interval,login_interval,recency, login_days,login_frequency, \
                            agg_session_duration, agg_eventcount_total]
            for var in vars:
                increment_list = list(sub_df[var])
                fre_rate = round(len([i for i in increment_list if i > 0]) / len(increment_list),4)
                values_list.append(fre_rate)
                values_list.append(self.Aggregate(increment_list))
                
            
            df_final.loc[len(df_final)] = values_list
        
        ### 2. remove outliers   
        if self.UOD:
            df_final = self.UnivariateOutlierProcess(df_final)
        
        if self.MOD:
            df_final = df_final.drop(['login_interval'],axis = 1)
            df_final = self.MultivariateOutlierProcess(df_final) 
        
        
        ### 3. one hot encoding
        for OnehotColumn in self.OnehotColumns:
            OnehotColumnList = [i[0] for i in df_final[[OnehotColumn]].values]
            for unique_value in set(OnehotColumnList):
                df_final['_'.join([OnehotColumn,unique_value])] = [1 if i == unique_value else 0 for i in OnehotColumnList]
        df_final.drop(self.OnehotColumns,axis=1,inplace=True)
        
        
        df_final.to_csv(self.output_name,index=False)
        
def Preprocessing():
    file_name_path = 'data_example.csv'
    
    original_df = pd.read_csv(file_name_path)
    df_1 = original_df.copy()
    df_1['unique_id'] = df_1['unique_id'].apply(str)
    
    
    
    # print (df_1)
    # 1. delete the overlap sub sessions.
    last_dates = [int(time.mktime(time.strptime(i, "%Y-%m-%d %H:%M:%S"))) if type(i) != float else i for i in df_1['last_date']]
    df_1['last_date_2'] = last_dates
    df_1['Lag_last_date'] = df_1.groupby(['unique_id'])['last_date_2'].shift(1)
    Lag_last_dates = list(df_1['Lag_last_date'])
    first_dates = [ int(time.mktime(time.strptime(i, "%Y-%m-%d %H:%M:%S"))) for i in df_1['first_date']]
   
    df_1['check_session_overlap'] =  [first_dates[i] - Lag_last_dates[i] for i in range(len(first_dates))]
    ##394654 322508
    df_1 = df_1[(df_1['check_session_overlap']>=0) | (df_1['last_date_2']>df_1['Lag_last_date'])]
    
    
    # 2. Increase_points
    df_1['Current_points'] = df_1['max_points']
    df_1['Lag_points'] = df_1.groupby(['unique_id'])['max_points'].shift(1)
    df_1['Lag_points'] = df_1['Lag_points'].apply(lambda x: 0 if np.isnan(x) else x)
    df_1['Increase_points'] = df_1['max_points'] - df_1['Lag_points']
    df_1 = df_1[(df_1['Increase_points'] >= 0)]
    
    # 3. Fill in the null for date_last_login and lifetime_logindays
    df_1['date_last_login'].fillna(df_1['last_date'][:10], inplace=True)
    last_date = [i[:10] for i in list(df_1['last_date'])]
    date_last_login = list(df_1['date_last_login'])
    df_1['date_last_login'] = [last_date[i] if type(date_last_login[i]) == float else date_last_login[i] for i in range(len(date_last_login))]
    df_1['lifetime_logindays'] = df_1['lifetime_logindays'].apply(lambda x: 1 if x == 0 else x)
    # 4. total count for the behaviors in the data is larger than the recorded total count
    df_1['eventcount_total_2'] =  df_1['eventcount_message']+df_1['eventcount_fight']+ \
                                    df_1['eventcount_trade']+df_1['eventcount_build']+df_1['eventcount_recruit']+ \
                                    df_1['eventcount_ally']+df_1['eventcount_research']
    
    df_1['eventcount_total'] = df_1[["eventcount_total", "eventcount_total_2"]].max(axis=1)
    
    # 5. drop quest_closed and eventcount_total_2
    df_1.drop(['quest_closed', 'eventcount_total_2','Lag_points'], axis=1,inplace=True)
    
    
    # 6. Getting the first 30 sessions
    df_1['session_number'] = df_1.sort_values(['first_date'], ascending=[True]).groupby(['unique_id']).cumcount() + 1
    df_1 = df_1[(df_1['session_number']<= 30)]
    
    
    # 7. sessions_total and max_points 
    df_1['sessions_total'] = df_1.groupby('unique_id')['sessions_total'].transform('count')
    # df_1 = df_1[(df_1['sessions_total']== 30)]
    df_1['max_points'] = df_1.groupby(['unique_id'])['max_points'].transform(max)
    # df_1['regis_platform'] = df_1['registration_platform'].apply(lambda x: x if x == 'Browser' else 'Mobile')
    
    df_1.drop(['last_date_2', 'Lag_last_date','check_session_overlap'], axis=1,inplace=True)
    
    df_1.to_csv('data_clean.csv',index=False)


def main():
    Preprocessing()
    
    FEApp = FeatureEngineering()
    FEApp.Featurization()
    
    App = Modeling()
    App.TrainModel()


if __name__ == '__main__':
    main()