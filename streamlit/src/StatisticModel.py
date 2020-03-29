#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:46:43 2020

@author: sraj
"""

from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from datetime import datetime, timedelta,date
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
import statsmodels.api as sm
from scipy.stats import chi2_contingency, chi2
import streamlit as st


class StatisticModel():
    
    def __init__(self, data, columns, churnColumn = 'Exit_Marker'):
       self.data = data.copy()
       
       print(self.data.describe())
       self.churnColumn = churnColumn
       
       self.columnsFinal = []
       for column in self.data.columns:
           if column != self.churnColumn:
               self.columnsFinal.append(column)

       self.endog = self.data[self.churnColumn]
       self.glm_model = sm.GLM(self.endog , self.data[self.columnsFinal] ,\
            family=sm.families.Binomial())
    #    print(self.data.columns)
       
    
    @st.cache
    def glm(self):
        
        res = self.glm_model.fit()

        nobs = res.nobs
        print('self.endog.sum()  ', self.endog.sum())
        print(self.endog[self.endog == 1])
        
        y = self.endog/self.endog.sum(0)
        yhat = res.mu
        print(list(zip(self.endog, yhat)))
        # print(y, yhat)
        #score = self.glm_model(res.params)
        return res.summary(), res.params, res.tvalues, self.endog.sum(), y, yhat
    
    @staticmethod
    def correlation(contingancy_table, prob=0.95):
        chi, pValue, dof, expected = chi2_contingency(contingancy_table)
        
        alpha = 1 - prob
        critical = chi2.ppf(prob, dof)
        
        return chi, critical, pValue, alpha
