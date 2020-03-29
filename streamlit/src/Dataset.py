#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:33:06 2020

@author: sraj
"""
import pandas as pd
import numpy as np

class Dataset():

    def __init__(self, path):
        self.path = path
       
    
    def loadData(self):
        
        data = pd.read_csv(self.path)
        
        #data = data[data.last_name != '']
        data = data.drop(columns='fullvisitorID')
        data = data.sort_values(by='start_')
        data['start_'] = data['start_'].astype('datetime64')
        data['end_'] = data['end_'].astype('datetime64')
        data['date'] = data['date'].astype('datetime64')
        data['trial_end_'] = data['trial_end_'].astype('datetime64')
        # data['resource_name'] = data['resource_name'].astype("category")
        # data['pageCategory'] = data['pageCategory'].astype("category")
        # data['deviceCategory'] = data['deviceCategory'].astype("category")
        # data['source'] = data['source'].astype("category")
        # data['term_name'] = data['term_name'].astype("category")
        # data['status'] = data['status'].astype("category")
        data['auto_renew_enabled'] = data['auto_renew_enabled'].astype(int)
        data['renewed'] = data['renewed'].astype(int)
        
        
        return data
    
    def loadAggData(self, data, columns = ['user_id_uid', 'start_', 'end_', 'paymentCycle_Marker', 'source']):
        
        dateColumns = columns.copy()
        dateColumns.append('date')

        agg_data = data[data['Phase_Marker'].isin([2, 3]) & data['paymentCycle_Marker'].notnull()]\
                  .loc[:, dateColumns]\
                  .groupby(columns)\
                  .agg({'date':"count"})
                  
        churn_agg_data = data.groupby(columns).Exit_Marker.mean().reset_index()
        churn_agg_data = churn_agg_data.sort_values('Exit_Marker', ascending=False)
              
        return agg_data, churn_agg_data
    
    def contigancyData(self, data, column):
        
        contingancy_table = pd.crosstab(index=data[column], columns=data['Exit_Marker'])
        return contingancy_table 
    
    def getFatures(self, data, columns, num=None):
        
        fatured_data = pd.get_dummies(data.loc[:, columns])
        if num is not None:
            fatured_data = fatured_data.head(num)
            fatured_data = fatured_data.loc[:, (fatured_data != 0).any(axis=0)]

        return fatured_data
    
    def columnList(self):
        columnList = ['auto_renew_enabled',
                    'term_name', 
                    'topic_signed_on', 
                    'source', 
                    'type', 
                    'pageCategory',
                   'deviceCategory',
                   'country']
        return columnList
    
    def columnListInd(self):
        columnListGraph = ['source', 
                    'type', 
                    'pageCategory',
                   'deviceCategory']
        columnListText = ['auto_renew_enabled',
                    'term_name', 
                    'topic_signed_on', 'country','Exit_Marker']
        
        return columnListGraph, columnListText
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
    