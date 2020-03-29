#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:39:00 2020

@author: sraj
"""

from Dataset import Dataset
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

from StatisticModel import StatisticModel


class Visualizer():
    
    def __init__(self, dataPath):
        st.title('Customer details')
        self.page = st.sidebar.selectbox("Choose a view", ["All", "Individual", "Model"])
        self.dataset = Dataset(dataPath)
    
    def draw(self): 
        """
        

        Returns
        -------
        None.

        """
        data = self.dataset.loadData()
        
        if self.page == 'All':
            self.write_all(data)
              
        elif self.page == 'Individual':
            self.write_individual(data)
            
        elif self.page == 'Model':
            
            #my_slot1.column(width=100)
            # Appends an empty slot to the app. We'll use this later.
            #st.text("Features")
            self.method = st.radio(label='', options=["Relation", "Test"])
            
            if self.method == "Relation":
                self.write_relation(data)

            # columnList = self.dataset.columnList()
            # columnList.append('Exit_Marker')
            # self.fatured_data = self.dataset.getFatures(data, columnList)
            
            # feature_columns  = self.fatured_data.columns
            # summary, params = StatisticModel(self.fatured_data, feature_columns).glm()
            # st.write(summary)
            # st.write(params)
            # # table = hv.Table({'Features':np.array(features)}, ['Features'])
            # # table.opts(height=140)
            # # st.hvplot(table)
            # st.markdown("Features")
            # st.dataframe(np.array(feature_columns))
            #model = Model().m()
    
    def write_relation(self, data):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        columnList = self.dataset.columnList()

        ### GLM model for feature mapping
        columnList.append('Exit_Marker')
        fatured_data = self.dataset.getFatures(data, columnList, num=10000)

        feature_columns  = fatured_data.columns
        summary, params, tValues, trials, y, y_hat = \
            StatisticModel(fatured_data.head(1000), feature_columns).glm()
        

        #### Plotting and printing
        figure = make_subplots(
              rows=1, cols=3, 
              column_titles=(["<b>Chi-Square correlation</b> (chi-Value)", \
                  "<b>Chi-Square correlation</b> (p-Value)", "<b>GLM Preditions</b>"]), \
              #horizontal_spacing=0.1,\
              #vertical_spacing=0.07,\
              specs=np.array([{"type": "heatmap"}, {"type": "heatmap"}, {"type": "scatter"}]).\
                  reshape((1, 3)).tolist())
            
        chiList = []
        pValList = []
        for i, c in enumerate(columnList):
            contingancy_table = self.dataset.contigancyData(data, column=c)
            chi, critical, pValue, alpha = StatisticModel.correlation(contingancy_table)
            chiList.append(chi - critical)
            pValList.append(alpha - pValue)
            
        axis_template = dict(
             showgrid = True,
             linecolor = 'white', showticklabels = True)
        
        sorted_chi = sorted(zip(columnList, chiList), key=lambda x: x[1], reverse=True)
        sorted_column, chi =  zip(*sorted_chi)
        figure.add_trace(go.Heatmap(name='Chi-Value', x=sorted_column, y=['Exit_Marker'], z=np.array(chi).reshape((1, len(columnList))), xgap=2),\
                         row=1, col=1)
        
        sorted_pVal = sorted(zip(columnList, pValList), key=lambda x: x[1], reverse=True)
        sorted_column, pVal =  zip(*sorted_pVal)
        figure.add_trace(go.Heatmap(name = 'p-Value' ,x=sorted_column, y=['Exit_Marker'], z=np.array(pVal).reshape((1, len(columnList))), xgap=2 ),\
                         row=1, col=2)

        figure.add_trace(go.Scatter(name = 'GLM Model' ,x=y, y=y_hat, mode='markers'),\
                         row=1, col=3)
        
        figure.update_layout(height=500, width=1500, showlegend=False, xaxis = axis_template, yaxis = axis_template)
        st.write(figure)
        
        

        st.text('Total number of trials: {}'.format(trials))
        st.markdown("----------------------------------- tValues ----------------------------------")
        st.text(tValues)
        st.markdown("----------------------------------- params ----------------------------------")
        st.text(params)
        st.markdown("----------------------------------- Summary ----------------------------------")
        st.text(summary)
        
        
        
    def write_individual(self, data):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.user_id = st.selectbox('User ID', np.sort(data['user_id_uid'].unique()))
            
        st.markdown("Data from table...!!!")
        filtered_data = data[data['user_id_uid']==self.user_id]
        st.dataframe(data[data['user_id_uid']==self.user_id])
        st.markdown('Aggregated data...!!!')
        ### Agregation
        
        columnListGraph, columnListText = self.dataset.columnListInd()
        text = 'User {} details : \n\n '.format(self.user_id)
        for i, c in enumerate(columnListText):
           v = filtered_data[c].unique()
           text = text + '\t{} : {}\n'.format(c, max(v) if isinstance(v[0],np.int64) else v)
         
        st.text(text)
        
        figure = make_subplots(
                rows=len(columnListGraph), cols=2,
                column_titles=("<b>Active</b>", "<b>Inactive</b>"),  row_titles=columnListGraph,\
                #horizontal_spacing=0.1,\
                #vertical_spacing=0.04,\
                specs=np.array([{"type": "pie"}, {"type": "bar"}] * len(columnListGraph)).reshape((len(columnListGraph), 2)).tolist())
        
        
        initColumns = ['user_id_uid', 'start_', 'end_', 'paymentCycle_Marker']
        
        for i, c in enumerate(columnListGraph):
            columns = initColumns.copy()
            columns.append(c)
            agg_data, _ = self.dataset.loadAggData(filtered_data, columns=columns)
            
            ### Monthly data
            self.pie(fig=figure, labels = np.asarray(agg_data.index.get_level_values(c)),\
                      values=agg_data['date'], row=i+1, col=1, legendgroup=1)
            self.barPlot(figure, agg_data, indexX = 'paymentCycle_Marker', indexY=c, row=i+1, col=2)
        
        figure.update_layout(height=1400, width=1400, showlegend=False)
        st.write(figure)
        
        
        
    def write_all(self, data):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        columnList = self.dataset.columnList()
            
        data_active = data[data.Exit_Marker == 0]
        data_churned= data[data.Exit_Marker == 1]

        figure = make_subplots(
                rows=len(columnList), cols=4,   
                column_titles=("<b>Active</b>", "<b>Inactive</b>", "<b>All</b> (Count)", "<b>Churn rate</b> (Mean)"),  row_titles=columnList,\
                horizontal_spacing=0.1,\
                vertical_spacing=0.06,\
                specs=np.array([{"type": "pie"}, {"type": "pie"}, {"type": "bar"}, {"type": "bar"}] * len(columnList)).\
                    reshape((len(columnList), 4)).tolist())
        
        for i, c in enumerate(columnList):
            active, _ = self.dataset.loadAggData(data_active, columns=[c])
            active = active[active['date'] > 100]
            inactive, _ = self.dataset.loadAggData(data_churned, columns=[c])
            inactive = inactive[inactive['date'] > 100]
            
            self.pie(fig=figure, labels = np.array(active.index),values=active['date'], row=i+1, col=1)
            self.pie(fig=figure, labels = np.array(inactive.index),values=inactive['date'], row=i+1, col=2)
            
            
            self.bar(fig=figure, labels = np.array(active.index),values=active['date'], row=i+1, col=3, legendgroup=i+1, name='Active', mc='green')
            self.bar(fig=figure, labels = np.array(inactive.index),values=inactive['date'], row=i+1, col=3, legendgroup=i+1, name='Inactive', mc='red')
            
            _, churnAggData = self.dataset.loadAggData(data, columns=[c])
            self.bar(fig=figure, labels = churnAggData[c] ,values=churnAggData['Exit_Marker'], row=i+1, col=4)
            
            #self.pie(fig=piefig, values=count_active[c], row=i, col=1, legendgroup=1)
        
        figure.update_layout(height=4400, width=1600)
        st.write(figure)        
        
    def pie(self, fig, labels, values, row=1, col=1, legendgroup=1):
         
        fig.add_trace(go.Pie(labels = labels, values=values, legendgroup=legendgroup,\
                             hoverinfo='label+percent', textinfo='value', textfont_size=20, showlegend=False), row=row, col=col)
        #fig.update_traces()
        
    def bar(self, fig, labels, values, name=None, row=1, col=1, legendgroup=1, mc=None):
         
        fig.add_trace(go.Bar(name=name, x=labels, y=values, legendgroup=legendgroup, hoverinfo='x+y', marker_color=mc,\
                             showlegend=False), row=row, col=col)
            
    def Scatter(self, fig, labels, values, row=1, col=1, legendgroup=1, marker=None):
         
        fig.add_trace(go.Scatter(x=labels, y=values, legendgroup=legendgroup, hoverinfo='x+y', mode='markers', \
                                 marker=None if marker is None else dict(size=marker*10,color=marker)), row=row, col=col)
        #fig.update_traces(textposition='auto' )
        
    def barPlot(self,fig, bar, indexX = None, indexY=None, row=1, col=1):
        if indexY is not None and indexX is not None:
            color = {}
            for j, y in enumerate(bar.index):
                y = np.array(y)
                
                if y[4] not in color.keys():
                    color[y[4]] = 'rgb({}, {}, {})'.\
                        format(np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255))
                
                fig.add_trace(go.Bar(name=y[4], x=[y[3]], y=[bar['date'][j]], marker_color= color[y[4]], text=y[4],\
                                   textposition='auto', hoverinfo= 'name+y', legendgroup=y[4]), row=row, col=col)
            
        
