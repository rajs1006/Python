#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:43:13 2020

@author: sraj
"""
from Visualizer import Visualizer


def main():
    
    vis = Visualizer(dataPath = 'churn-rate_v1.csv')
    vis.draw()
    
main()