# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:39:51 2017

@author: caoxun
"""

import numpy as np 
import pandas as pd 

orders = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/orders.csv', engine='c')
prior = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/order_products__prior.csv', engine='c')
train = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/order_products__train.csv', engine='c')
order_prior = pd.merge(prior,orders,on=['order_id','order_id'])
order_prior = order_prior.sort_values(by=['user_id','order_id'])
order_prior.head()
products = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/products.csv', engine='c')
aisles = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/aisles.csv', engine='c')
_mt = pd.merge(prior,products, on = ['product_id','product_id'])
_mt = pd.merge(_mt,orders,on=['order_id','order_id'])
mt = pd.merge(_mt,aisles,on=['aisle_id','aisle_id'])
mt.head(10)

#cout the top 10 popular products
mt['product_name'].value_counts()[0:10]

#number of unique products
len(mt['product_name'].unique())