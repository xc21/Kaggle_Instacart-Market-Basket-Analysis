# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 09:58:19 2017

@author: caoxun
"""

import numpy as np 
import pandas as pd 

nlargest = 3. #Just use the latest 3 orders per user

def GrabTestData():
    orders =  pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/orders.csv')
    
    
    #paul out the testing set from the total order data set
    orderstest = orders[orders.eval_set=='test']
    #find out users who are in the test data set and their user id
    testusers = orderstest.user_id.values
    
    
    #paul out the prior set from the total order data set
    ordersprior = orders[orders.eval_set=='prior']
    #using the user id as the connection key to find out test users' prior orders
    orderstestprior = ordersprior[ordersprior.user_id.isin(testusers)]
    
    #grpids as a index for test set's prior orders
    orderstestprior['grpids'] = range(orderstestprior.shape[0])
    grporderstestprior = orderstestprior.groupby(['user_id'])['grpids'].nlargest(int(nlargest)).reset_index()
    orderstestprior = orderstestprior[orderstestprior.grpids.isin(grporderstestprior.grpids)]
    prior = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/order_products__prior.csv')
    orderstestprior.drop(['eval_set','grpids'],inplace=True,axis=1)
    orderstestprior = orderstestprior.merge(prior,on='order_id')
    x = orderstestprior.groupby(['user_id','product_id'])['reordered'].mean().reset_index()
    x.columns = ['user_id','product_id','romean']
    x = x[x.romean>=.5]  
    suborderstest = orders[orders.eval_set=='test']
    suborderstest.drop(['eval_set'],inplace=True,axis=1)
    suborderstest = suborderstest.merge(x,on=['user_id'])
    return suborderstest[['order_id','product_id']]

test = GrabTestData()
sub = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/sample_submission.csv')

d2 = dict()
for row in test.itertuples():
    try:
        d2[row.order_id] += ' ' + str(row.product_id)
    except:
        d2[row.order_id] = str(row.product_id)

for order in sub.order_id:
    if order not in d2:
        d2[order] = 'None'
sub = pd.DataFrame.from_dict(d2, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub = sub.sort_values(by='order_id')
sub = sub.reset_index(drop=True)
sub.products = sub.products.astype(str)

sub.to_csv('simples.csv',index=False)