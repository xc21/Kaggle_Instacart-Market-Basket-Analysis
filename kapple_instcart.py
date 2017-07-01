# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 07:41:10 2017

@author: Xun Cao
"""

import pandas as pd # dataframes
import numpy as np # algebra & calculus
import nltk # text preprocessing & manipulation
# from textblob import TextBlob
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting

from functools import partial # to reduce df memory consumption by applying to_numeric

color = sns.color_palette() # adjusting plotting style
import warnings
warnings.filterwarnings('ignore') # silence annoying warnings

#%matplotlib inline # insert graph into the notebook



#data loading数据导入
# products


#load training data set
op_train = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/order_products__train.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})

#load testing data set
test = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/sample_submission.csv', engine='c')

#prior dataset
op_prior = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/order_products__prior.csv', engine='c', 
                       dtype={'order_id': np.int32, 
                              'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 
                              'reordered': np.int8})
# orders
orders = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/orders.csv', engine='c', dtype={'order_id': np.int32, 
                                                           'user_id': np.int32, 
                                                           'order_number': np.int32, 
                                                           'order_dow': np.int8, 
                                                           'order_hour_of_day': np.int8, 
                                                           'days_since_prior_order': np.float16})
products = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/products.csv', engine='c')
departments = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/departments.csv', engine='c')
aisles = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/departments.csv', engine='c')


# combine aisles, departments and products (left joined to products)
goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')
# to retain '-' and make product names more "standard"
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 
print(goods.info())
goods.head()

# merge train and prior together iteratively, to fit into 8GB kernel RAM
# split df indexes into parts
indexes = np.linspace(0, len(op_prior), num=10, dtype=np.int32) #把prior orders分成十块

# initialize it with train dataset
order_details = pd.merge(
                left=op_train,
                 right=orders, 
                 how='left', 
                 on='order_id'
        ).apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))

# add order hierarchy
order_details = pd.merge(
                left=order_details,
                right=goods[['product_id', 
                             'aisle_id', 
                             'department_id']].apply(partial(pd.to_numeric, 
                                                             errors='ignore', 
                                                             downcast='integer')),
                how='left',
                on='product_id'
)

print(order_details.shape, op_train.shape)

# delete (redundant now) dataframes
del op_train


# update by small portions
for i in range(len(indexes)-1):
    order_details = pd.concat(
        [   
            order_details,
            pd.merge(left=pd.merge(
                            left=op_prior.loc[indexes[i]:indexes[i+1], :],
                            right=goods[['product_id', 
                                         'aisle_id', 
                                         'department_id' ]].apply(partial(pd.to_numeric, 
                                                                          errors='ignore', 
                                                                          downcast='integer')),
                            how='left',
                            on='product_id'
                            ),
                     right=orders, 
                     how='left', 
                     on='order_id'
                ) #.apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))
        ]
    )
        
print('Datafame length: {}'.format(order_details.shape[0]))
print('Memory consumption: {:.2f} Mb'.format(sum(order_details.memory_usage(index=True, 
                                                                         deep=True) / 2**20)))
# check dtypes to see if we use memory effectively
print(order_details.dtypes)

# make sure we didn't forget to retain test dataset :D
test_orders = orders[orders.eval_set == 'test']

# delete (redundant now) dataframes
del op_prior, orders

test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]
last_orders = test_history.groupby('user_id')['order_number'].max()

def get_last_orders_reordered():
    t = pd.merge(
            left=pd.merge(
                    left=last_orders.reset_index(),
                    right=test_history[test_history.reordered == 1],
                    how='left',
                    on=['user_id', 'order_number']
                )[['user_id', 'product_id']],
            right=test_orders[['user_id', 'order_id']],
            how='left',
            on='user_id'
        ).fillna(-1).groupby('order_id')['product_id'].apply(lambda x: ' '.join([str(int(e)) for e in set(x)]) 
                                                  ).reset_index().replace(to_replace='-1', 
                                                                          value='None')
    t.columns = ['order_id', 'products']
    return t

# save submission
get_last_orders_reordered().to_csv('less_dumb_subm_last_order_reordered_only.csv', 
                         encoding='utf-8', 
                         index=False)
