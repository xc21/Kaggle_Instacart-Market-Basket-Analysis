# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 16:46:49 2017

@author: caoxun
"""

import pandas as pd # dataframes
import numpy as np # algebra & calculus
import seaborn as sns # plotting
from functools import partial # to reduce df memory consumption by applying to_numeric
color = sns.color_palette() # adjusting plotting style
import warnings
warnings.filterwarnings('ignore') # silence annoying warnings


#ETL

#读取数据,分为两个方向处理：1）商品(Products)side 2).顾客(Orders)side


#商品(Product)side:
products = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/products.csv', engine='c')
departments = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/departments.csv', engine='c')
aisles = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/departments.csv', engine='c')
                     
#合并prduct, aisles, department, all left join to products, 保证products中的全部
#合并后的变量成为goods
goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')
# 在产品名称中间加下划线，使其更加standard
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 
print(goods.info())
goods.head()
#good中有的变量： product_id, product_name（具体内容）, aisle_id, department_id, department(具体内容)





#顾客(Order) Side
orders = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/orders.csv', engine='c')
#orders中有的变量： orderID, userID, eval_set(prior/train), order_number, orderdow(星期几)， orderHour, days_since_prior_order(和前一单之间隔了多久)
op_prior = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/order_products__prior.csv', engine='c')
#OP_prior/train中有的变量： orderID, productID, addToCart, reorderd(1/0,这一单中的物品是否是之前（任何一单)曾经买过的）
op_train = pd.read_csv('C:/Users/caoxun/Box Sync/kaggle/input/order_products__train.csv', engine='c')
#OP_train 中的变量维度同上


#将顾客(Order) Side & 商品（Product)部分合并 >>> training set
# initialize it with train dataset，初始化，将op_train信息和order信息合并，以orderTrain为准，成为order_details
order_details = pd.merge(
                left=op_train,
                right=orders, 
                 how='left', 
                 on='order_id'
        ).apply(partial(pd.to_numeric, errors='ignore', downcast='integer')) #partial的function是让对象变为numeric,
                                                                             #减少运行时的内存消耗，具体原因不太懂 @-@

# add order hierarchy, 将order_detail(前一步已合并的)与good(商品部分)合并，合并顾客与商品, two sides combined
# 以上 以下两个合并部分，均只保留顾客/产品的id信息而不加入id对应的内容
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

# delete (redundant now) dataframes，删除op_train, 因为信息已经合并到order_details
del op_train

# update by small portions, 不太懂这一块，question asked
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


#Repeat Last Order (Reordered Products Only)
def get_last_orders_reordered():
    t = pd.merge(
            left=pd.merge(
                    left=last_orders.reset_index(),
                    right=test_history[test_history.reordered == 1],
                    how='left',
                    on=['user_id', 'order_number']
                )[['user_id', 'product_id']],    #left table是test里曾今reorder过的，取user id, product id两个变量
            right=test_orders[['user_id', 'order_id']],
            how='left',
            on='user_id'
        )
    t = t.fillna(-1).groupby('order_id')['product_id'].apply(lambda x: ' '.join([str(int(e)) for e in set(x)]) 
                                                  ).reset_index().replace(to_replace='-1', 
                                                                          value='None')
    t.columns = ['order_id', 'products']
    return t

# save submission
get_last_orders_reordered().to_csv('less_dumb_subm_last_order_reordered_only.csv', 
                         encoding='utf-8', 
                         index=False)
