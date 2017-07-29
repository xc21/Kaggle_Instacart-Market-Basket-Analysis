# This is my first try to visualize Instacart orders

#First, we will reload our data to the enviroment:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
#turn off the warnings
pd.options.mode.chained_assignment = None  # default='warn'

order_products_train_df = pd.read_csv("C:/Users/caoxun/Box Sync/kaggle/input/order_products__train.csv")
order_products_prior_df = pd.read_csv("C:/Users/caoxun/Box Sync/kaggle/input/order_products__prior.csv")
orders_df = pd.read_csv("C:/Users/caoxun/Box Sync/kaggle/input/orders.csv")
products_df = pd.read_csv("C:/Users/caoxun/Box Sync/kaggle/input/products.csv")
aisles_df = pd.read_csv("C:/Users/caoxun/Box Sync/kaggle/input/aisles.csv")
departments_df = pd.read_csv("C:/Users/caoxun/Box Sync/kaggle/input/departments.csv")

#To see the header of each table:
orders_df.head()
order_products_prior_df.head()
order_products_train_df.head()
#There is one column in the orders table named eval_set indicating the final 
#destination of this entry: prior, test or training. 
#To visualize such data division percentage, we draw the histogram as below:

    
cnt_srs = orders_df.eval_set.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Eval set type', fontsize=12)
plt.title('Count of rows in each dataset', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

#In number:
def get_unique_count(x):
    return len(np.unique(x))

cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)
cnt_srs
#eval_set
#prior    206209
#test      75000


#Among the 206209 customers, 
#we want to see the distribution of their reorder behaviors: 
cnt_srs = orders_df.groupby("user_id")["order_number"].aggregate(np.max).reset_index()
cnt_srs = cnt_srs.order_number.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])
plt.ylabel('Customer Reorder Behavior', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Maximum order number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()    
    