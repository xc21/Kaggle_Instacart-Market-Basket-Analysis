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

%matplotlib inline #insert graph into the notebook



#explorative analysis
# products
products = pd.read_csv('C:/Users/Xun Cao/Desktop/kaggle/input/products.csv', engine='python')
print('Total products: {}'.format(products.shape[0]))
products.head(5)

#same for other csv 
#try orders
orders = pd.read_csv('C:/Users/Xun Cao/Desktop/kaggle/input/orders.csv', engine='c')
print('Total orders: {}'.format(orders.shape[0]))
orders.head(5)