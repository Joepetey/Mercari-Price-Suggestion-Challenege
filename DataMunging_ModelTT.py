# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 17:47:26 2018

@author: goldw
"""
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import xgboost as xgb

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

trainset_data = pd.read_csv("C:\\Users\\goldw\\Documents\\train.tsv", sep='\t')
testset_data = pd.read_csv("C:\\Users\\goldw\\Documents\\test.tsv", sep='\t')

description = trainset_data.ix[:,7] 
price = trainset_data.ix[:,5]
brand = trainset_data.ix[:,4]
name = trainset_data.ix[:,1]
category = trainset_data.ix[:,3]
train_id = trainset_data.ix[:,0]
item_condition = trainset_data.ix[:,2]
shipping = trainset_data.ix[:,6]

description_test = testset_data.ix[:,6] 
brand_test = testset_data.ix[:,4]
name_test = testset_data.ix[:,1]
category_test = testset_data.ix[:,3]
train_id_test = testset_data.ix[:,0]
item_condition_test = testset_data.ix[:,2]
shipping_test = testset_data.ix[:,5]

#brand iteration
brand = brand.fillna(-1) 
brand = brand.astype('str')
brand = brand.as_matrix()
brand_test = brand_test.fillna(-1) 
brand_test = brand_test.astype('str')
brand_test = brand_test.as_matrix()

#category cleaning
category = category.astype('str')
category_array = category.as_matrix()
cat1 = [0] * 1482535
cat2 = [0] * 1482535
cat3 = [0] * 1482535
category_iter = category.iteritems()
z = 0
for i,j in category_iter:
    if j == "nan":
        cat1[z] = -1
        cat2[z] = -1
        cat3[z] = -1 
    else:
        temp_string = j
        cat1[z] = temp_string.split('/')[0]
        cat2[z] = temp_string.split('/')[1]
        cat3[z] = temp_string.split('/')[2]
    z = z+1
    
cat_array1 = np.array(cat1, dtype = "str")
cat_array2 = np.array(cat2, dtype = "str")
cat_array3 = np.array(cat3, dtype = "str")

#category test cleaning
category_test = category_test.astype('str')
category_array = category_test.as_matrix()
cat1 = [0] * 693359
cat2 = [0] * 693359
cat3 = [0] * 693359
category_iter = category_test.iteritems()
z = 0
for i,j in category_iter:
    if j == "nan":
        cat1[z] = -1
        cat2[z] = -1
        cat3[z] = -1 
    else:
        temp_string = j
        cat1[z] = temp_string.split('/')[0]
        cat2[z] = temp_string.split('/')[1]
        cat3[z] = temp_string.split('/')[2]
    z = z+1
    
cat_testarray1 = np.array(cat1, dtype = "str")
cat_testarray2 = np.array(cat2, dtype = "str")
cat_testarray3 = np.array(cat3, dtype = "str")
        
description = description.str.replace("No description yet", "nan")
description_test = description_test.str.replace("No description yet", "nan")


#brand binary encoding
encoder = preprocessing.LabelEncoder()
encoder.fit(brand)
brand_features = encoder.transform(brand)
brand_features = pd.Series(data = brand_features)
encoder.fit(brand_test)
brand_features_test = encoder.transform(brand_test)
brand_features_test = pd.Series(data = brand_features)

#category binary encoding
cat_encoder = preprocessing.LabelEncoder()
cat_encoder.fit(cat_array1)
cat_features1 = cat_encoder.transform(cat_array1)
cat_features1 = pd.Series(data = cat_features1)
cat_encoder.fit(cat_array2)
cat_features2 = cat_encoder.transform(cat_array2)
cat_features2 = pd.Series(data = cat_features2)
cat_encoder.fit(cat_array3)
cat_features3 = cat_encoder.transform(cat_array3)
cat_features3 = pd.Series(data = cat_features3)
cat_encoder.fit(cat_testarray1)
cat_testfeatures1 = cat_encoder.transform(cat_testarray1)
cat_testfeatures1 = pd.Series(data = cat_testfeatures1)
cat_encoder.fit(cat_testarray2)
cat_testfeatures2 = cat_encoder.transform(cat_testarray2)
cat_testfeatures2 = pd.Series(data = cat_testfeatures2)
cat_encoder.fit(cat_testarray3)
cat_testfeatures3 = cat_encoder.transform(cat_testarray3)
cat_testfeatures3 = pd.Series(data = cat_testfeatures3)

#startingtfid
vectorizer = TfidfVectorizer(ngram_range = (2,3), lowercase=True, use_idf=True, min_df = 0.01, max_df =.9, stop_words = "english")
tfid_full = vectorizer.fit_transform(description.values.astype(str))
bag_word =  pd.DataFrame(tfid_full.toarray(), columns = vectorizer.get_feature_names())
tfid_full = vectorizer.fit_transform(description_test.values.astype(str))
bag_word_test =  pd.DataFrame(tfid_full.toarray(), columns = vectorizer.get_feature_names())
bag_word_test = bag_word_test.drop("new tag",axis = 1)

#Transformed Data Sets
traindata_df = pd.concat([item_condition, cat_features1, cat_features2, cat_features3, brand_features, shipping, bag_word], axis = 1)
testdata_df = pd.concat([item_condition_test, cat_testfeatures1, cat_testfeatures2, cat_testfeatures3, brand_features_test, shipping_test, bag_word_test], axis = 1)

#XGBoost Training and Testing
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

xgb_model = xgb.XGBClassifier(parameters)
xgb_model.fit(traindata_df,price)
predictions = xgb_model.predict(testdata_df)




