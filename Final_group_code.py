# -*- coding: utf-8 -*-
"""

@author: Hasan
"""

import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

#find outliers in the dataset for numerical and catagorical data
    
numeric_col =['Year of Record','Size of City','Work Experience in Current Job [years]',
              'Age','Yearly Income in addition to Salary (e.g. Rental Income)'] 


category_col =['Housing Situation','Satisfation with employer'
,'Gender'
,'Country'
,'Profession'
,'University Degree'] 



#for c in category_col:
#    print (c)
#    print (dataset[c].value_counts())
    
#for c in numeric_col:
#    print (c)
#    print (dataset[c].value_counts())



training = pd.read_csv("tcd-ml-1920-group-income-train.csv")
test= pd.read_csv("tcd-ml-1920-group-income-test.csv")
instance_col=test['Instance']

#drop empty prediction col
test = test.iloc[:,:-1]



#dropping DUPLICATES
#training =training.drop_duplicates()   

#Remove less significant Column 
training = training.drop("Instance",1)
test = test.drop("Instance",1)
training = training.drop("Wears Glasses",1)
test = test.drop("Wears Glasses",1)
training = training.drop("Hair Color",1)
test = test.drop("Hair Color",1)
training = training.drop("Crime Level in the City of Employement",1)
test = test.drop("Crime Level in the City of Employement",1)
training = training.drop("Body Height [cm]",1)
test = test.drop("Body Height [cm]",1)



training['Yearly Income in addition to Salary (e.g. Rental Income)'] = training['Yearly Income in addition to Salary (e.g. Rental Income)'].str.split(" ", n = 1, expand = True)[0]

test['Yearly Income in addition to Salary (e.g. Rental Income)'] = test['Yearly Income in addition to Salary (e.g. Rental Income)'].str.split(" ", n = 1, expand = True)[0]



training['Yearly Income in addition to Salary (e.g. Rental Income)'] = training['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)
test['Yearly Income in addition to Salary (e.g. Rental Income)'] = test['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)

#turning bad strings to np.nan

training[:] = training[:].replace('#NUM!',np.nan)
training[:] = training[:].replace('nA',np.nan)
training[:] = training[:].replace('unknown',np.nan)
training[:] = training[:].replace('#NA',np.nan)

test[:] = test[:].replace('#NUM!',np.nan)
test[:] = test[:].replace('nA',np.nan)
test[:] = test[:].replace('unknown',np.nan)
test[:] = test[:].replace('#NA',np.nan)

#Handle Year of Record
#Looking at the data the missing are after 2019 so asssume its 2019 


training["Year of Record"] = training.replace( np.nan ,2019)
test["Year of Record"] = test.replace( np.nan ,2019)

#training['Year of Record'] = training['Year of Record']**(1/2)
#test['Year of Record'] = test['Year of Record']**(1/2)

# Age
#training['Age'] = training['Age']**(1/2)
#test['Age'] = test['Age']**(1/2)



from sklearn.impute import SimpleImputer
simpleimputer=SimpleImputer(strategy='mean')

for x in numeric_col:
   training[x]=simpleimputer.fit_transform(training[x].values.reshape(-1,1))
   test[x]=simpleimputer.fit_transform(test[x].values.reshape(-1,1))

    
for x in category_col:
    training[x] = training[x].fillna('Nan_data')
    test[x] = test[x].fillna('Nan_data')

    ########################## CATAGORICAL DATA ###################################

#Testing varaiations of cleaning data


# Housing Situation

#training['Housing Situation'] = training['Housing Situation'].replace( '0' ,'0')
#training['Housing Situation'] = training['Housing Situation'].replace( 'nA' ,'0')
#training['Housing Situation'] = training['Housing Situation'].replace( 'Large House' ,'1')
#training['Housing Situation'] = training['Housing Situation'].replace( 'Medium House' ,'1')
#training['Housing Situation'] = training['Housing Situation'].replace( 'Small House' ,'1')
#training['Housing Situation'] = training['Housing Situation'].replace( 'Large Apartment' ,'2')
#training['Housing Situation'] = training['Housing Situation'].replace( 'Medium Apartment' ,'2')
#training['Housing Situation'] = training['Housing Situation'].replace( 'Small Apartment' ,'2')
#training['Housing Situation'] = training['Housing Situation'].replace( 'Castle' ,'3')

#test['Housing Situation'] = test['Housing Situation'].replace( '0' ,'0')
#test['Housing Situation'] = test['Housing Situation'].replace( 'nA' ,'0')
#test['Housing Situation'] = test['Housing Situation'].replace( 'Large House' ,'1')
#test['Housing Situation'] = test['Housing Situation'].replace( 'Medium House' ,'1')
#test['Housing Situation'] = test['Housing Situation'].replace( 'Small House' ,'1')
#test['Housing Situation'] = test['Housing Situation'].replace( 'Large Apartment' ,'2')
#test['Housing Situation'] = test['Housing Situation'].replace( 'Medium Apartment' ,'2')
#test['Housing Situation'] = test['Housing Situation'].replace( 'Small Apartment' ,'2')
#test['Housing Situation'] = test['Housing Situation'].replace( 'Castle' ,'3')

#training['Housing Situation'] = training['Housing Situation'].astype('int64')
#test['Housing Situation'] = test['Housing Situation'].astype('int64')

#training['Housing Situation'] = training['Housing Situation'].replace( '0' ,'0')
#training['Housing Situation'] = training['Housing Situation'].replace( 'nA' ,'0')
#test['Housing Situation'] = test['Housing Situation'].replace( '0' ,'0')
#test['Housing Situation'] = test['Housing Situation'].replace( 'nA' ,'0')

#Satisfation with employer

#training['Satisfation with employer'] = training['Satisfation with employer'].replace( 'Somewhat Happy' ,'Happy')
#test['Satisfation with employer'] = test['Satisfation with employer'].replace( 'Somewhat Happy' ,'Happy')


# Gender


training['Gender'] = training['Gender'].replace( 'f' ,'female')
#training['Gender'] = training['Gender'].replace( np.nan ,'Nan_data')
training['Gender'] = training['Gender'].replace( '0' ,'Nan_data')
#training['Gender'] = training['Gender'].replace( 'unknown' ,'Nan_data')
training['Gender'] = training['Gender'].replace( 'other' ,'Nan_data')

test['Gender'] = test['Gender'].replace( 'f' ,'female')
#test['Gender'] = test['Gender'].replace( np.nan ,'Nan_data')
test['Gender'] = test['Gender'].replace( '0' ,'Nan_data')
#test['Gender'] = test['Gender'].replace( 'unknown' ,'Nan_data')
test['Gender'] = test['Gender'].replace( 'other' ,'Nan_data')



#Country 

training['Country'] = training['Country'].replace( '0' ,'Mexico')
test['Country'] = test['Country'].replace( '0' ,'Mexico')

#Profession 

#training['Profession'] = training['Profession'].replace( np.nan ,'Nan_data')

#test['Profession'] = test['Profession'].replace( np.nan ,'Nan_data')



#University Degree

#training['University Degree'] = training['University Degree'].replace( np.nan ,'Nan_data')
training['University Degree'] = training['University Degree'].replace( '0' ,'Nan_data')

#test['University Degree'] = test['University Degree'].replace( np.nan ,'Nan_data')
test['University Degree'] = test['University Degree'].replace( '0' ,'Nan_data')


#hair Color 

#training['Hair Color'] = training['Hair Color'].replace( np.nan ,'Nan_data')
#training['Hair Color'] = training['Hair Color'].replace( '0' ,'Nan_data')
#training['Hair Color'] = training['Hair Color'].replace( 'Unknown' ,'Nan_data')

#test['Hair Color'] = test['Hair Color'].replace( np.nan ,'Nan_data')
#test['Hair Color'] = test['Hair Color'].replace( '0' ,'Nan_data')
#test['Hair Color'] = test['Hair Color'].replace( 'Unknown' ,'Nan_data')

    
X = training.iloc[:,:-1]
y = training.iloc[:,-1]

#Target encoding for categorical features.

te = TargetEncoder()
te.fit(X,y)

X = te.transform(X)

predict_dataset = te.transform(test)


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(X, y, test_size=0.3, random_state=42)

#from catboost import CatBoostRegressor

# Using CatBoost
#cat_model3 = CatBoostRegressor(iterations=125000)
#cat_model3.fit(x_train, y_train)

#catboost_model = CatBoostRegressor(iterations=200, verbose=False)
#prediction = catboost_model.predict(predict_dataset)

#df = pd.DataFrame({'Instance':instance_col,
#                       'Total Yearly Income [EUR]':prediction})

#df.to_csv("catbooster_Final.csv",index=False)


#lgb best ensemble model for this dataset

import lightgbm as lgb
lgb_train = lgb.Dataset(x_train, label=y_train)
lgb_val = lgb.Dataset(x_val, label=y_val)
params = {}
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
     'num_leaves': 35,
    'learning_rate': 0.015,
    'verbose': 0,
    #'max_bin': 255,
    #'feature_fraction': 0.8,
   # 'bagging_fraction': 0.8,
   # 'bagging_freq': 5,
   # 'min_data_in_leaf': 50,
   # 'min_sum_hessian_in_leaf': 5
}


clf = lgb.train(params, lgb_train, 100000, valid_sets = [lgb_train, lgb_val], verbose_eval=1000, early_stopping_rounds=750)


prediction = clf.predict(predict_dataset)

#prediction = catboost_model.predict(predict_dataset)

df = pd.DataFrame({'Instance':instance_col,
                       'Total Yearly Income [EUR]':prediction})

df.to_csv("LGB_Final9.csv",index=False)







