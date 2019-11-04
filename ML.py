import pandas as pd
import numpy as np
import matplotlib as mpl
import math
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import  r2_score
import time


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def median_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

def mse_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.square((y_true - y_pred) / y_true*100))

np.set_printoptions(threshold= np.inf)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)


hw = pd.read_csv('Data Science ZExercise_TRAINING_CONFIDENTIAL1.csv')

Zlist = ['SF 5000',
'R6',
'R4',
'RA5',
'R5',
'SF 7200',
'R8',
'SR6',
'RS7200',
'R3.5',
'RS7.2',
'RSA 6',
'R1',
'RA2.5',
'MU',
'R6P',
'R7',
'RSX 7.2',
'RS9.6',
'URPSO',
'R9.6',
'RS 8.5',
'UL7200',
'LDR',
'SR4.5'
         ]

hw.loc[~hw.ZoneCodeCounty.isin(Zlist),'ZoneCodeCounty'] = 'other'




def bin_column(col, bins, labels, na_label='unknown'):
    """
    Takes in a column name, bin cut points and labels, replaces the original column with a
    binned version, and replaces nulls (with 'unknown' if unspecified).
    """
    hw[col] = pd.cut(hw[col], bins=bins, labels=labels, include_lowest=True)
    hw[col] = hw[col].astype('str')
    hw[col].fillna(na_label, inplace=True)


hw['BGM'] = hw.BGMedRent
hw['Gara'] = hw.GarageSquareFeet
hw['Gara'].fillna(0,inplace = True)


bin_column('BGM',
           bins=[0,1000,1250, 1600, max(hw.BGMedRent)],
           labels=['0-1',
                   '1-1.25',
                   '1.25-1.6',
                   '>'
                   ],
           na_label='unknown')




hw.TransDate = pd.to_datetime(hw.TransDate)
hw['days'] = (datetime(2019,10,27)-hw.TransDate).astype('timedelta64[D]')
hw['Long'] = abs(hw.Longitude)



temp = hw['BGMedYearBuilt'].median()

hw['BGMedYearBuilt'].fillna(temp,inplace = True)

temp = hw['BGMedHomeValue'].median()

hw['BGMedHomeValue'].fillna(temp,inplace = True)



bin_column('ViewType',
           bins=[0,78,79, 82,241,244,246,247],
           labels=['1',
                   '2',
                   '3',
                   '4',
                   '5',
                   '6',
                   '7'
                   ],
           na_label='unknown')



todrop = ['Usecode','Longitude','censusblockgroup','GarageSquareFeet','BGMedRent','PropertyID','TransDate']
df = hw.drop(todrop,axis=1)


transformed_df = pd.get_dummies(df)




numer = [
    'Gara',
'SaleDollarCnt',
'BedroomCnt',
'BathroomCnt',
'FinishedSquareFeet',
'LotSizeSquareFeet',
'StoryCnt',
'BuiltYear',
'Latitude',
'Long',
'BGMedHomeValue',
'BGMedYearBuilt',
'BGPctOwn',
'BGPctVacant',
'BGMedIncome',
'BGPctKids',
'BGMedAge',
'days']




for col in numer:
    transformed_df[col] = transformed_df[col].astype('float64').replace(0.0,0.01)
    transformed_df[col] = np.log(transformed_df[col])
    tdf[col] = tdf[col].astype('float64').replace(0.0, 0.01)
    tdf[col] = np.log(tdf[col])


x = transformed_df.drop('SaleDollarCnt',axis=1)
y = transformed_df.SaleDollarCnt


scaler = StandardScaler()
x= pd.DataFrame(scaler.fit_transform(x),columns = list(x.columns))



xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state = 2000)



# Fitting the model
xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(xtrain, ytrain)
training_preds_xgb_reg = xgb_reg.predict(xtrain)
val_preds_xgb_reg = xgb_reg.predict(xtest)

ytrain = np.exp(ytrain)
training_preds_xgb_reg = np.exp(training_preds_xgb_reg)
ytest = np.exp(ytest)
val_preds_xgb_reg = np.exp(val_preds_xgb_reg)



# Printing the results

print("\nTraining r2:", round(r2_score(ytrain, training_preds_xgb_reg),4))
print("Validation r2:", round(r2_score(ytest, val_preds_xgb_reg),4))
print("\nTraining AAPE:", round(mean_absolute_percentage_error(ytrain, training_preds_xgb_reg),4))
print("Validation AAPE:", round(mean_absolute_percentage_error(ytest, val_preds_xgb_reg),4))
print("\nTraining MAPE:", round(median_absolute_percentage_error(ytrain, training_preds_xgb_reg),4))
print("Validation MAPE:", round(median_absolute_percentage_error(ytest, val_preds_xgb_reg),4))
print("\nTraining MSE:", round(mse_percentage_error(ytrain, training_preds_xgb_reg),4))
print("Validation MSE:", round(mse_percentage_error(ytest, val_preds_xgb_reg),4))



ft_weights_xgb_reg = pd.DataFrame(xgb_reg.feature_importances_, columns=['weight'], index=xtrain.columns)
ft_weights_xgb_reg.sort_values('weight', inplace=True)



ax = ft_weights_xgb_reg.plot.bar(
                  y='weight',
                  align='center', figsize = (13,10),
                  alpha=0.5,fontsize= 7)



fig = ax.get_figure()

fig.subplots_adjust( bottom=0.3)

fig.savefig('fifi.png')
plt.show()



# Neural Network Deep Learning
from keras import models, layers, optimizers, regularizers


# Building the model
nn2 = models.Sequential()
nn2.add(layers.Dense(128, input_shape=(xtrain.shape[1],),kernel_regularizer= regularizers.l1(0.005), activation='relu'))
nn2.add(layers.Dense(256,kernel_regularizer= regularizers.l1(0.005), activation='relu'))
nn2.add(layers.Dense(256,kernel_regularizer= regularizers.l1(0.005), activation='relu'))
nn2.add(layers.Dense(512,kernel_regularizer= regularizers.l1(0.005), activation='relu'))
nn2.add(layers.Dense(1, activation='linear'))

# Compiling the model
nn2.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['mean_squared_error'])


# Training the model
nn2_history = nn2.fit(xtrain,
                  ytrain,
                  epochs=150,
                  batch_size=256,
                  validation_split = 0.1)

y_test_pred = nn2.predict(xtest)
y_train_pred = nn2.predict(xtrain)

ytrain = np.exp(ytrain)
y_test_pred = np.exp(y_test_pred)
ytest = np.exp(ytest)
y_train_pred= np.exp(y_train_pred)


res = nn2.predict(x)
res = np.exp(res)



print("\nTraining r2:", round(r2_score(ytrain, y_train_pred), 4))
print("Validation r2:", round(r2_score(ytest, y_test_pred), 4))

print("\nTraining AAPE:", round(mean_absolute_percentage_error(ytrain, y_train_pred),4))
print("Validation AAPE:", round(mean_absolute_percentage_error(ytest,y_test_pred),4))
print("\nTraining MAPE:", round(median_absolute_percentage_error(ytrain, y_train_pred),4))
print("Validation MAPE:", round(median_absolute_percentage_error(ytest, y_test_pred),4))

def nn_model_evaluation(model, skip_epochs=0):
    model_results = model.history.history
    plt.plot(list(range((skip_epochs+1),len(model_results['loss'])+1)), model_results['loss'][skip_epochs:], label='Train')
    plt.plot(list(range((skip_epochs+1),len(model_results['val_loss'])+1)), model_results['val_loss'][skip_epochs:], label='Test', color='green')
    plt.legend()
    plt.title('Training and test loss at each epoch', fontsize=14)
    plt.show()

nn_model_evaluation(nn2)
