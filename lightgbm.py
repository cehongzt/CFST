# @Time : 2022/5/19 11:07
# @Author : hongzt
# @File : lgbmcfst
# @Time : 2022/5/11 19:23
# @Author : hongzt
# @File : xgboost_s
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

trains = pd.read_excel('F:\\database\\cfst\\SCFST.xlsx')
trainm = pd.read_excel('F:\\database\\cfst\\MCFST.xlsx')
rmse = []
rmse1 = []
rmse2 = []

trainl = pd.read_excel('F:\\database\\cfst\\LCFST.xlsx')
dems = trains[["D (mm)", "t (mm)", "Le (mm)", "fy (MPa)", "fc (MPa)"]].values
objects = trains[["N Test (kN)"]].values
demm = trainm[["D (mm)", "t (mm)", "Le (mm)", "fy (MPa)", "fc (MPa)"]].values
objectm = trainm[["N Test (kN)"]].values
deml = trainl[["D (mm)", "t (mm)", "Le (mm)", "fy (MPa)", "fc (MPa)"]].values
objectl = trainl[["N Test (kN)"]].values

from sklearn.model_selection import train_test_split

x_trains, x_tests, y_trains, y_tests = train_test_split(dems, objects, test_size=0.35, shuffle=True)
x_trainm, x_testm, y_trainm, y_testm = train_test_split(demm, objectm, test_size=0.35, shuffle=True)
x_trainl, x_testl, y_trainl, y_testl = train_test_split(deml, objectl, test_size=0.35, shuffle=True)

import joblib

from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
for train_indices, test_indices in kf.split(dems):
    X_train, X_test = dems[train_indices], dems[test_indices]
    Y_train, Y_test = objects[train_indices], objects[test_indices]
    lgbmRLRs = lgbm.LGBMRegressor(colsample_bytree=0.6, subsample=0.5, max_depth=3, n_estimators=100, max_delta_step=0,
                                 learning_rate=0.32, reg_alpha=0.5, min_child_weight=1.99, reg_lambda=0.229)

    lgbmRLRs.fit(X_train, Y_train)

    y_pred = lgbmRLRs.predict(X_test)
    RSM = np.sqrt(mean_squared_error(Y_test, (y_pred)))
    rmse.append(RSM)

    if RSM <= min(rmse) and len(rmse) != 0:
        joblib.dump(filename='lgbm_s.model1', value=lgbmRLRs)

    rmse_1 = np.sqrt(mean_squared_error(y_trains, lgbmRLRs.predict(x_trains)))

    rmse1.append(rmse_1)
print(rmse)
print(np.mean(rmse), np.mean(rmse1))
'''
for train_indices, test_indices in kf.split(dems):
    X_train, X_test = dems[train_indices], dems[test_indices]
    Y_train, Y_test = objects[train_indices], objects[test_indices]
    xgbtRLRs = xgbt.XGBRegressor(colsample_bytree=0.6, subsample=0.5, max_depth=3, n_estimators=100, max_delta_step=0,
                                 learning_rate=0.32, reg_alpha=0.5, min_child_weight=1.99, reg_lambda=0.229)
    xgbtRLRs.fit(X_train, Y_train)

    y_pred = xgbtRLRs.predict(X_test)
    RSM = np.sqrt(mean_squared_error(Y_test, (y_pred)))
    rmse.append(RSM)

    if RSM <= min(rmse) and len(rmse) != 0:
        joblib.dump(filename='xgboost_s.model1', value=xgbtRLRs)

    rmse_1 = np.sqrt(mean_squared_error(y_trains, xgbtRLRs.predict(x_trains)))

    rmse1.append(rmse_1)
'''
RMSE2 = []
print(np.mean(rmse), np.mean(rmse1))
for train_indicem, test_indicem in kf.split(demm):
    X_trainM, X_testM = demm[train_indicem], demm[test_indicem]
    Y_trainM, Y_testM = objectm[train_indicem], objectm[test_indicem]
    lgbmRLRm = lgbm.LGBMRegressor(colsample_bytree=0.6, subsample=0.5, max_depth=3, n_estimators=100, max_delta_step=0,
                                 learning_rate=0.32, reg_alpha=0.5, min_child_weight=1.99, reg_lambda=0.229)
    lgbmRLRm.fit(X_trainM, Y_trainM)

    y_predM = lgbmRLRm.predict(X_testM)
    RSMm = np.sqrt(mean_squared_error(Y_testM, (y_predM)))

    RMSE2.append(RSMm)

    if RSMm <= min(RMSE2) and len(RMSE2) != 0:
        joblib.dump(filename='lgbm_m.model1', value=lgbmRLRm)
print(np.mean(RMSE2))
RMSE3 = []

# print(np.mean(rmse), np.mean(rmse1))
for train_indicel, test_indicel in kf.split(deml):
    X_trainl, X_testl = deml[train_indicel], deml[test_indicel]
    Y_trainl, Y_testl = objectl[train_indicel], objectl[test_indicel]
    lgbmRLRl = lgbm.LGBMRegressor(colsample_bytree=0.6, subsample=0.5, max_depth=3, n_estimators=100, max_delta_step=0,
                                 learning_rate=0.32, reg_alpha=0.5, min_child_weight=1.99, reg_lambda=0.229)
    lgbmRLRl.fit(X_trainl, Y_trainl)

    y_predl = lgbmRLRl.predict(X_testl)
    RSMl = np.sqrt(mean_squared_error(Y_testl, (y_predl)))

    RMSE3.append(RSMl)

    if RSMl <= min(RMSE3) and len(RMSE3) != 0:
        joblib.dump(filename='lgbm_l.model1', value=lgbmRLRl)
print(np.mean(RMSE3))
'''
rmse_2=np.sqrt(mean_squared_error(objects, xgbtRLR.predict(dems)))
for train_indicem,test_indicem in kf.split(demm):
    X_train,X_test=demm[train_indicem],demm[test_indicem]
    Y_train, Y_test = objectm[train_indicem], objects[test_indicem]
    xgbtRLR=xgbt.XGBRegressor(colsample_bytree=0.6,subsample=0.5,max_depth=3,n_estimators=100,max_delta_step=0,learning_rate=0.32,reg_alpha=0.5,min_child_weight= 1.99, reg_lambda=0.229)
    xgbtRLR.fit(X_train,Y_train)

    y_pred = xgbtRLR.predict(X_test)
    RSM=np.sqrt(mean_squared_error(Y_test,(y_pred)))
    rmse_1=np.sqrt(mean_squared_error(y_trains, xgbtRLR.predict(x_trains)))

    rmse.append(RSM)
    rmse1.append(rmse_1)
rmse_2=np.sqrt(mean_squared_error(objects, xgbtRLR.predict(dems)))

print(rmse)
print(rmse1)
print(f'average rmse:{np.mean(rmse)}')
print(f'average rmse:{np.mean(rmse1)}')
print(rmse_2)
#xgbtRLR = xgbt.XGBRegressor(colsample_bytree=0.6, subsample=0.5, max_depth=3, n_estimators=100, max_delta_step=0,
                    #        learning_rate=0.32, reg_alpha=0.5, min_child_weight=1.99, reg_lambda=0.229) r,v,s328 326 325
'''
