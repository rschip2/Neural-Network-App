import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

import pandas as pd
import numpy as np
import sys
import warnings
import random
import gc
import pAUCc as PAUCC

import keras
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras import backend as K

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
#from sklearn import metrics
#from sklearn.model_selection import StratifiedKFold #, RepeatedKFold, cross_val_score

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


#########################################################################################
#################### DEFINE FXNS ########################################################
#########################################################################################


def get_partial_auc(x, y):

    x_part = []
    y_part = []
    for i, xi in enumerate(x):
        if xi < 0.25:
            x_part.append(xi)
            y_part.append(y[i])
        
    AUC = PAUCC.concordant_partial_AUC(x_part, y_part)
    return AUC
    



def get_results(df, predictors, outcome, hidden_layers, inodes, nodes, folds,
    epochs, patience, batch_size, learning_rate):

    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    df = pd.read_json(df)
    df.dropna(how='any', axis=0, inplace=True)
    
    items = predictors+outcome
    items = list(set(items))
    df = df.filter(items=items, axis=1)
    
    y = df.filter(items=outcome, axis=1)
    #le = preprocessing.LabelEncoder()
    #y = y.apply(le.fit_transform)
    
    outcome = outcome[0]
    df[outcome] = y[outcome].tolist()
    df['index'] = df.index
    
    kf = KFold(n_splits=folds,
                shuffle=True,
                random_state=1,
                )
    PRED = []
    INDS = []
    
    best_accuracy = 0
    best_model = str()
    best_scaler = str()
    
    ct = 0
    for train_index, test_index in kf.split(df):
        ct += 1
        
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        y_train = train_df.filter(items = [outcome], axis = 1)
        x_train = train_df.drop(columns = ['index', outcome], axis = 1)
                
        y_test = test_df.filter(items = [outcome], axis = 1)
        x_test = test_df.drop(columns = ['index', outcome], axis = 1)
        
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
                
        early_stopping = EarlyStopping(patience=patience, monitor='loss')
                 
        # get the model
        n_features = x_train.shape[1]
        
        # define model
        model = Sequential()
        
        # define first layer, add dropout
        model.add(Dense(inodes, input_dim=n_features, activation='relu', kernel_initializer='uniform'))
        #model.add(Dropout(0.5))
        
        if hidden_layers > 0:
            for j in list(range(hidden_layers)):
                model.add(Dense(nodes, activation='relu', kernel_initializer='uniform'))
                #model.add(Dropout(0.5))
            
        # define output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # define loss and optimizer
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate) #rho=0.95, epsilon=1e-07,
        
        model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                
        # fit model
        weights = class_weight.compute_class_weight('balanced',
                    np.unique(y_train[outcome]), y_train[outcome])
        weights = {i : weights[i] for i in range(len(list(set(y_train[outcome]))))}
        
        model.fit(x_train,
                  y_train,
                  class_weight = weights,
                  epochs=epochs,
                  verbose=0,
                  batch_size=batch_size,
                  callbacks=[early_stopping],
                  shuffle=False,
                )

        np.set_printoptions(precision=4, suppress=True)
        eval_results = model.evaluate(x_test, y_test, verbose=0)
        print("\n")
        print(ct, ": Loss, accuracy on test data: ")
        print("%0.4f %0.2f%%" % (eval_results[0], eval_results[1]*100))
        
        if eval_results[1]*100 > best_accuracy:
            best_model = model
            best_accuracy = eval_results[1]*100
            best_scaler = scaler
            
    # Use best performing model to get predictions for entire dataset
    x_test = df.drop(columns = ['index', outcome], axis = 1)
    x_test = best_scaler.transform(x_test)
    
    pred = best_model.predict(x_test)
    pred = pred.flatten()
    pred = np.array(pred)
    df['prediction'] = list(pred)
        
    del model
    del scaler
    gc.collect()
    K.clear_session()
    
    tpr_ls = []
    tnr_ls = []
    fpr_ls = []
    fnr_ls = []
    ppv_ls = []
    npv_ls = []
    ct_ls = []
    tp_ls = []
    fp_ls = []
    tn_ls = []
    fn_ls = []
    n_ls = []

    cut_ls = np.linspace(-0.01, 1.01, 1000).tolist()
    for c in cut_ls:
        
        tp, fp, tn, fn = 0,0,0,0
        tpr, fpr, fnr, tnr = 0,0,0,0
        ppv, npv = 0, 0

        obs = df[outcome].values.tolist()
        exp = df['prediction'].values.tolist()
        
        for i, o in enumerate(obs):
                
            e = exp[i]
            
            if o == 0 and e < c:
                tn += 1
            elif o == 0 and e >= c:
                fp += 1
            elif o == 1 and e >= c:
                tp += 1
            elif o == 1 and e < c:
                fn += 1
        
        if tp + fn > 0:
            tpr = tp/(tp + fn)
            fnr = fn/(tp + fn)
        else:
            tpr = np.nan
            fnr = np.nan
        
        if fp + tn > 0:
            fpr = fp/(fp + tn)
            tnr = tn/(fp + tn)
        else:
            fpr = np.nan
            tnr = np.nan
        
        if tp + fp > 0:
            ppv = tp/(tp + fp)
        else:
            ppv = np.nan
            
        if tn + fn > 0:
            npv = tn/(tn + fn)
        else:
            npv = np.nan
        
        tpr_ls.append(tpr)
        fpr_ls.append(fpr)
        tnr_ls.append(tnr)
        fnr_ls.append(fnr)
        ppv_ls.append(ppv)
        npv_ls.append(npv)
        ct_ls.append(np.round(c,6))
        
        tp_ls.append(tp)
        fp_ls.append(fp)
        tn_ls.append(tn)
        fn_ls.append(fn)
        n_ls.append(tp + fp + tn + fn)
        
    n_ls  = ['N: ' + str(s) for s in n_ls]
    tp_ls = ['TP: ' + str(s) for s in tp_ls]
    fp_ls = ['FP: ' + str(s) for s in fp_ls]
    tn_ls = ['TN: ' + str(s) for s in tn_ls]
    fn_ls = ['FN: ' + str(s) for s in fn_ls]
    ct_ls = ['Threshold: ' + str(s) for s in ct_ls]

    
    ddf = pd.DataFrame(columns=['TPR'])
    ddf['TPR'] = tpr_ls
    ddf['TNR'] = tnr_ls
    ddf['FPR'] = fpr_ls
    ddf['FNR'] = fnr_ls
    ddf['PPV'] = ppv_ls
    ddf['NPV'] = npv_ls
    ddf['threshold'] = ct_ls
    ddf['TP'] = tp_ls
    ddf['FP'] = fp_ls
    ddf['TN'] = tn_ls
    ddf['FN'] = fn_ls
    ddf['N'] = n_ls
    
    df.drop(labels='index', axis=1, inplace=True)
    return df.to_json(), ddf.to_json()

