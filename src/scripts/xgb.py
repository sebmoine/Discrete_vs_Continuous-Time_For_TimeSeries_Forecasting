import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import random
from sklearn.multioutput import MultiOutputRegressor


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

def xgboost(cfg, df):
    # Data Preparation
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    counts_per_day = df.resample('D').size()
    freq_data_day = round(counts_per_day.mean())
    df.drop(columns=["HUFL","HULL"], inplace=True) # analyse manuelle matrice de corrélation

    mode = cfg["MULTIVARIATE"]
    cfg_logging = cfg["logging"]
    os.makedirs(cfg_logging["figures"], exist_ok=True)
    os.makedirs(cfg_logging["predictions"], exist_ok=True)
    os.makedirs(cfg_logging["scores"], exist_ok=True)
    os.makedirs(cfg_logging["checkpoints"], exist_ok=True)


    assert isinstance(mode,bool), "Parameter 'MULTIVARIATE' must be a boolean."
    xgb_multivariate(cfg, df) if mode else xgb_univariate(cfg, df)

def preprocessing(df_, num_periods_input, num_periods_output, test_size=24 * 120):  # Soit 120j à prédire car h=24, ~16.5% du dataset (référence pour le dataset)
    """
        - Reprise de la méthode générale du code du papier, en faisant directement l'opération pour ne conserver que la dernière valeur des covariables (vs en post-process dans le code original)
        - Des suppressions de reshaping inutiles
    """
    df_['DayofWeek']=df_.index.dayofweek
    df_['Week']=df_.index.isocalendar().week
    df_['dayofyear']=df_.index.dayofyear
    Train=df_.iloc[:len(df_) - test_size]
    Test=df_.iloc[len(df_) - test_size:]

    #################################################################################
    TARGET = "OT"
    COVARIABLES = df_.drop(columns=[TARGET], inplace=False).columns
        
    #############################  Normalization on train  #############
    X_train = Train[COVARIABLES]
    y_train = Train[TARGET]

    normalizer = MinMaxScaler().fit(X_train)
    X_train=normalizer.transform(X_train)

    y_train=np.reshape(y_train,(len(y_train),1))
    Train=np.append(y_train, X_train, axis=1)   #rajout de la target après normalisation des covariables, unique set d'entrainement
    
    ############################################ TRAIN windows ##################################
    end=len(Train)
    start=0
    next=0
    x_batches=[]
    y_batches=[]
    
    limit = num_periods_output + num_periods_input
    while start + limit <= end:
        next = start + num_periods_input
        
        # lags de y uniquement (colonne 0)
        y_lags = Train[start:next, 0]
        
        # covariables à t (dernière ligne de la fenêtre, colonnes 1:) on exclut target indice 0
        X_t = Train[next-1, 1:]
        
        # concat → vecteur plat
        x_batches.append(np.concatenate([y_lags, X_t]))
        y_batches.append(Train[next:next+num_periods_output, 0])
        start += 1

    x_batches=np.asarray(x_batches) # x_batches size   = N_windows
                                    # One window size = (w,num_features)
                                    # Now same but NumPy Array (instead of a Python list)
    y_batches=np.asarray(y_batches)
    
    ###########################################TEST Normalization##################################
    y_test=Test[TARGET]
    X_test=Test[COVARIABLES]

    X_test=normalizer.transform(X_test) 

    y_test=np.reshape(y_test, (len(y_test), 1))
    Test=np.append(y_test, X_test, axis=1)
    ############################################ TEST windows ##################################
    end_test=len(Test)
    start_test=0
    next_test=0
    x_testbatches=[]
    y_testbatches=[]

    while start_test+(limit)<=end_test:
        next_test = start_test + num_periods_input
        y_lags = Test[start_test:next_test, 0]
        X_t = Test[next_test-1, 1:]
        x_testbatches.append(np.concatenate([y_lags, X_t]))
        y_testbatches.append(Test[next_test:next_test + num_periods_output, 0])
        start_test = start_test+num_periods_output #incrémente par l'horizon

    x_testbatches=np.asarray(x_testbatches)
    y_testbatches=np.asarray(y_testbatches)
    
    return x_batches, y_batches, x_testbatches, y_testbatches

def xgb_multivariate(cfg, df):
    """
        XGBoostWB_Forecasting_Using_Hybrid_DL_Framework_Pm2.5_(1,6)
        source : https://github.com/Daniela-Shereen/GBRT-for-TSF/tree/main
    """
    TESTSIZE = cfg["TESTSIZE"]
    VALSIZE = cfg["VALSIZE"]
    N_FOLDS = cfg["N_FOLDS"]
    HORIZON = cfg["HORIZON"]
    WINDOW = cfg["WINDOW"]

    num_periods_output  = HORIZON # HORIZON TO PREDICT
    num_periods_input   = WINDOW  # w, the WINDOW
    ALL_Test_Data, ALL_Test_Prediction =[], []

    data = df.copy()
    All_Training_Instances, y_batches, All_Testing_Instances, Y_Test = preprocessing(data, num_periods_input, num_periods_output, test_size=TESTSIZE + WINDOW)

    #---------------------shuffle minibatches X and Y together-------------------------------------
    combined = list(zip(All_Training_Instances, y_batches))
    random.seed(42)
    random.shuffle(combined)
    shuffled_batch_features, shuffled_batch_y = zip(*combined)

    # Cross-Validation
    tscv = TimeSeriesSplit(n_splits=N_FOLDS, test_size=VALSIZE, gap=24)

    base_model = xgb.XGBRegressor()
    multi_model = MultiOutputRegressor(base_model)

    param_dist = {
        'estimator__n_estimators': [100, 200, 400],
        'estimator__subsample': [0.8, 1],
        'estimator__colsample_bytree': [0.8, 1],
        'estimator__learning_rate': [0.1, 0.01],
        'estimator__max_depth': [1, 2],
        'estimator__min_samples_split': [1, 2, 3],
        'estimator__scale_pos_weight': [0.8, 1],
        'estimator__n_jobs': [-1],
        'estimator__random_state': [42],

    }

    random_search2 = RandomizedSearchCV(
        multi_model, 
        param_distributions=param_dist, 
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_iter=10,
        n_jobs=-1
    )

    random_search2.fit(shuffled_batch_features, shuffled_batch_y)
    best_params2 = random_search2.best_params_
    clean_params2 = {k.replace('estimator__', ''): v for k, v in best_params2.items()}
    print("Paramètres optimaux extraits :", clean_params2)

def create_features(df):
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)   
    df['dayofweek'] = df.index.dayofweek
    df['quarter']   = df.index.quarter
    df["month"]     = df.index.month
    df["year"]      = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["dayofmonth"]= df.index.day
    df["weekofyear"]= df.index.isocalendar().week
    return df

def add_lags(df, TARGET):
    target_map = df[TARGET].to_dict()
    df["lag1"] = (df.index - pd.Timedelta("1 hour")).map(target_map)
    df["lag2"] = (df.index - pd.Timedelta("2 hours")).map(target_map)
    df["lag3"] = (df.index - pd.Timedelta("3 hours")).map(target_map)
    df["lag4"] = (df.index - pd.Timedelta("4 hours")).map(target_map)
    df["lag5"] = (df.index - pd.Timedelta("5 hours")).map(target_map)
    df["lag6"] = (df.index - pd.Timedelta("6 hours")).map(target_map)
    df["lag7"] = (df.index - pd.Timedelta("7 hours")).map(target_map)
    df["lag24"] = (df.index - pd.Timedelta("24 hours")).map(target_map)
    df["lag48"] = (df.index - pd.Timedelta("48 hours")).map(target_map) #2j
    df["lag168"] = (df.index - pd.Timedelta("168 hours")).map(target_map) # 7j
    return df

def xgb_univariate(cfg, df):
    # ---------- UNIVARIATE ----------
    cfg_logging = cfg["logging"]
    TESTSIZE = cfg["TESTSIZE"]
    VALSIZE = cfg["VALSIZE"]
    N_FOLDS = cfg["N_FOLDS"]
    TARGET = "OT"

    univariate = pd.DataFrame(df[TARGET].copy())

    # Feature Engineering
    univariate = create_features(univariate)
    univariate = add_lags(univariate, TARGET)

    # Train, Val & Test Splitting
    univariate_train = univariate.iloc[TESTSIZE:]
    univariate_test  = univariate.iloc[:TESTSIZE]
    tss = TimeSeriesSplit(n_splits=N_FOLDS, test_size = VALSIZE, gap=24) #test_size = 2880 heures (référence du dataset) | gap simule le délai réel de disponibilité des données en production, disons ici 12h


    # Finetuning with GridSearch & Cross-Validation by Forward-Walk
    y = TARGET
    X = list(univariate_train.drop(columns=[y]).columns)

    base_model = xgb.XGBRegressor()
    param_dist = {
        'estimator__n_estimators': [300, 400, 500],
        'estimator__subsample': [0.8, 1],
        'estimator__colsample_bytree': [0.6, 0.8, 1],
        'estimator__learning_rate': [0.1],
        'estimator__max_depth': [1, 2],
        'estimator__min_samples_split': [2, 3, 4],
        'estimator__scale_pos_weight': [0.8, 1],
        'estimator__n_jobs': [-1],
        'estimator__random_state': [42],
    }
    random_search_1 = RandomizedSearchCV(
        base_model, 
        param_distributions=param_dist, 
        cv=tss,
        scoring='neg_mean_squared_error',
        n_iter=100,
        n_jobs=-1)

    train = univariate_train.iloc[:-VALSIZE]
    test  = univariate_train.iloc[-VALSIZE:]
    X_train = train[X]
    y_train = train[y]
    X_test = test[X]
    y_test = test[y]

    random_search_1.fit(X_train, y_train)
    best_params1 = random_search_1.best_params_
    clean_params1 = {k.replace('estimator__', ''): v for k, v in best_params1.items()}
    print("Paramètres optimaux extraits :", clean_params1)

    # Training and Predictions
    preds, scores = [], []
    for train_idx, test_idx in tss.split(univariate_train):
        train = univariate_train.iloc[train_idx]
        test  = univariate_train.iloc[test_idx]
        X_train = train[X]
        y_train = train[y]
        X_test = test[X]
        y_test = test[y]

        regression = xgb.XGBRegressor(**clean_params1)
        regression.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=20)
        y_pred = regression.predict(X_test)
        preds.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)


    # MAPEs of each validation fold
    shift_value = np.abs(np.min(y_test)) + 1
    y_test_shifted = y_test + shift_value
    MAPEs = []
    for idx, (train_idx, test_idx) in enumerate(tss.split(univariate_train)):
        # Because some values of Y_Test are 0 or very close, we shift the series.
        preds_shifted = preds[idx] + shift_value
        y_test_shifted = univariate_train.iloc[test_idx][TARGET] + shift_value
        mape = np.mean(np.abs((y_test_shifted - preds_shifted) / y_test_shifted)) * 100
        MAPEs.append(mape)


    # Plot predictions of each fold
    plt.figure(figsize=(16, 8))
    plt.plot(univariate.index, univariate[TARGET], label='Série Originale', color='gray', alpha=0.5)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    fold = 0
    for idx, (train_idx, test_idx) in enumerate(tss.split(univariate_train)):
        val_dates = univariate_train.index[test_idx]
        y_pred = preds[fold]

        plt.plot(val_dates, y_pred, 
                color=colors[fold], 
                label=f'Prédiction Fold {fold+1}\nMAPE:{round(MAPEs[idx],3)}%\n', 
                linewidth=2)
        plt.axvline(val_dates.min(), color=colors[fold], linestyle='--', alpha=0.5)
        fold += 1
    plt.title(f"Superposition des prédictions des Folds (Mean MAPE of the {N_FOLDS}-folds: {round(np.mean(MAPEs),3)}%)", fontsize=15)
    plt.xlabel("date")
    plt.ylabel("TARGET")
    plt.xlim(pd.Timestamp('2017-01-01'), univariate_train.index.max())
    plt.legend(loc='best', ncol=2)
    plt.grid(True, alpha=0.2)
    plt.savefig(cfg_logging["figures"] + "xgb_UNIVARIATE_preds_val-folds.png")


    # plot feature importances of the model on the last fold
    pd.DataFrame(data=regression.feature_importances_,
             index=regression.feature_names_in_,
             columns=["importance"]).sort_values("importance").plot(kind='barh')
    plt.savefig(cfg_logging["figures"] + f"xgb_UNIVARIATE_lastFold_VALSIZE{VALSIZE}_feature-importances.png")


    # Training the cross-validated model on full train set
    X_train = univariate_train[X]
    y_train = univariate_train[y]

    X_test = univariate_test[X]
    y_test = univariate_test[y]

    regression = xgb.XGBRegressor(**clean_params1)
    regression.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=False)


    # Predictions on the last 24th hours
    y_pred = regression.predict(X_test)

    ONE_DAY_VALSIZE = 24
    ONE_DAY_dates = univariate.index[-ONE_DAY_VALSIZE:]
    ONE_DAY_RMSE = np.sqrt(mean_squared_error(y_test[-ONE_DAY_VALSIZE:].values, y_pred[-ONE_DAY_VALSIZE:]))
    ONE_DAY_MAPE = np.mean(np.abs((y_test[-ONE_DAY_VALSIZE:].values - y_pred[-ONE_DAY_VALSIZE:]) / y_test[-ONE_DAY_VALSIZE:].values))

    plt.figure(figsize=(15, 6))
    plt.plot(univariate.index, univariate[TARGET], label='Série Originale\n', color='gray', alpha=0.5)
    plt.plot(ONE_DAY_dates, y_pred[-ONE_DAY_VALSIZE:], 
                color="orange",
                label=f'Prédiction\nRMSE: {round(ONE_DAY_RMSE,3)}\nMAPE: {round(ONE_DAY_MAPE,3)}%', 
                linewidth=2)
    plt.axvline(ONE_DAY_dates.min(), color="black", linestyle='--', alpha=0.5)
    plt.title(f"Prédictions {ONE_DAY_VALSIZE}h", fontsize=15)
    plt.xlabel("date")
    plt.ylabel("TARGET")
    plt.xlim(univariate.index.max() - pd.Timedelta("99 hours"), univariate.index.max())
    plt.legend(loc='best', ncol=2)
    plt.grid(True, alpha=0.2)
    plt.ylim(3,14)
    plt.savefig(cfg_logging["figures"] + "xgb_UNIVARIATE_preds_last_24h.png")