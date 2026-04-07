import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import *

from prophet import Prophet


def linear(cfg, df):
    figures_path = cfg["logging"]["figures"]
    predictions_path = cfg["logging"]["predictions"]
    scores_path = cfg["logging"]["scores"]
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(predictions_path, exist_ok=True)
    os.makedirs(scores_path, exist_ok=True)


    # Data Prep
    df['date'] = pd.to_datetime(df['date'])
    df_prophet = df.copy()
    df_prophet.rename(columns={"date":"ds", "OT": "y"}, inplace = True)
    df.set_index('date', inplace=True)
    counts_per_day = df.resample('D').size()
    freq_data_day = round(counts_per_day.mean())
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    data_reindexed = df.reindex(full_range)
    data_filled = data_reindexed['OT'].interpolate(method='linear').ffill().bfill()
    # Shift + Log, valeur minimale soit à 1.0 (log(1) = 0)
    shift_value = abs(data_filled.min()) + 1.0
    data_transformed = np.log(data_filled + shift_value)

    # Decomposition
    period = freq_data_day # journalière
    stl = STL(data_transformed, period=int(period), seasonal=17, trend=freq_data_day*28+1).fit()
    data_desais = data_transformed - stl.seasonal
    data_arma_ready = data_desais.diff().dropna()



    # ----------------- Estimation du modèle entrainé ARIMA sur FULL-24 -----------------
    testsize = 24
    train_data = data_arma_ready.iloc[:-testsize]
    test_actuals = df['OT'].iloc[-testsize:] #série originale

    model_fit = SARIMAX(train_data, order=(1, 0, 1)).fit(disp=False)
    forecast_diff = model_fit.forecast(steps=testsize)

    # On récupère la dernière valeur connue de data_desais avant le test pour inverser la différenciation
    last_val_desais = data_desais.iloc[-(testsize + 1)]
    forecast_desais = forecast_diff.cumsum() + last_val_desais
    # On récupère et réintroduit les composantes saisonnières correspondantes aux dates du test
    seasonal_part = stl.seasonal.iloc[-testsize:]
    forecast_log = forecast_desais + seasonal_part.values
    # Inversion du log et du shift
    forecast_final = np.exp(forecast_log) - shift_value

    rmse = np.sqrt(mean_squared_error(test_actuals, forecast_final))
    mape = np.mean(np.abs((test_actuals - forecast_final) / test_actuals)) * 100
    mae  = mean_absolute_error(test_actuals, forecast_final)

    plt.figure(figsize=(15, 6))
    plt.plot(df.index[-100:], df['OT'][-100:], label='Série Originale\n', color='tab:blue', alpha=0.7)
    plt.plot(test_actuals.index, forecast_final, label=f'Prévision ({testsize}h)\nMAE:{mae:.4f}\nRMSE:{rmse:.4f}\nMAPE:{mape:.2f}%', color='red', linewidth=1.5)
    plt.title(f"Prévisions ARIMA(1,0,1) sur les {testsize} dernières heures (Échelle Originale)")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_path + "ARIMA.png")

    result_table = pd.DataFrame(columns=["Model","Train_size","Val_size","Test_size", "MAE","RMSE","MAPE(%)"])
    scores_model = {"Model":"ARIMA", "Train_size":train_data.shape[0], "Val_size":0, "Test_size":testsize, "MAE":mae, "RMSE":rmse, "MAPE(%)":mape}
    result_table.loc[len(result_table)] = scores_model
    forecast_final.to_frame(name="ARIMA_prediction").to_parquet(predictions_path + "ARIMA_prediction.parquet") # save preds



    # ----------------- Estimation du modèle entrainé SARIMA sur FULL-24 -----------------
    start_train = 6000
    train = df.iloc[start_train:-testsize] # On coupe des vieilles données car explosion de la RAM avec la saisonnalité à 24
    y_true = df.iloc[-testsize:]['OT']
    model = SARIMAX(train["OT"],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    result = model.fit(disp=False)

    forecast = result.forecast(steps=testsize)

    rmse = np.sqrt(mean_squared_error(y_true, forecast))
    mae = mean_absolute_error(y_true, forecast)
    mape = np.mean(np.abs((y_true - forecast) / y_true)) * 100

    plt.figure(figsize=(15, 6))
    plt.plot(df.index[-100:], df['OT'][-100:], label='Série Originale\n', color='tab:blue', alpha=0.7)
    plt.plot(forecast.index, forecast, label=f'Prévision ({testsize}h)\nMAE:{mae:.4f}\nRMSE:{rmse:.4f}\nMAPE:{mape:.2f}%)', color='red', linewidth=1.5)
    plt.title(f"Prévisions SARIMA(1, 1, 1)x(1, 1, 1, 24) sur les {testsize} dernières heures (Échelle Originale)")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_path + "SARIMAX_1.png")

    scores_model = {"Model":"SARIMAX_1", "Train_size":train.shape[0], "Val_size":0, "Test_size":testsize, "MAE":mae, "RMSE":rmse, "MAPE(%)":mape}
    result_table.loc[len(result_table)] = scores_model
    forecast.to_frame(name="SARIMAX_1_prediction").to_parquet(predictions_path + "SARIMAX_1_prediction.parquet") # save preds


    # ----------------- Estimation du modèle entrainé SARIMA-GridSearched sur FULL-24 -----------------
    best_params_AIC, best_seasonal_AIC = (1, 1, 1), (1, 0, 1, 24)

    # best_params_AIC, best_seasonal_AIC, _, _ = sarima_gridsearch(train)
    final_model_AIC = SARIMAX(train["OT"], order=best_params_AIC, seasonal_order=best_seasonal_AIC).fit()

    forecast_diff_AIC = final_model_AIC.forecast(steps=testsize)
    rmse = np.sqrt(mean_squared_error(y_true, forecast_diff_AIC))
    mae  = mean_absolute_error(y_true, forecast_diff_AIC)
    mape = np.mean(np.abs((y_true - forecast_diff_AIC) / y_true)) * 100

    plt.figure(figsize=(15, 6))
    plt.plot(df.index[-100:], df['OT'][-100:], label='Série Originale\n', color='tab:blue', alpha=0.7)
    plt.plot(forecast_diff_AIC.index, forecast_diff_AIC, label=f'Prévision ({testsize}h)\nMAE:{mae:.4f}\nRMSE:{rmse:.4f}\nMAPE:{mape:.2f}%', color='red', linewidth=1.5)
    plt.title(f"Prévisions SARIMA{best_params_AIC}{best_seasonal_AIC} sur les {testsize} dernières heures (Échelle Originale)")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_path + "SARIMAX_2_GridSearched.png")

    scores_model = {"Model":"SARIMAX_2_GridSearched", "Train_size":train.shape[0], "Val_size":0, "Test_size":testsize, "MAE":mae, "RMSE":rmse, "MAPE(%)":mape}
    result_table.loc[len(result_table)] = scores_model
    forecast_diff_AIC.to_frame(name="SARIMAX_2_GridSearched_prediction").to_parquet(predictions_path + "SARIMAX_2_GridSearched_prediction.parquet") # save preds



    # ----------------- Prophet by Meta -----------------
    train  = df_prophet.iloc[:-testsize,:][["ds","y"]]
    test   = df_prophet.iloc[-testsize:,:][["ds","y"]]

    m = Prophet().fit(train)
    forecast = m.predict(test)
    rmse = np.sqrt(mean_squared_error(test["y"].values, forecast["yhat"].values))
    mape = np.mean(np.abs((test["y"].values - forecast["yhat"].values) / test["y"].values)) * 100
    mae  = mean_absolute_error(test["y"].values, forecast["yhat"].values)

    plt.figure(figsize=(15, 6))
    plt.plot(df_prophet["ds"][-100:], df_prophet["y"][-100:].values, label='Série Originale\n', color='tab:blue', alpha=0.7)
    plt.plot(test["ds"].values, forecast["yhat"].values, label=f'Prévision ({testsize}h)\nMAE:{mae:.4f}\nRMSE:{rmse:.4f}\nMAPE:{mape:.2f}%)', color='red', linewidth=1.5)
    plt.plot(test["ds"].values, forecast["yhat_lower"].values, label=f'Lower 95% interval ({testsize}h)', color='green', linewidth=1.5)
    plt.plot(test["ds"].values, forecast["yhat_upper"].values, label=f'Upper 95% interval ({testsize}h', color='green', linewidth=1.5)
    plt.title(f"Prévisions de Prophet sur les {testsize} dernières heures (Échelle Originale)")
    plt.xlabel("Date")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_path + "Prophet.png")

    scores_model = {"Model":"Prophet", "Train_size":train.shape[0], "Val_size":0, "Test_size":testsize, "MAE":mae, "RMSE":rmse, "MAPE(%)":mape}
    result_table.loc[len(result_table)] = scores_model
    result_table.to_parquet(scores_path + "linear_scores.parquet") # save dataframe
    forecast["yhat"].to_frame(name="prophet_prediction").to_parquet(predictions_path + "prophet_prediction.parquet") # save preds

    with open(f"{scores_path}linear_scores.parquet.md", "w", encoding="utf-8") as f:
        f.write(result_table.to_markdown(index=False))



from statsmodels.tsa.statespace.sarimax import *
from tqdm import tqdm
import itertools
import warnings

def sarima_gridsearch(train):
    warnings.filterwarnings(action='ignore')

    # on génère toutes les combinaisons possibles des paramèters
    p = d = q = P = D = Q = range(2)
    params = list(itertools.product(p, d, q, P, D, Q))

    best_aic, best_bic = np.inf, np.inf
    best_params_AIC, best_params_BIC = None, None
    best_seasonal_AIC, best_seasonal_BIC = None, None

    for p,d,q,P,D,Q in tqdm(params):
        try:
            tmp_model = SARIMAX(train["OT"],
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 24),
                                enforce_stationarity=True,
                                enforce_invertibility=True)
            
            # disp=False et low_memory=True pour économiser la RAM
            res = tmp_model.fit(disp=False, low_memory=True)

            if res.aic < best_aic:
                best_aic = res.aic
                best_params_AIC = (p, d, q)
                best_seasonal_AIC = (P, D, Q, 24)
            if res.bic < best_bic:
                best_bic = res.bic
                best_params_BIC = (p, d, q)
                best_seasonal_BIC = (P, D, Q, 24)
        except Exception as e:
            continue
    return best_params_AIC, best_seasonal_AIC, best_params_BIC, best_seasonal_BIC