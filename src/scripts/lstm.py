import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import os
import torch
import pathlib
import logging

from src.data import get_dataloaders
from src.models import VanillaLSTM
from src.utils import generate_unique_logpath, setup_logging, ModelCheckpoint, train_one_epoch, validate, get_loss


def create_sequences(data, params):
    TARGET, WINDOW, HORIZON = params
    input, output = [], []
    if isinstance(data, pd.DataFrame):
        for idx in range(0, len(data) - WINDOW - HORIZON + 1):
            input.append(data[TARGET][idx:idx + WINDOW]) # de t à t+w
            output.append(data[TARGET].iloc[idx + WINDOW:idx + WINDOW + HORIZON]) # de t+w à t+w+h

    elif isinstance(data, np.ndarray):
        for idx in range(0, len(data) - WINDOW - HORIZON + 1):
            input.append(data[idx:idx + WINDOW, 0])
            output.append(data[idx + WINDOW:idx + WINDOW + HORIZON, 0])
    return np.asarray(input), np.asarray(output)

def splitting(X, y, n_test):
    X  = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    X_train, X_val, X_test = X[:-2*n_test], X[-2*n_test:-n_test], X[-n_test:]
    y_train, y_val, y_test = y[:-2*n_test], y[-2*n_test:-n_test], y[-n_test:]

    assert len(X_train)+len(X_test)+len(X_val) == len(X)
    assert len(y_train)+len(y_val)+len(y_test) == len(y)
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test

def lstm(cfg, df, device):
    df.drop(columns=["HUFL","HULL"], inplace=True) # analyse manuelle de la matrice de corrélation
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # ----------------- Univariate -----------------
    TESTSIZE =          cfg["TESTSIZE"]
    WINDOW =            cfg["WINDOW"]
    HORIZON =           cfg["HORIZON"]
    TARGET =            cfg["TARGET"]
    EPOCHS =            cfg["EPOCHS"]
    BATCH_SIZE =        cfg["BATCH_SIZE"]
    NB_OF_FEATURES =    cfg["NB_OF_FEATURES"]
    HIDDEN_SIZE_LAYER = cfg["HIDDEN_SIZE_LAYER"]
    WEIGHT_DECAY =      cfg["weight_decay"]
    LR =                cfg["lr"]
    LOSSNAME =          cfg["loss"]

    NUM_WORKERS = os.cpu_count()-2 if os.cpu_count() > 4 else 1
    if NB_OF_FEATURES > 1:
        TS_MODE = "MULTIVARIATE"
    else:
        TS_MODE = "UNIVARIATE"
    CONFIG = f"{TS_MODE}_TESTSIZE{TESTSIZE}_WINDOW{WINDOW}_HORIZON{HORIZON}"

    # ***** DATA *****
    univariate = pd.DataFrame(df[TARGET].copy())
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(univariate)
    X, y = create_sequences(scaled_data, [TARGET, WINDOW, HORIZON])
    X = X.reshape(X.shape[0], X.shape[1], 1) # reshape for LSTM input : (n,WINDOW) --> (n,WINDOW,1)
    print("X' shape:",X.shape,"Y' shape :",y.shape)
    X_train, X_val, X_test, y_train, y_val, y_test = splitting(X,y,TESTSIZE)
    print("On a bien des séquences [lag-w, ..., lag-1], ce qui est le bon sens pour le LSTM.") if y_train[0] == X_train[1][WINDOW-1] else print("Les séquences sont à l'envers, c'est à dire [lag-1, ..., lag-w]")
    train_loader, val_loader, test_loader = get_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, [BATCH_SIZE, NUM_WORKERS])

    # ***** MODEL *****
    model = VanillaLSTM(input_size=NB_OF_FEATURES, hidden_size=HIDDEN_SIZE_LAYER, num_stacked_layer=1, output_size=HORIZON).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss  = get_loss(LOSSNAME)

    # ***** CALLBACKS *****
    cfg_logging = cfg["logging"]
    os.makedirs(cfg_logging["checkpoints"], exist_ok=True)
    os.makedirs(cfg_logging["figures"], exist_ok=True)
    os.makedirs(cfg_logging["predictions"], exist_ok=True)
    os.makedirs(cfg_logging["scores"], exist_ok=True)

    logname = f"LSTM_TEST_{CONFIG}"
    logdir = generate_unique_logpath(cfg_logging["checkpoints"], logname)
    setup_logging(logdir, mode="train")
    logdir = pathlib.Path(logdir)
    checkpoint = ModelCheckpoint(model, str(logdir / "best_model.pt"), min_is_best=True)

    # ***** TRAINING *****
    for epoch in tqdm.tqdm(range(EPOCHS)):
        train_loss = train_one_epoch(model, train_loader, loss, optim, device)
        val_loss = validate(model, val_loader, loss, device)
        logging.info(f"Epoch {epoch+1},\tTrain Loss : {train_loss:.5f},\tVal Loss : {val_loss:.5f}")
        
        updated = checkpoint.update(val_loss)
        if updated:
            logging.info("New best model saved!")

    # ***** PREDICTIONS & PLOTS *****
    with torch.no_grad():
        y_pred_train = model(X_train.to(device)).detach().to('cpu').numpy()
        y_pred_test= model(X_test.to(device)).detach().to('cpu').numpy()

    y_pred_train_rescaled = scaler.inverse_transform(y_pred_train)
    y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
    plt.figure(figsize = (15,5))
    plt.plot(df.index[:len(y_train_rescaled)], y_train_rescaled, color='grey', alpha=1, label="Actual temperature")
    plt.plot(df.index[:len(y_train_rescaled)], y_pred_train_rescaled, color ='orange', label="Predicted temperature")
    plt.xlabel('Date')
    plt.ylabel("Oil Temperature in Celsius")
    plt.title("Predictions des températures du TRAIN SET")
    plt.legend()
    plt.savefig(cfg_logging["figures"] + f"train_lstm_{CONFIG}.png")


    y_pred_test_rescaled = scaler.inverse_transform(y_pred_test)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    shift_value = np.abs(min(min(y_test_rescaled), min(y_pred_test_rescaled))) + 1
    mae = round(mean_absolute_error(y_test_rescaled, y_pred_test_rescaled),3)
    rmse = round(np.sqrt(mean_squared_error(y_test_rescaled, y_pred_test_rescaled)),3)
    mape = round(np.mean((np.abs((y_test_rescaled+shift_value) - (y_pred_test_rescaled+shift_value))/(y_test_rescaled+shift_value))*100),3)
    logging.info(f"\nTest metrics:\n\t- MAE:{mae}\n\t- RMSE:{rmse}\n\t- MAPE:{mape}%")

    if TESTSIZE == 24:
        y_rescaled = scaler.inverse_transform(y.reshape(-1, 1))
        plt.figure(figsize = (15,5))
        plt.plot(df.index[-100:], y_rescaled[-100:], color="grey", label="Actual temperature\n")
        plt.plot(df.index[-len(y_test_rescaled):], y_pred_test_rescaled, color="orange", label=f"Predicted temperature\nMAE:{mae}\nRMSE:{rmse}\nMAPE:{mape}%")
        plt.axvline(df.index[-len(y_test_rescaled):].min(), color="black", linestyle='--', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel("Oil Temperature in Celsius")
        plt.title("Predictions des températures du TEST SET")
        plt.legend()
        plt.savefig(cfg_logging["figures"] + f"test_lstm_{CONFIG}.png")
        dates = univariate.index[-len(y_pred_test_rescaled):]
        pred_series = pd.Series(y_pred_test_rescaled.flatten(), index=dates, name="prediction") # from numpy array to pandas Series (Timestamps|Values)
        pred_series.to_frame(name=f"LSTM_{CONFIG}_prediction").to_parquet(cfg_logging["predictions"] + f"LSTM_{CONFIG}_prediction.parquet") # save preds
    else:
        plt.figure(figsize = (15,5))
        plt.plot(df.index[-len(y_test_rescaled):], y_test_rescaled, label="Actual temperature\n")
        plt.plot(df.index[-len(y_test_rescaled):], y_pred_test_rescaled, label=f"Predicted temperature\nMAE:{mae}\nRMSE:{rmse}\nMAPE:{mape}%")
        plt.xlabel('Date')
        plt.ylabel("Oil Temperature in Celsius")
        plt.title(f"Predictions des températures du TEST SET ({X_test.size(0)}h)")
        plt.legend()
        plt.savefig(cfg_logging["figures"] + f"test_lstm_{CONFIG}.png")
        dates = univariate.index[-len(y_pred_test_rescaled):]
        pred_series = pd.Series(y_pred_test_rescaled.flatten(), index=dates, name="prediction") # from numpy array to pandas Series (Timestamps|Values)
        pred_series.to_frame(name=f"LSTM_{CONFIG}_prediction").to_parquet(cfg_logging["predictions"] + f"LSTM_{CONFIG}_prediction.parquet") # save preds


    y_val_rescaled = scaler.inverse_transform(y_val.reshape(-1, 1))
    plt.figure(figsize = (15,5))
    plt.plot(df.index[WINDOW:len(y_train_rescaled)+WINDOW], y_train_rescaled, color='blue', alpha=1, label="Train (Actual)")
    plt.plot(df.index[-2*len(y_val_rescaled):-len(y_val_rescaled)], y_val_rescaled, color="purple", label="Validation (Actual)")
    plt.plot(df.index[-len(y_test_rescaled):], y_test_rescaled, color="orange", label="Test (Actual)")
    plt.plot(df.index[-len(y_test_rescaled):], y_pred_test_rescaled, color="red", alpha=0.8, label="Predictions")
    plt.xlabel('Date')
    plt.ylabel("Oil Temperature in Celsius")
    plt.title("Predictions des températures du TEST SET (avec TRAIN/VAL SET)")
    plt.legend()
    plt.savefig(cfg_logging["figures"] + f"train-val-test_lstm_{CONFIG}.png")

    scores_path = cfg_logging["scores"] + "lstm_scores.parquet"
    if os.path.exists(scores_path):
        result_table = pd.read_parquet(scores_path)
        if (result_table["Model"] == f"LSTM_{CONFIG}").any():
            print("Cette configuration de LSTM a déjà été renseigné (donc testée), fin du processus.")
        else:
            print("Ajout d'une nouvelle ligne aux scores des LSTM...")
            scores_model = {"Model":f"LSTM_{CONFIG}",
                            "Train_size":X_train.size(0),
                            "Val_size":X_val.size(0),
                            "Test_size":X_test.size(0),
                            "MAE":mae,
                            "RMSE":rmse,
                            "MAPE(%)":mape}
            result_table.loc[result_table.shape[0]] = scores_model
            result_table.sort_values(by=["Test_size", "MAE"], inplace=True)
            result_table.to_parquet(scores_path) # save dataframe
            # Export markdown (lisible)
            with open(f"{scores_path}.md", "w", encoding="utf-8") as f:
                f.write(result_table.to_markdown(index=False))
    else: 
        print("Fichier des scores LSTM inexistant, création.")
        result_table = pd.DataFrame(columns=["Model","Train_size","Val_size","Test_size", "MAE","RMSE","MAPE(%)"])
        scores_model = {"Model":f"LSTM_{CONFIG}", "Train_size":X_train.size(0), "Val_size":X_val.size(0), "Test_size":X_test.size(0), "MAE":mae, "RMSE":rmse, "MAPE(%)":mape}
        result_table.loc[len(result_table)] = scores_model
        result_table.to_parquet(scores_path) # save dataframe
        with open(f"{scores_path}.md", "w", encoding="utf-8") as f:
            f.write(result_table.to_markdown(index=False))