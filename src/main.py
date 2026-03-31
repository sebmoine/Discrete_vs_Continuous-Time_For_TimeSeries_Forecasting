import logging
import sys
import time
import torch
import logging
import yaml
import torch
import pandas as pd

from src.utils import print_time
from src.scripts import linear, xgboost, lstm, patchtst#, lnn

import sys
import os
sys.path.append(os.path.abspath("../"))  # ou le bon chemin vers utils
sys.path.append("../models")


def run(config, model_name):
    cfg = config[model_name]

    # CUDA device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Dataset loading
    df = pd.read_csv("/home/sebnm/AIProjects/Forecasting/src/data/ETT/ETTh1.csv")

    # Match the day to predefined patterns
    match model_name:
        case "linear":
            print(f"Loading results for {model_name}...")
            run_start = time.time()
            logging.info("===== TRAINING TIME =====") if logging else print("===== TRAINING TIME =====")
            linear(cfg, df)
            print_time(run_start, time.time())
        case "xgb":
            print(f"Loading results for {model_name}...")
            run_start = time.time()
            logging.info("===== TRAINING TIME =====") if logging else print("===== TRAINING TIME =====")
            xgboost(cfg, df)
            print_time(run_start, time.time())
        case "lstm":
            print(f"Loading results for {model_name}...")
            run_start = time.time()
            logging.info("===== TRAINING TIME =====") if logging else print("===== TRAINING TIME =====")
            lstm(cfg, df, device)
            print_time(run_start, time.time())
        case "patchtst":
            print(f"Loading results for {model_name}...")
            run_start = time.time()
            logging.info("===== TRAINING TIME =====") if logging else print("===== TRAINING TIME =====")
            patchtst(cfg)
            print_time(run_start, time.time())
        # case "lnn":
        #     print(f"Loading results for {model_name}...")
        #     lnn(cfg, df, device)
        case _:
            print("That's not a valid model / This model is not implemented (yet).")



if __name__ == "__main__":
    # python -m src.main <model_name>
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 2:
        logging.error(f"Usage : {sys.argv[0]} <model_name>")
        sys.exit(-1)

    yaml_path = "src/configs/config.yaml"
    model_name = sys.argv[1] # model_name

    logging.info(f"Loading {yaml_path}")
    config = yaml.safe_load(open(yaml_path, "r"))
    run(config, model_name=model_name.lower())