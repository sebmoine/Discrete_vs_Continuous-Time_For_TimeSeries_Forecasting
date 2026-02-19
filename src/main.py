import logging
import sys
import os
import time
import pathlib
import torch
import logging
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo


from src import data
from src import models
from src import scripts
from src.utils import losses, optim, run, log_checkpoint
from src.scripts import time_fcts


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, val_loader, test_loader, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = losses.get_loss(config["loss"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    logname = model_config["class"]
    logdir = log_checkpoint.generate_unique_logpath(logging_config["logdir"], logname)
    log_checkpoint.setup_logging(logdir, mode="train")

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Wandb
    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"], name=str(logdir).split('/')[-1])
        wandb_log = wandb.log
        wandb_log(config)
        wandb.watch(model, log=None) #log="gradients, log_freq=100" en debug
        wandb.log({"num_parameters": sum(p.numel() for p in model.parameters())})
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {data.log_loader_info(train_loader)}\n"
        + f"Validation : {data.log_loader_info(val_loader)}\n"
        + f"Test : {data.log_loader_info(test_loader)}"
    )
    
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = log_checkpoint.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    logging.info(f"Nombre de GPUs dispobnibles : {torch.cuda.device_count()}")
    logging.info(f"Utilisation de DataParallel: {'DataParallel' in str(type(model))}")

    run_start = time.time()
    run.fit(
        config,
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        device,
        model_checkpoint,
        logdir
    )
    logging.info("===== TRAINING TIME =====")
    time_fcts.print_time(time.time() - run_start)
    wandb.finish()



@torch.no_grad()
def test(config, weights_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    data_config = config["data"]
    model_config = config["model"]
    criterion = losses.get_loss(config["loss"])

    logdir = config["logging"]["logdir"] + '/' + str(weights_path).split('/')[-2]
    log_checkpoint.setup_logging(logdir, mode="test")

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    _, _, test_loader, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)

    model = models.build_model(model_config, input_size, num_classes).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    epoch_loss = 0
    num_samples = 0

    inference_start = time.time()
    for inputs, targets in test_loader:

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        epoch_loss += loss.item() * inputs.size(0)
        num_samples += targets.size(0)

    epoch_loss /= num_samples

    logging.info("===== TESTING TIME =====")
    time_fcts.print_time(time.time() - inference_start)

    logging.info("====== TEST RESULTS ======")
    logging.info("Batch-Ponderated Loss\t: %.3f", epoch_loss)



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} <config.yaml|model_name> <train|test>")
        sys.exit(-1)

    arg = sys.argv[1]       #config.yaml|model_name
    command = sys.argv[2]   #train|test

    if command == "train":
        logging.info(f"Loading {arg}")
        config = yaml.safe_load(open(arg, "r"))
        train(config)
    elif command == "test":
        model_name = arg
        logging.info(f"Searching checkpoint for '{model_name}'...")
        model_dir = scripts.get_latest_model_dir(model_name)
        logging.info(f"Using checkpoint: {model_dir}")
        config_path, weights_path = scripts.get_checkpoint_files(model_dir)
        config = yaml.safe_load(open(config_path, "r"))

        test(config, weights_path)
    else:
        logging.error("Command must be 'train' or 'test'")
        sys.exit(-1)