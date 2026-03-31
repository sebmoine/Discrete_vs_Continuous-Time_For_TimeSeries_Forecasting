import torch
import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()

    epoch_loss = 0
    num_samples = 0

    for inputs, targets in tqdm.tqdm(loader, leave=False):
        inputs  = inputs.to(device,non_blocking=True) #non_blocking : accelère les transferts CPU->GPU
        targets = targets.to(device,non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        num_samples += targets.size(0)

    epoch_loss = epoch_loss / num_samples

    return epoch_loss


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    epoch_loss = 0.0
    num_samples = 0

    for inputs, targets in tqdm.tqdm(loader, leave=False):
        inputs = inputs.to(device,non_blocking=True)
        targets = targets.to(device,non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        epoch_loss += loss.item() * inputs.size(0)
        num_samples += targets.size(0)

    epoch_loss = epoch_loss / num_samples

    return epoch_loss