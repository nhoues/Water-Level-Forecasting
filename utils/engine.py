import gc
from tqdm import tqdm

import pandas as pd


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def loss_fn(y_hat, y):
    return nn.MSELoss()(y_hat, y)


def train_fn(data_loader, model, optimizer, device, verbose):
    """
    computes the model training for one epoch
    """
    model.train()
    tr_loss = 0
    counter = 0
    if verbose:
        losses = AverageMeter()
        tk0 = tqdm(enumerate(data_loader), total=len(data_loader))
    else:
        tk0 = enumerate(data_loader)

    for bi, d in tk0:
        targets = d["target"].to(device, dtype=torch.float).view(-1, 1)
        enc = d["num_feat"].to(device, dtype=torch.float)
        day = d["dayOfweek"].to(device, dtype=torch.long)
        hour = d["hour"].to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(enc, day, hour)
        loss = loss_fn(outputs, targets)

        tr_loss += loss.item()
        counter += 1
        loss.backward()
        optimizer.step()
        if verbose:
            losses.update(loss.item(), targets.size(0))
            tk0.set_postfix(loss=losses.avg)
    return tr_loss / counter


def eval_fn(data_loader, model, device, verbose):
    """
    computes the model evaluation for one epoch
    """
    model.eval()
    fin_loss = 0
    counter = 0
    if verbose:
        losses = AverageMeter()
        tk0 = tqdm(enumerate(data_loader), total=len(data_loader))
    else:
        tk0 = enumerate(data_loader)
    with torch.no_grad():
        for bi, d in tk0:
            targets = d["target"].to(device, dtype=torch.float).view(-1, 1)
            enc = d["num_feat"].to(device, dtype=torch.float)
            day = d["dayOfweek"].to(device, dtype=torch.long)
            hour = d["hour"].to(device, dtype=torch.long)
            outputs = model(enc, day, hour)
            loss = loss_fn(outputs, targets)
            fin_loss += loss.item()
            counter += 1
            if verbose:
                losses.update(loss.item(), targets.size(0))
                tk0.set_postfix(loss=losses.avg)

        return fin_loss / counter


def run(
    model,
    train_dataset,
    valid_dataset,
    lr,
    EPOCHS,
    TRAIN_BATCH_SIZE,
    VALID_BATCH_SIZE,
    device,
    path,
    verbose,
):
    """
    trains a given model for a given number of epochs and paramters
    """
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, num_workers=4
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=4, shuffle=False
    )
    num_train_steps = int(len(train_data_loader)) * EPOCHS
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5
    )
    train_loss = []
    val_loss = []
    best = 50000
    patience = 0
    for epoch in range(EPOCHS):
        if verbose:
            print(f"--------- Epoch {epoch} ---------")
        elif epoch % 10 == 0:
            print(f"--------- Epoch {epoch} ---------")

        tr_loss = train_fn(
            train_data_loader,
            model,
            optimizer,
            device,
            verbose,
        )
        train_loss.append(tr_loss)
        if verbose:
            print(f" train_loss  = {tr_loss}")
        elif epoch % 10 == 0:
            print(f" train_loss  = {tr_loss}")

        val = eval_fn(valid_data_loader, model, device, verbose)
        val_loss.append(val)
        scheduler.step(val)
        if verbose:
            print(f" val_loss  = {val}")
        elif epoch % 10 == 0:
            print(f" val_loss  = {val}")

        if val < best:
            best = val
            patience = 0
            torch.save(model.state_dict(), path)
        else:
            patience += 1
        if patience > 10:
            print(f"Eraly Stopping on Epoch {epoch}")
            print(f"Best Loss =  {best}")
            break

    model.load_state_dict(torch.load(path), strict=False)
    return val_loss, train_loss


def predict(model, dataset, device=torch.device("cuda")):
    """
    computes the prediction a given model and data
    """
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=4, shuffle=False
    )
    losses = AverageMeter()
    rmse = AverageMeter()
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for bi, d in tk0:
            enc = d["num_feat"].to(device, dtype=torch.float)
            day = d["dayOfweek"].to(device, dtype=torch.long)
            hour = d["hour"].to(device, dtype=torch.long)
            outputs = model(enc, day, hour)
            if bi == 0:
                out = outputs
            else:
                out = torch.cat([out, outputs], dim=0)
    return out.cpu().detach().numpy()
