import numpy as np
import os
import torch
import yaml
from dataset.dataset import ShapeNetDataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from models.dgcnn import DGCNN
from loss import XSampleContrastiveLoss
import wandb

def train_one_epoch(epoch,
                    model, 
                    dataloader, 
                    loss_fn, 
                    optimizer, 
                    device, 
                    log_interval=100):
    model.train()
    total_epoch_loss = 0
    total_data_count = 0
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    for idx, data in enumerate(loop):
        pointcloud = data["pointcloud"].to(device).permute(0, 2, 1)
        embeddings = data["embeddings"].to(device)
        optimizer.zero_grad()
        out = model(pointcloud)
        loss = loss_fn(out, embeddings)
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()*pointcloud.shape[0]
        total_data_count += pointcloud.shape[0]
        if idx % log_interval == 0:
            loop.set_description(f"Train Loss: {total_epoch_loss/total_data_count:.6f}")
            wandb.log({"batch_train_loss": total_epoch_loss/total_data_count, "step": epoch*len(dataloader)+idx})
    return total_epoch_loss/total_data_count

def val_one_epoch(model,
                  dataloader,
                  loss_fn,
                  device,
                  log_interval=100):
    model.eval()
    total_epoch_loss = 0
    total_data_count = 0
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    for idx, data in enumerate(loop):
        pointcloud = data["pointcloud"].to(device).permute(0, 2, 1)
        embeddings = data["embeddings"].to(device)
        with torch.no_grad():
            out = model(pointcloud)
            loss = loss_fn(out, embeddings)
        total_epoch_loss += loss.item()*pointcloud.shape[0]
        total_data_count += pointcloud.shape[0]
        if idx % log_interval == 0:
            loop.set_description(f"Val Loss: {total_epoch_loss/total_data_count:.6f}")
    return total_epoch_loss/total_data_count
        

def main():
    config_filename = "./config/shapenet.yaml"
    with open(config_filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_name = config["project_name"]
    wandb.init(project=project_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    filenames_list_filename = config["data"]['filenames_file']
    with open(filenames_list_filename, 'r') as file:
        filenames = json.load(file)

    train_dataset = ShapeNetDataset(config["data"], filenames["train"])
    val_dataset = ShapeNetDataset(config["data"], filenames["test"])
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config["train"]["batch_size"], 
                                  shuffle=True, 
                                  num_workers=config["train"]["num_workers"])
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config["train"]["batch_size"],
                                shuffle=False,
                                num_workers=config["train"]["num_workers"]) 

    print("Train dataset length: ", len(train_dataset))
    print("Val dataset length: ", len(val_dataset))      

    # make test model
    model = DGCNN(config["model"])
    model.to(device)

    loss_fn = XSampleContrastiveLoss(config["loss"]["labels_temperature"], 
                                     config["loss"]["preds_temperature"],
                                     device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    if not os.path.exists(config["train"]["save_folder"]):
        os.makedirs(config["train"]["save_folder"])
    start_epoch = 0
    min_val_loss = float("inf")
    if config["train"]["resume"]:
        checkpoint = torch.load(config["train"]["resume"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        min_val_loss = checkpoint["val_loss"]
        print(f"Model loaded from {config['train']['resume']} at epoch {start_epoch}")

    for epoch in range(start_epoch, config["train"]["num_epochs"]+start_epoch):
        train_loss = train_one_epoch(epoch, model, train_dataloader, loss_fn, optimizer, device, config["train"]["log_interval"])
        val_loss = val_one_epoch(model, val_dataloader, loss_fn, device, config["train"]["log_interval"])
        wandb.log({"epoch_train_loss": train_loss, "epoch_val_loss": val_loss})
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_dict = {}
            best_dict["epoch"] = epoch
            best_dict["train_loss"] = train_loss
            best_dict["val_loss"] = min_val_loss
            best_dict["model"] = model.state_dict()
            best_dict["optimizer"] = optimizer.state_dict()
            with open(os.path.join(config["train"]["save_folder"], f"{project_name}_best.pth"), 'wb') as file:
                torch.save(best_dict, file)
            print(f"Best Model saved at {config['train']['save_folder']} with loss: {val_loss}")
    
        # save last model
        latest_dict = {}
        latest_dict["epoch"] = epoch
        latest_dict["train_loss"] = train_loss
        latest_dict["val_loss"] = min_val_loss
        latest_dict["model"] = model.state_dict()
        latest_dict["optimizer"] = optimizer.state_dict()
        with open(os.path.join(config["train"]["save_folder"], f"{project_name}_latest.pth"), 'wb') as file:
            torch.save(latest_dict, file) 



if __name__ == "__main__":
    main()