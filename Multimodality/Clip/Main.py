import os
import torch
import itertools

from Model import CLIPModel
from Utils import read_settings
from Dataset import make_train_valid_dfs, build_loaders
from Train import train_epoch, valid_epoch
from transformers import DistilBertTokenizer

config = read_settings()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(config["text_tokenizer"])
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(device)
    params = [
            {"params": model.image_encoder.parameters(), "lr": config["image_encoder_lr"]},
            {"params": model.text_encoder.parameters(), "lr": config["text_encoder_lr"]},
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": config["head_lr"], "weight_decay": config["weight_decay"]}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=config["patience"], factor=config["factor"]
        )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(config["epochs"]):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
            
        lr_scheduler.step(valid_loss.avg)

if __name__ == "__main__":
    if (os.path.exists("./best.pt")) == False:
        main()