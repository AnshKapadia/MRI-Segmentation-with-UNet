from val import run_val
from dataloader import get_training_loader
from modules import get_unet_model, save_checkpoint
import torch
from monai.losses import DiceLoss
from tqdm import tqdm
import os

def main():
    train_path = os.path.join(os.path.dirname(os.getcwd()), "data", "training_data1_v2")
    val_path = os.path.join(os.path.dirname(os.getcwd()), "validation_data")
    inference_interval = 50
    total_epochs = 150

    model, device = None, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_training_loader(train_path)

    model = get_unet_model().to(device)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(1, total_epochs + 1):
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            try: 
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step += 1

            except Exception as e:
                filepath = batch_data["image"].meta["filename_or_obj"][0]
                filename = os.path.basename(os.path.dirname(filepath))
                print(f"Skipping {filename} due to error {e} ")
        avg_loss = epoch_loss / step
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        save_checkpoint(model, os.path.join(os.path.dirname(os.getcwd()), "checkpoints", f"unet_brats_{epoch}.pt"))
        if epoch % inference_interval == 0:
            print(f"[Epoch {epoch}] Running inference...")
            run_val(model, device, val_path=val_path)

if __name__ == "__main__":
    main()

