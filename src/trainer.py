from val import run_val
from dataloader import get_training_loader, get_val_loader
from modules import get_unet_model, save_checkpoint, save_nifti, overlay_segmentation
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, SaveImage
from monai.data import decollate_batch
from monai.losses import DiceLoss
from tqdm import tqdm
import os
import glob
from torch.amp import GradScaler, autocast

def init_trainer(args):
    s = trainer(args)
    args.epoch = 1
    checkpoint_path = glob.glob(os.path.join(args.project_dir,"checkpoints","*.pt"))
    if args.init_model != "":
        print("Model %s loaded from pretrain!"%args.init_model)
        s.load_parameters(args.init_model)

    elif len(checkpoint_path)>0:
        checkpoint_path, args.epoch = sorted([(i,int(i[i.rfind('_')+1:-3])) for i in checkpoint_path], key=lambda x:x[-1])[-1]
        s.load_parameters(checkpoint_path)
    return s

class trainer(torch.nn.Module):
    def __init__(self, args):
        super(trainer, self).__init__()
        self.model = get_unet_model().to(args.device)
        self.unet_loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scaler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
        if args.train: self.train_loader = get_training_loader(args.train_path)
        self.val_loader = get_val_loader(args.eval_path)

    def train_model(self, args):
        device = args.device
        self.train()
        scaler = GradScaler()
        for epoch in range(1, args.max_epoch + 1):
            epoch_loss = 0
            step = 0
            for batch_data in tqdm(self.train_loader):
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                self.optim.zero_grad()
                try:
                    with autocast(device_type='cuda'): 
                        outputs = self.model(inputs)
                        loss = self.unet_loss(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optim)
                    scaler.update()
                    epoch_loss += loss.detach().cpu().numpy()
                    step += 1
            
                except Exception as e:
                    filepath = batch_data["image"].meta["filename_or_obj"][0]
                    filename = os.path.basename(os.path.dirname(filepath))
                    print(f"Skipping {filename} due to error {e} ")
            avg_loss = epoch_loss / step
            print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
            save_checkpoint(self.model, os.path.join(os.path.dirname(os.getcwd()), "checkpoints", f"unet_brats_{epoch}.pt"))
            if epoch % args.inference_interval == 0:
                print(f"[Epoch {epoch}] Running inference...")
                self.eval_model(args)
                """     def eval_model(self, args):
        val_loader = get_val_loader(args.eval_path)
        device = args.device
        self.model.eval()

        post_pred = AsDiscrete(argmax=True)

        for batch in tqdm(val_loader):
            ref_path = batch["image"].meta["filename_or_obj"][0]
            case_id = os.path.basename(os.path.dirname(ref_path))
            inputs = batch["image"].to(device)

            with torch.no_grad():
                outputs = sliding_window_inference(inputs, roi_size=(-1, -1, -1), sw_batch_size=1, predictor=self.model)
                outputs = [post_pred(o) for o in decollate_batch(outputs)]

            pred = outputs[0][0]  # shape: (H, W, D)

            # Use one of the input channels as shape reference            
            out_path = 
            save_nifti(output[0][0], batch["image"].meta["filename_or_obj"][0], os.path.join(args.project_dir,"results", f"{case_id}_seg.nii.gz")) 
"""
    
    def eval_model(self, args):
        device = args.device
        self.model.eval()

        post_pred = AsDiscrete(argmax=True)
        save_image = SaveImage(
            output_dir=os.path.join(args.project_dir, "results"),
            output_postfix="_seg",
            output_ext=".nii.gz",
            separate_folder=False,
            resample=False
        )
        total_loss=0
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                t1c = batch["image"][0][1].cpu().numpy()
                outputs = sliding_window_inference(inputs, roi_size=(-1, -1, -1), sw_batch_size=1, predictor=self.model)
                loss = self.unet_loss(outputs,labels)
                total_loss += loss.item()
                outputs = [post_pred(o) for o in decollate_batch(outputs)]
                case_id = os.path.basename(os.path.dirname(batch["image"].meta["filename_or_obj"][0]))
                meta_dict = batch["image"].meta
                save_nifti(outputs[0][0], meta_dict["filename_or_obj"][0], os.path.join(args.project_dir,"results", f"{case_id}_seg.nii.gz"))
                pred_np = outputs[0].cpu().numpy()
                label_np = labels[0].cpu().numpy()
                #plt.imsave(os.path.join(args.project_dir,"output_images", f"{case_id}_pred.png"), np.squeeze(pred_slice,axis=0).astype(np.uint8), cmap='gray')
                overlay_segmentation(t1c, pred_np[0], pred_np[0].shape[1]//2,f"{case_id} - Prediction", os.path.join(args.project_dir,"output_images", f"{case_id}_pred.png"))
                #plt.imsave(os.path.join(args.project_dir,"output_images", f"{case_id}_label.png"), np.squeeze(label_slice,axis=0).astype(np.uint8), cmap='gray')               
                overlay_segmentation(t1c, label_np[0], label_np[0].shape[1]//2,f"{case_id} - gt", os.path.join(args.project_dir,"output_images", f"{case_id}_gt.png"))
            avg_loss = total_loss / len(self.val_loader)
            print(f"Validation Loss: {avg_loss:.4f}")

    def load_parameters(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint)
        print(f"Loaded model parameters from {path}")