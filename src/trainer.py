from val import run_val
from dataloader import get_training_loader, get_val_loader
from modules import get_unet_model, save_checkpoint, save_nifti
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.losses import DiceLoss
from tqdm import tqdm
import os
import glob
from torch.amp import GradScaler, autocast

def init_trainer(args):
    s = trainer(args)
    args.epoch = 1
    checkpoint_path = sorted(glob.glob(os.path.join(r"C:\Users\anshk\MRI_segmentation\checkpoints","*.pt")))
    if args.init_model != "":
        print("Model %s loaded from pretrain!"%args.init_model)
        s.load_parameters(args.init_model)

    elif len(checkpoint_path)>0:
        s.load_parameters(checkpoint_path[-1])
    return s

class trainer(torch.nn.Module):
    def __init__(self, args):
        super(trainer, self).__init__()
        self.model = get_unet_model().to(args.device)
        self.unet_loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scaler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
        self.train_loader = get_training_loader(args.train_path)

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

    def eval_model(self, args):
        val_loader = get_val_loader(args.eval_path)
        device = args.device
        self.model.eval()

        post_pred = AsDiscrete(argmax=True)
        os.makedirs("inference_outputs", exist_ok=True)

        for batch in tqdm(val_loader):
            ref_path = batch["image"].meta["filename_or_obj"][0]
            case_id = os.path.basename(os.path.dirname(ref_path))
            inputs = batch["image"].to(device)

            with torch.no_grad():
                outputs = sliding_window_inference(inputs, roi_size=(-1, -1, -1), sw_batch_size=1, predictor=self.model)
                outputs = [post_pred(o) for o in decollate_batch(outputs)]

            pred = outputs[0][0]  # shape: (H, W, D)

            # Use one of the input channels as shape reference            
            out_path = os.path.join(r"C:\Users\anshk\MRI_segmentation\results", f"{case_id}_seg.nii.gz")
            save_nifti(pred, ref_path, out_path)

    def load_parameters(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)
        print(f"Loaded model parameters from {path}")