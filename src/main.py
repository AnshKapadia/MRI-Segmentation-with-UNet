import argparse, glob, os, warnings, time
from torch import device as torch_device
from torch import cuda
from trainer import *
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "UNet for MRI segmentation")

	### Training setting
	parser.add_argument('--device', type=str, default=torch_device("cuda" if cuda.is_available() else "cpu"), help='Set device for cuda')
	parser.add_argument('--max_epoch',  type=int,   default=150,      help='Maximum number of epochs')
	parser.add_argument('--inference_interval', type=int,   default=50,      help='Frequency of validation')
	parser.add_argument('--lr',         type=float, default=0.0001,    help='Learning rate')
	parser.add_argument("--test_step",   type=int, default=1,     help='Test step')
	parser.add_argument("--lr_decay",   type=float, default=0.90,     help='Learning rate decay every [test_step] epochs')
	
    ### Testing setting
	parser.add_argument('--init_model',  type=str,   default="",  help='Init model from checkpoint')

	### Data path
	parser.add_argument('--project_dir', type=str, default=os.path.dirname(os.getcwd()))
	parser.add_argument('--train_path', type=str,   default=os.path.join(os.path.dirname(os.getcwd()), "data", "training_data1_v2"), help='The path of the training data')
	parser.add_argument('--eval_path',  type=str,   default=os.path.join(os.path.dirname(os.getcwd()), "data", "validation_data"), help='The path of the evaluation data')
	parser.add_argument('--save_path',  type=str,    default=os.path.join(os.path.dirname(os.getcwd()), "checkpoints"), help='The path where model checkpoints to be saved')

	### Others
	parser.add_argument('--train', type=bool, default=False, help='Do training')
	parser.add_argument('--eval', type=bool, default=False, help='Do evaluation')

	## Init folders, trainer and loader
	args = parser.parse_args()
	s = init_trainer(args)

	## Evaluate only
	if args.eval == True:
		s.eval_model(args)
		quit()

	## Training
	if args.train == True:
		s.train_model(args)
		quit()