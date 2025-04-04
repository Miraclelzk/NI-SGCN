import os
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from models_NI_HSGCN.denoise_Noisy import *
from models_NI_HSGCN.utils_Noisy import chamfer_distance_unit_sphere

from spikingjelly.activation_based import neuron, functional


# Arguments
parser = argparse.ArgumentParser()
## Dataset and loader
parser.add_argument('--dataset_root', type=str, default='../data')
parser.add_argument('--dataset', type=str, default='PUNet')
parser.add_argument('--patch_size', type=int, default=1000)
parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'])
parser.add_argument('--noise_min', type=float, default=0.005)
parser.add_argument('--noise_max', type=float, default=0.020)
parser.add_argument('--train_batch_size', type=int, default=32)
# parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
## Model architecture
parser.add_argument('--supervised', type=eval, default=True, choices=[True, False])
parser.add_argument('--frame_knn', type=int, default=32)
parser.add_argument('--num_train_points', type=int, default=128)
parser.add_argument('--num_clean_nbs', type=int, default=4, help='For supervised training.')
parser.add_argument('--num_selfsup_nbs', type=int, default=8, help='For self-supervised training.')
parser.add_argument('--dsm_sigma', type=float, default=0.01)
parser.add_argument('--score_net_hidden_dim', type=int, default=128)
parser.add_argument('--score_net_num_blocks', type=int, default=4)
## Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
## Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
# parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--exp_dir', type=str, default='', help='experiment name')
parser.add_argument('--summary_dir', type=str, help='log')
# snn
parser.add_argument('--T', type=int, default=4, metavar='T', help='Time')
parser.add_argument('--mu', type=float, default=0.0, metavar='mu', help='parameter beta')
parser.add_argument('--sigma', type=float, default=0.2, metavar='sigma', help='sigma_init')

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=1*MILLION)
parser.add_argument('--val_freq', type=int, default=2000)
parser.add_argument('--val_upsample_rate', type=int, default=4)
parser.add_argument('--val_num_visualize', type=int, default=4)
parser.add_argument('--val_noise', type=float, default=0.015)
parser.add_argument('--ld_step_size', type=float, default=0.2)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

# exp_name = 'snn-IF-T4'
# exp_dir = 'Exp/' + exp_name
# summary_dir = exp_dir + '/logs'

# args.exp_dir = exp_dir
# args.summary_dir = summary_dir

args.network_model_dir = os.path.join(args.exp_dir, "checkpoint")
args.train_batch_size = 32
# args.num_workers = 8

# Logging
if args.logging:

    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)
    if not os.path.exists(args.network_model_dir):
        os.makedirs(args.network_model_dir)
    # log_dir = get_new_log_dir(args.log_root, prefix='D%s_' % (args.dataset), postfix='_' + args.tag if args.tag is not None else '')

    logger = get_logger('train', args.exp_dir)
    writer = torch.utils.tensorboard.SummaryWriter(args.summary_dir)
    ckpt_mgr = CheckpointManager(args.network_model_dir)
    log_hyperparams(writer, args.exp_dir, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets')
train_dset = PairedPatchDataset(
    datasets=[
        PointCloudDataset(
            root=args.dataset_root,
            dataset=args.dataset,
            split='train',
            resolution=resl,
            transform=standard_train_transforms(noise_std_max=args.noise_max, noise_std_min=args.noise_min, rotate=args.aug_rotate)
        ) for resl in args.resolutions
    ],
    patch_size=args.patch_size,
    patch_ratio=1.2,
    on_the_fly=True  
)
val_dset = PointCloudDataset(
        root=args.dataset_root,
        dataset=args.dataset,
        split='test',
        resolution=args.resolutions[0],
        transform=standard_train_transforms(noise_std_max=args.val_noise, noise_std_min=args.val_noise, rotate=False, scale_d=0),
    )
train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True))

# Model
logger.info('Building model...')
model = DenoiseNet_snn_Noisy(args).to(args.device)
functional.set_step_mode(model, step_mode='m')
# functional.set_backend(model, backend='cupy', instance=NIIFNode)   
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)
    pcl_noisy = batch['pcl_noisy'].to(args.device)
    pcl_clean = batch['pcl_clean'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    if args.supervised:
        loss = model.get_supervised_loss(pcl_noisy=pcl_noisy, pcl_clean=pcl_clean)
    else:
        loss = model.get_selfsupervised_loss(pcl_noisy=pcl_noisy)

    functional.reset_net(model)
    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    # Logging
    if it % 100 == 0:
        logger.info('[Train] Iter %04d | Loss %.6f | Grad %.6f' % (
            it, loss.item(), orig_grad_norm,
        ))

    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush() 

def validate(it):
    all_clean = []
    all_denoised = []
    for i, data in enumerate(tqdm(val_dset, desc='Validate')):
        pcl_noisy = data['pcl_noisy'].to(args.device)
        pcl_clean = data['pcl_clean'].to(args.device)
        pcl_denoised = patch_based_denoise(model, pcl_noisy, ld_step_size=args.ld_step_size)
        functional.reset_net(model) # reset_net
        all_clean.append(pcl_clean.unsqueeze(0))
        all_denoised.append(pcl_denoised.unsqueeze(0))
    all_clean = torch.cat(all_clean, dim=0)
    all_denoised = torch.cat(all_denoised, dim=0)

    avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()

    logger.info('[Val] Iter %04d | CD %.6f  ' % (it, avg_chamfer))
    writer.add_scalar('val/chamfer', avg_chamfer, it)
    writer.add_mesh('val/pcl', all_denoised[:args.val_num_visualize], global_step=it)
    writer.flush()

    # scheduler.step(avg_chamfer)
    return avg_chamfer


# Main loop
logger.info('Start training...')
try:
    for it in range(1, args.max_iters+1):
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            cd_loss = validate(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
            # ckpt_mgr.save(model, args, 0, opt_states, step=it)

except KeyboardInterrupt:
    logger.info('Terminating...')
