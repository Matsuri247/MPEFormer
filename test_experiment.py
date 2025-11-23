import torch
import numpy as np
import os
import glob
import argparse
import torch.backends.cudnn as cudnn
from architecture import *
from utils import *
from torch.utils.data import DataLoader
from ARAD_dataset import TestARADDataset
from Chikusei_dataset import TestChikuseiDataset

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Multispectral Image Demosaicing Toolbox")
parser.add_argument('--method', type=str, default='My')
parser.add_argument('--msfa_size', type=int, default=4)
parser.add_argument('--dataset', type=str, default='CAVE', help='CAVE,ICVL,ARAD')
parser.add_argument('--pretrained_model_path', type=str, default='./model_zoo/MCAN_CAVE.pth')
parser.add_argument('--test_dir', type=str, default='./dataset/CAVE/test/')
parser.add_argument('--outf', type=str, default='./test_exp/My/', help='output files')
parser.add_argument('--gpu_id', type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
print(f"Start testing: {opt.method}")
print(f"\nLoading dataset: {opt.dataset}...")

test_data = TestARADDataset(data_root=opt.test_dir, msfa_size=opt.msfa_size, add_noise_std=None)
# select dataset, add_noise_std (None or 10 or 30 or 50)
if opt.dataset == 'ARAD':
    test_data = TestARADDataset(data_root=opt.test_dir, msfa_size=opt.msfa_size, add_noise_std=None)
elif opt.dataset == 'Chikusei':
    test_data = TestChikuseiDataset(data_root=opt.test_dir, msfa_size=opt.msfa_size, add_noise_std=None)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

# gt list (get save file_name)
file_type = '*.mat' # '*.mat' or '*.tif'
file_list = glob.glob(os.path.join(opt.test_dir, file_type))
file_list.sort()
filename_list = [os.path.split(file)[1] for file in file_list]

# criterion
criterion_psnr = Loss_PSNR()
criterion_ssim = SSIM()
criterion_sam = SAM()
criterion_ergas = ERGAS()

if torch.cuda.is_available():
    criterion_psnr.cuda()
    criterion_ssim.cuda()
    criterion_sam.cuda()
    criterion_ergas.cuda()

def test(opt, test_loader, method, model):
    model.eval()
    metric_psnr = AverageMeter()
    metric_ssim = AverageMeter()
    metric_sam = AverageMeter()
    metric_ergas = AverageMeter()

    with torch.no_grad():
        for i, (raw, sparse_raw, msi) in enumerate(test_loader):
            input_raw, input_sparse_raw, gt = raw, sparse_raw, msi
            _, _, H, W = raw.size()
            input_raw = input_raw.cuda()
            input_sparse_raw = input_sparse_raw.cuda()
            gt = gt.cuda()

            if method == 'MCAN_splitraw' or method == 'MCAN_conv':
                scale_coord_map = input_matrix_wpn(H, W, opt.msfa_size)
                scale_coord_map = scale_coord_map.cuda()
                output = model([input_sparse_raw, input_raw], scale_coord_map)

            elif method == 'MSFN' or method == 'MSFN_onlySpectral' or method == 'MSFN_onlySpatial' or method == 'MSFN_mambair' or method == 'MSFN_mambair_noFM':
                output = model(input_sparse_raw)

            #elif method == 'MSFN_onlySpatial_3branches' or method == 'MSFN_winattn_convpos_3branches':
            elif method == 'MPEFormer' or method == 'MPEFormer_new1' or method == 'MPEFormer_new2' or method == 'MPEFormer_new_complete' or method == 'Model1_W_SW_MSA' or method == 'Model2_W_CW_MSA' or method == 'Model3_SW_CW_MSA' or method == 'Model4_SW_CW_PSA':
                _, output = model(input_raw, input_sparse_raw)

            # calculate metrics
            psnr = criterion_psnr(output, gt)
            ssim = criterion_ssim(output, gt)
            sam = criterion_sam(output, gt)
            ergas = criterion_ergas(output, gt)
            # record metrics
            metric_psnr.update(psnr.data)
            metric_ssim.update(ssim.data)
            metric_sam.update(sam.data)
            metric_ergas.update(ergas.data)
            # save to .mat
            result = output.cpu().numpy() * 1.0 # ndarray, [b,c,h,w]
            result = np.transpose(np.squeeze(result), [1, 2, 0]) # ndarray,[b,c,h,w] -> [h,w,c]
            result = np.minimum(result, 1.0)
            result = np.maximum(result, 0)
            result_name = filename_list[i]
            result_dir = os.path.join(opt.outf, result_name)
            save_matv73(result_dir, 'cube', result)
    return metric_psnr.avg, metric_ssim.avg, metric_sam.avg, metric_ergas.avg

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    msfa_size = opt.msfa_size
    model = model_generator(method, msfa_size, pretrained_model_path).cuda()
    psnr, ssim, sam, ergas = test(opt, test_loader, method, model)
    print(f'method:{method}, PSNR:{psnr}, SSIM:{ssim}, SAM:{sam}, ERGAS:{ergas}.')