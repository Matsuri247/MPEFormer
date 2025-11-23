import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from tqdm import tqdm
from utils import *
from architecture import *
from ARAD_dataset import TrainARADDataset, TestARADDataset
from Chikusei_dataset import TrainChikuseiDataset, TestChikuseiDataset
import datetime

#torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description="Multispectral Image Demosaicing Toolbox")
parser.add_argument('--method', type=str, default='My')
parser.add_argument('--msfa_size', type=int, default=4)
parser.add_argument('--dataset', type=str, default='ARAD', help='ARAD, Chikusei')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=16, help="batch size")
parser.add_argument('--end_epoch', type=int, default=4000, help="number of iterations")
parser.add_argument('--init_lr', type=float, default=4e-4, help="initial learning rate")
#parser.add_argument('--lr_step', type=int, default=1000, help="adjust learning rate")
parser.add_argument('--outf', type=str, default='./exp/My/', help='output files')
parser.add_argument('--train_dir', type=str, default='./dataset/ARAD/train/')
parser.add_argument('--test_dir', type=str, default='./dataset/ARAD/test/')
parser.add_argument('--gpu_id', type=str, default='0')
opt = parser.parse_args()
# 获取环境变量
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# load dataset
print(f"Start training: {opt.method}")
print(f"\nLoading dataset: {opt.dataset}...")

train_data = TrainARADDataset(data_root=opt.train_dir, msfa_size=opt.msfa_size, patch_size=160, augment=True)
test_data = TestARADDataset(data_root=opt.test_dir, msfa_size=opt.msfa_size)
# select dataset, add_noise_std (None or 10 or 30 or 50)
if opt.dataset == 'ARAD':
    train_data = TrainARADDataset(data_root=opt.train_dir, msfa_size=opt.msfa_size, patch_size=160, augment=True, add_noise_std=None)
    test_data = TestARADDataset(data_root=opt.test_dir, msfa_size=opt.msfa_size, add_noise_std=10)
elif opt.dataset == 'Chikusei':
    train_data = TrainChikuseiDataset(data_root=opt.train_dir, msfa_size=opt.msfa_size, patch_size=160, augment=True, add_noise_std=None)
    test_data = TestChikuseiDataset(data_root=opt.test_dir, msfa_size=opt.msfa_size, add_noise_std=None)


per_epoch_iteration = len(train_data) / opt.batch_size
print(f'Training set samples: {len(train_data)}')
print(f'Test set samples: {len(test_data)}')
total_iteration = int(per_epoch_iteration * opt.end_epoch)
print(f'Total iteration: {total_iteration}')

# criterion
criterion_psnr = Loss_PSNR()
criterion_ssim = SSIM()
criterion_sam = SAM()
criterion_ergas = ERGAS()

# select corresponding loss function
criterion = L1_Charbonnier_loss()

# model
pretrained_model_path = opt.pretrained_model_path
model = model_generator(opt.method, opt.msfa_size, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_psnr.cuda()
    criterion_ssim.cuda()
    criterion_sam.cuda()
    criterion_ergas.cuda()
    criterion.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))

# scheduler select
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

def adjust_learning_rate(optimizer, epoch, step=1000):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.init_lr * (0.5 ** (epoch // step))
    return lr

def train(opt, train_loader, optimizer, scheduler, method, model, criterion, epoch, num_epochs):
    # adjust learning rate
    lr = optimizer.param_groups[0]['lr']

    train_bar = tqdm(train_loader)
    model.train()
    batchSize_count = 0

    for batch in train_bar:
        input_raw, input_sparse_raw, gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
        N, C, H, W = batch[0].size()
        batchSize_count += N
        input_raw = input_raw.cuda()
        input_sparse_raw = input_sparse_raw.cuda()
        gt = gt.cuda()

        label_PPI = gt.clone().mean(1).unsqueeze(1)
        pred_ppi, output = model(input_raw, input_sparse_raw)
        loss1 = 0.125 * criterion(pred_ppi, label_PPI)
        loss2 = 0.125 * criterion(output, gt)
        loss = loss1 + loss2

        optimizer.zero_grad()
        #with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_bar.set_description(desc='Epoch[%d/%d] Loss: %.4f, lr: %.9f' % (epoch, num_epochs, loss.item() / batchSize_count, lr))

    logger.info('Epoch[%d/%d] Loss: %.4f, lr: %.9f' % (epoch, num_epochs, loss.item() / batchSize_count, lr))

# This is actually for validate
def validate(opt, test_loader, method, model, epoch, num_epochs):
    test_bar = tqdm(test_loader)
    model.eval()
    metric_psnr = AverageMeter()
    metric_ssim = AverageMeter()
    metric_sam = AverageMeter()
    metric_ergas = AverageMeter()

    with torch.no_grad():
        for batch in test_bar:
            input_raw, input_sparse_raw, gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
            N, C, H, W = batch[0].size()
            input_raw = input_raw.cuda()
            input_sparse_raw = input_sparse_raw.cuda()
            gt = gt.cuda()

            _, output = model(input_raw, input_sparse_raw)

            psnr = criterion_psnr(output, gt)
            ssim = criterion_ssim(output, gt)
            sam = criterion_sam(output, gt)
            ergas = criterion_ergas(output, gt)

            # record metrics
            metric_psnr.update(psnr.data)
            metric_ssim.update(ssim.data)
            metric_sam.update(sam.data)
            metric_ergas.update(ergas.data)

            test_bar.set_description(desc='Epoch[%d/%d] psnr: %.4f, ssim: %.4f, sam: %.4f, ergas: %.4f' %
                                          (epoch, num_epochs, metric_psnr.avg, metric_ssim.avg, metric_sam.avg, metric_ergas.avg))
    # logger
    logger.info('Epoch[%d/%d] psnr: %.4f, ssim: %.4f, sam: %.4f, ergas: %.4f' %
                                          (epoch, num_epochs, metric_psnr.avg, metric_ssim.avg, metric_sam.avg, metric_ergas.avg))


def main():
    cudnn.benchmark = True
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=10,
                              pin_memory=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
    for epoch in range(1, opt.end_epoch+1):
        train(opt, train_loader, optimizer, scheduler, opt.method, model, criterion, epoch, opt.end_epoch)
        if epoch % 10 == 0:
            validate(opt, test_loader, opt.method, model, epoch, opt.end_epoch)
        if epoch % 100 == 0:
            print(f'Saving to {opt.outf}')
            #save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
            save_checkpoint(opt.outf, epoch, model, optimizer)


if __name__ == '__main__':
    main()
    print(torch.__version__)
