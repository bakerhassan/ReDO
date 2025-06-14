import argparse
from PIL import Image

import torch
import torchvision

import models
import datasets
from torch.utils.data import TensorDataset
import torch.nn.functional as F

import matplotlib.cm as cm

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=40, help='number of images')
parser.add_argument('--device', default='cpu', help='cpu | cuda:n, device to be loaded on')
parser.add_argument('--statePath', default=None, help='path to pretrained weights')
parser.add_argument('--statePathM', default=None, help='path to pretrained weights for mask predictor')
parser.add_argument('--statePathX', default=None, help='path to pretrained weights for region generator')
parser.add_argument('--statePathZ', default=None, help='path to pretrained weights for noise reconstruction')
parser.add_argument('--statePathD', default=None, help='path to pretrained weights for discriminator')
parser.add_argument('--dataroot', default=None, help='path to data')

load_options = parser.parse_args()

device = torch.device(load_options.device)

if not load_options.statePath is None:
    states = torch.load(load_options.statePath, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    if "netEncM" in states:
        netEncM = models._netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(device)
        netEncM.load_state_dict(states["netEncM"])
        netEncM.eval()
    if "netGenX" in states:
        netGenX = models._netGenX(sizex=opt.sizex, nOut=opt.nx, nc=opt.nz, nf=opt.nfX, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG).to(device)
        netGenX.load_state_dict(states["netGenX"])
        netGenX.eval()
    if "netRecZ" in states:
        netRecZ = models._netRecZ(sizex=opt.sizex, nIn=opt.nx, nc=opt.nz, nf=opt.nfZ, nMasks=opt.nMasks).to(device)
        netRecZ.load_state_dict(states["netRecZ"])
        netRecZ.eval()
    if "netDX" in states:
        netDX = models._resDiscriminator128(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(device)
        netDX.load_state_dict(states["netDX"])
        netDX.eval()

if not load_options.statePathM is None:
    states = torch.load(load_options.statePathM, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    netEncM = models._netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(device)
    netEncM.load_state_dict(states["netEncM"])
    netEncM.eval()
    
if not load_options.statePathX is None:
    states = torch.load(load_options.statePathX, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    netGenX = models._netGenX(sizex=opt.sizex, nOut=opt.nx, nc=opt.nz, nf=opt.nfX, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG).to(device)
    netGenX.load_state_dict(states["netGenX"])
    netGenX.eval()
    
if not load_options.statePathZ is None:
    states = torch.load(load_options.statePathZ, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    netRecZ = models._netRecZ(sizex=opt.sizex, nIn=opt.nx, nc=opt.nz, nf=opt.nfZ, nMasks=opt.nMasks).to(device)
    netRecZ.load_state_dict(states["netRecZ"])
    netRecZ.eval()

if not load_options.statePathD is None:
    states = torch.load(load_options.statePathD, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    netDX = models._resDiscriminator128(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(device)
    netDX.load_state_dict(states['netDX'])
    netDX.eval()

if opt.dataset == "lfw":
    dataset = datasets.LFWDataset(dataPath=load_options.dataroot,
                                  sets='test',
                                  transform=torchvision.transforms.Compose([torchvision.transforms.Resize(opt.sizex, Image.NEAREST),
                                                                            torchvision.transforms.CenterCrop(opt.sizex),
                                                                            torchvision.transforms.ToTensor(),
                                  ]),)
if opt.dataset == 'cub':
    dataset = datasets.CUBDataset(load_options.dataroot,
                                  "train",
                                  torchvision.transforms.Compose([torchvision.transforms.Resize(opt.sizex, Image.NEAREST),
                                                                  torchvision.transforms.CenterCrop(opt.sizex),
                                                                  torchvision.transforms.ToTensor(),
                                  ]))
if opt.dataset == 'flowers':
    dataset = datasets.FlowersDataset(load_options.dataroot,
                                      "train",
                                      torchvision.transforms.Compose([torchvision.transforms.Resize(opt.sizex, Image.NEAREST),
                                                                      torchvision.transforms.CenterCrop(opt.sizex),
                                                                      torchvision.transforms.ToTensor(),
                                      ]))
if opt.dataset == 'cmnist':
    dataset = datasets.CMNISTDataset(dataPath=load_options.dataroot,
                                     sets='train')
if opt.dataset == 'sss':
    data = torch.load('/lustre/cniel/onr/sss_fgbg.pt')
    fg_images, masks = data['images'].repeat(1, 3, 1, 1), data['masks']
    fg_images = fg_images[data['masks'].float().mean((1,2,3)) == 0]
    masks = masks[data['masks'].float().mean((1,2,3)) == 0]
    class TransformWrapper(torch.utils.data.Dataset):
        def __init__(self, tensor_dataset, transform_fn):
            self.dataset = tensor_dataset
            self.transform = transform_fn

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            return self.transform(x, y)


    def transform_fn(img, mask):
        img = F.interpolate(img.unsqueeze(0), size=opt.sizex, mode='nearest').squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=opt.sizex, mode='nearest').squeeze(0).long()
        return img, mask


    raw_dataset = TensorDataset(fg_images, masks)
    dataset = TransformWrapper(raw_dataset, transform_fn)

loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
xData, mData = next(iter(loader))
xData = xData.to(device)
mData = mData.to(device)

## Use the same z for all images in batch: ##
# z = torch.randn(1, opt.nMasks, opt.nz, 1, 1).repeat(batch_size, 1, 1, 1, 1).to(device)

## or use different z: ##
z = torch.randn(load_options.batch_size, opt.nMasks, opt.nz, 1, 1).to(device)

with torch.no_grad():
    # Using the mask predictor:
    mPred = netEncM(xData)

    # Redrawing using soft predictred masks:   
    xGen = netGenX(mPred, z) + (xData.unsqueeze(1) * (1-mPred.unsqueeze(2)))

    # or using binarized predictred masks:
    # xGen = netGenX((mPred >= .5).float(), z) + (xData.unsqueeze(1) * (mPred < .5).float().unsqueeze(2))

    # or using ground truth masks:
    # xGen = netGenX(torch.cat((mData, 1-mData),1), z) + (xData.unsqueeze(1) * torch.cat((1-mData, mData),1).unsqueeze(2))
    def apply_colormap_bupu_r(tensor):
        """
        Apply the BuPu_r colormap to a 2D or 3D tensor (C=1 or HxW), return 3-channel RGB tensor.
        """
        if tensor.ndim == 3:  # [1, H, W] or [C, H, W]
            tensor = tensor.squeeze(0)
        np_img = tensor.cpu().numpy()
        # Normalize to [0, 1]
        np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min() + 1e-8)
        cmap = cm.get_cmap('BuPu_r')
        colored = cmap(np_img)[:, :, :3]  # RGBA → RGB
        return torch.tensor(colored,dtype=torch.float32).permute(2, 0, 1)  # [H, W, C] → [C, H, W]


    # Apply colormap to all except mData and mPred
    xData_color = torch.stack([apply_colormap_bupu_r(img[0]) for img in xData])
    xGen0_color = torch.stack([apply_colormap_bupu_r(img[0]) for img in xGen[:, 0]])
    xGen1_color = torch.stack([apply_colormap_bupu_r(img[0]) for img in xGen[:, 1]])

    # Expand mData and mPred to match RGB
    mData_rgb = mData.expand(-1, 3, -1, -1)  # [N, 1, H, W] → [N, 3, H, W]
    mPred0_rgb = mPred[:, 0:1].expand(-1, 3, -1, -1)
    print(xGen0_color.min(),xGen0_color.max())
    mPred1_rgb = (mPred[:, 1:2] >= 0.5).float().expand(-1, 3, -1, -1)
    # Concatenate all
    out = torch.cat((
        xData_color.cpu(),  # colored
        mData_rgb.cpu(),  # true masks
        mPred0_rgb.cpu(),  # grayscale/binary
        mPred1_rgb.cpu(),  # grayscale/binary thresholded
        xGen0_color.cpu(),  # colored
        xGen1_color.cpu()  # colored
    ), dim=1)

    out = out.clamp(0, 1)
    torchvision.utils.save_image(out.view(-1,3,128,128), 'out.png', normalize=False, nrow=6)
