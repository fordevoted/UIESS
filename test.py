import argparse
import os

import time
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from models import *
from datasets import *

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="Refactor testing",
# parser.add_argument("--exp_name", type=str, default="release",
                    help="name of the experiment")
parser.add_argument("--test_dir", type=str, default=r'D:\Underwater\UIEB\raw-890', help="path to test image directory")
parser.add_argument("--out_dir", type=str, default=r'./output', help="path to output image directory")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--checkpoint", type=int, default=29, help="number of epoch of checkpoint")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
parser.add_argument("--num_sample", type=int, default=2, help="number of the style code")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=40, help="number of filters in first encoder layer")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--gpu", type=str, default='1', help="set GPU")
parser.add_argument("--print_model_complexity", type=bool, default=True,
                    help="Print number of params and run time speed")
parser.add_argument("--data_root", type=str, default=r'D:\scene_other\UNIT\DAUW', help="If you want to use "
                                                                                       "'test_plot_latent_tsne' or "
                                                                                       "'test_samples', "
                                                                                       "prepare dataset your dataset "
                                                                                       "follow dataset format in "
                                                                                       "README")
parser.add_argument("--seed", type=int, default=123, help="Random state")
opt = parser.parse_args()
# print(opt)
cuda = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


def test_REAL_image(epoch):
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_REAL_image', str(epoch))
    os.makedirs(out_path, exist_ok=True)

    transforms_val = [
        transforms.ToTensor(),
    ]
    enhancedValDataset = EnhancedValDataset(transforms_=transforms_val, dataset_path=opt.test_dir)
    enhancedValDataset.files += [os.path.join(r'D:\Underwater\UIEB\challenging-60', x) for x in
                                  os.listdir(r'D:\Underwater\UIEB\challenging-60') if is_image_file(x)]
    val_dataloader = DataLoader(enhancedValDataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True,
                                )
    if opt.print_model_complexity:
        num_params = 0
        models = [c_Enc, real_sty_Enc, G, T]
        for model in models:
            for param in model.parameters():
                num_params += param.numel()
            # print(net)
        print('Total number of parameters: %d' % num_params)

    time_all = 0
    for i, batch in enumerate(val_dataloader):
        imgReal = batch["img"]
        name = batch["name"][0].split(os.sep)[-1]
        with torch.no_grad():
            # Create copies of image
            XReal = imgReal
            XReal = Variable(XReal.type(Tensor)).cuda()

            # Generate samples
            start = time.time()

            c_code_Real, s_code_Real = c_Enc(XReal), real_sty_Enc(XReal)
            en_s_code_Real = T(s_code_Real)
            enhanced_Real = G(c_code_Real, en_s_code_Real)
            if opt.print_model_complexity and i != 0:
                time_all += time.time() - start
                # print("time: ", time.time() - start)

            ndarr = enhanced_Real.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                  torch.uint8).numpy()
            im = Image.fromarray(ndarr)

            ori_im = Image.open(batch["name"][0])
            im = im.resize(ori_im.size)

            im.save(os.path.join(out_path, name))

    print("Total time: %f, average time: %f, FPS: %f, dataloader: %d" % (
        time_all, time_all / (len(val_dataloader) - 1), (len(val_dataloader) - 1) / time_all, len(val_dataloader)))


def test_SYN_image():
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_SYN_image')
    os.makedirs(out_path, exist_ok=True)

    transforms_val = [
        transforms.ToTensor(),
    ]
    enhancedValDataset = EnhancedValDataset(transforms_=transforms_val, dataset_path=opt.test_dir)
    val_dataloader = DataLoader(enhancedValDataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True,
                                )
    if opt.print_model_complexity:
        num_params = 0
        models = [c_Enc, syn_sty_Enc, G, T]
        for model in models:
            for param in model.parameters():
                num_params += param.numel()
            # print(net)
        print('Total number of parameters: %d' % num_params)

    time_all = 0
    for i, batch in enumerate(val_dataloader):
        imgSyn = batch["img"]
        name = batch["name"][0].split(os.sep)[-1]
        with torch.no_grad():
            # Create copies of image
            XSyn = imgSyn
            XSyn = Variable(XSyn.type(Tensor)).cuda()

            # Generate samples
            start = time.time()

            c_code_Syn, s_code_Syn = c_Enc(XSyn), syn_sty_Enc(XSyn)
            en_s_code_Syn = T(s_code_Syn)
            enhanced_Syn = G(c_code_Syn, en_s_code_Syn)
            if opt.print_model_complexity and i != 0:
                time_all += time.time() - start
                print("time: ", time.time() - start)

            ndarr = enhanced_Syn.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                 torch.uint8).numpy()
            im = Image.fromarray(ndarr)

            ori_im = Image.open(batch["name"][0])
            im = im.resize(ori_im.size)

            im.save(os.path.join(out_path, name))

    print("Total time: %f, average time: %f, FPS: %f, dataloader: %d" % (
        time_all, time_all / (len(val_dataloader) - 1), (len(val_dataloader) - 1) / time_all, len(val_dataloader)))


def test_plot_latent_tsne():
    transforms_val = [
        transforms.ToTensor(),
    ]
    val_dataloader = DataLoader(
        EnhancedDataset(opt.data_root,
                        transforms_=transforms_val, mode="val"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    feature = []
    label = []
    for i, batch in enumerate(val_dataloader):
        imgReal = batch["Real"]
        imgSyn = batch["Syn"]

        # Create copies of image
        with torch.no_grad():
            # Create copies of image
            XReal = imgReal
            XReal = Variable(XReal.type(Tensor)).cuda()
            XSyn = imgSyn
            XSyn = Variable(XSyn.type(Tensor)).cuda()

            # Generate samples
            s_code_Real = real_sty_Enc(XReal)
            en_s_code_Real = T(s_code_Real)

            # Generate samples
            s_code_Syn = syn_sty_Enc(XSyn)
            en_s_code_Syn = T(s_code_Syn)
            feature.append(s_code_Real.cpu().detach().numpy().squeeze())
            label.append('syn')

            feature.append(s_code_Syn.cpu().detach().numpy().squeeze())
            label.append('real-world')

            feature.append(en_s_code_Real.cpu().detach().numpy().squeeze())
            label.append('clean')
            # label.append('real-world → clean')

            feature.append(en_s_code_Syn.cpu().detach().numpy().squeeze())
            label.append('clean')
            # label.append('syn → clean')

    feature = np.array(feature)
    label = np.array(label)
    X = TSNE(perplexity=18.0, learning_rate='auto', random_state=123, verbose=1).fit_transform(feature)
    sns.scatterplot(X[:, 0], X[:, 1], hue=label, legend='full', palette='Set2')
    plt.show()


def test_latent_manipulation():
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_latent_manipulation')
    os.makedirs(out_path, exist_ok=True)

    # You can manually adjust alphas value to obtain satisfactory result
    # alphas = np.linspace(-1, 2, num=10)
    alphas = np.array([0, 0.25, 0.5, 0.7, 0.8, 1.0, 1.33, 1.5, 1.7])
    # alphas = np.array([0, 0.25, 0.6, 0.8, 0.9, 1.0, 1.2, 1.33, 1.35, 1.40])
    # alphas = np.array([0, 0.25, 0.45, 0.65, 0.75, 1.0, 1.3, 1.35, 1.45])

    transforms_val = [
        transforms.ToTensor(),
    ]
    transforms_val = transforms.Compose(transforms_val)

    img_list = os.listdir(opt.test_dir)
    for img_name in img_list:
        img_path = os.path.join(opt.test_dir, img_name)
        img = load_img(img_path)
        w, h = img.size
        new_w, new_h = w, h
        if (w / 4) % 1 != 0:
            new_w = w // 4 * 4
        if (h / 4) % 1 != 0:
            new_h = h // 4 * 4
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h))

        img = transforms_val(img).cuda().unsqueeze(0)
        item_list = []
        img_enhanceds = None
        with torch.no_grad():
            # Create copies of image
            X = img
            X = Variable(X.type(Tensor)).cuda()

            # Generate samples
            c_code, s_code_ori = c_Enc(X), real_sty_Enc(X)
            en_s_code = T(s_code_ori)

            for alpha in alphas:
                s_code = s_code_ori + alpha * (en_s_code - s_code_ori)
                enhanced = G(c_code, s_code)

                item_list.append(enhanced)

            img = None
            for item in item_list:
                img = item if img is None else torch.cat((img, item), -1)

            img_enhanceds = img if img_enhanceds is None else torch.cat((img_enhanceds, img), -2)
            save_image(img_enhanceds, os.path.join(out_path, img_name), nrow=1, normalize=True, range=(0, 1))
            print(img_name)


def test_samples():
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_samples')
    os.makedirs(out_path, exist_ok=True)

    transforms_val = [
        transforms.ToTensor(),
    ]

    val_dataloader = DataLoader(
        EnhancedDataset(opt.data_root,
                        transforms_=transforms_val, mode="val"),
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    def sample_images(batches_done):
        """Saves a generated sample from the validation set"""
        for i, batch in enumerate(val_dataloader):
            # img_samples = None
            img_enhanceds_Real = None
            img_enhanceds_Syn = None
            for imgReal, imgSyn, label_Syn in zip(batch["Real"], batch["Syn"], batch["label"]):
                with torch.no_grad():
                    # Create copies of image
                    XReal = imgReal.unsqueeze(0)
                    XReal = Variable(XReal.type(Tensor)).cuda()

                    XSyn = imgSyn.unsqueeze(0)
                    XSyn = Variable(XSyn.type(Tensor)).cuda()

                    # Generate samples
                    c_code_Real, s_code_Real = c_Enc(XReal), real_sty_Enc(XReal)
                    c_code_Syn, s_code_Syn = c_Enc(XSyn), syn_sty_Enc(XSyn)
                    XRealSyn = G(c_code_Real, s_code_Syn)
                    XSynReal = G(c_code_Syn, s_code_Real)

                    en_s_code_Real = T(s_code_Real)
                    en_s_code_Syn = T(s_code_Syn)

                    enhanced_Real = G(c_code_Real, en_s_code_Real)
                    enhanced_Syn = G(c_code_Syn, en_s_code_Syn)

                    # cycle consistent
                    c_code_SynReal, s_code_SynReal = c_Enc(XSynReal), real_sty_Enc(XSynReal)
                    c_code_RealSyn, s_code_RealSyn = c_Enc(XRealSyn), syn_sty_Enc(XRealSyn)
                    XRealSynReal = G(c_code_RealSyn, s_code_Real)
                    XSynRealSyn = G(c_code_SynReal, s_code_Syn)

                    # Reconstruct images
                    XRealReal = G(c_code_Real, s_code_Real)
                    XSynSyn = G(c_code_Syn, s_code_Syn)

                    # Concatenate samples horisontally
                    item_list = [XSynSyn, XSynReal, XSynRealSyn, enhanced_Syn, label_Syn.cuda().unsqueeze(0)]
                    imgSyn = imgSyn.cuda().unsqueeze(0)
                    for item in item_list:
                        imgSyn = torch.cat((imgSyn, item), -1)

                    item_list = [XRealReal, XRealSyn, XRealSynReal, enhanced_Real]
                    imgReal = imgReal.cuda().unsqueeze(0)
                    for item in item_list:
                        imgReal = torch.cat((imgReal, item), -1)

                    # Concatenate with previous samples vertically
                    img_enhanceds_Real = imgReal if img_enhanceds_Real is None else torch.cat(
                        (img_enhanceds_Real, imgReal), -2)
                    img_enhanceds_Syn = imgSyn if img_enhanceds_Syn is None else torch.cat(
                        (img_enhanceds_Syn, imgSyn), -2)

            save_image(img_enhanceds_Real, os.path.join(out_path, "%s_I2I_Enhanced_Real.png") % str(i), nrow=1,
                       normalize=True, range=(0, 1))
            save_image(img_enhanceds_Syn, os.path.join(out_path, "%s_I2I_Enhanced_Syn.png") % str(i), nrow=1,
                       normalize=True, range=(0, 1))
            if i > batches_done:
                return

    sample_images(batches_done=300)


if __name__ == '__main__':

    testing = 'test_REAL_image'
    testing_list = ['test_REAL_image', 'test_SYN_image', 'test_plot_latent_tsne', 'test_latent_manipulation',
                     'test_samples']
    # Initialize encoders, generators and discriminators

    # torch.cuda.empty_cache()
    # import gc
    # gc.collect()

    c_Enc = ContentEncoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual)
    G = Generator(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
    real_sty_Enc = StyleEncoder(dim=opt.dim, n_downsample=opt.n_downsample, style_dim=opt.style_dim)
    syn_sty_Enc = StyleEncoder(dim=opt.dim, n_downsample=opt.n_downsample, style_dim=opt.style_dim)
    T = StyleTransformUnit(dim=opt.dim, style_dim=opt.style_dim)

    if cuda:
        c_Enc = c_Enc.cuda()
        G = G.cuda()
        real_sty_Enc = real_sty_Enc.cuda()
        syn_sty_Enc = syn_sty_Enc.cuda()
        T = T.cuda()
    # for epoch in range(25, 36):
        # Load pretrained models
        c_Enc.load_state_dict(torch.load("saved_models/%s/c_Enc_29.pth" % opt.exp_name))
        G.load_state_dict(torch.load("saved_models/%s/G_29.pth" % opt.exp_name))
        real_sty_Enc.load_state_dict(torch.load("saved_models/%s/real_sty_Enc_29.pth" % opt.exp_name))
        syn_sty_Enc.load_state_dict(torch.load("saved_models/%s/syn_sty_Enc_29.pth" % opt.exp_name))
        T.load_state_dict(torch.load("saved_models/%s/T_29.pth" % opt.exp_name))

        # c_Enc.load_state_dict(torch.load("saved_models/%s/c_Enc_%d.pth" % (opt.exp_name, epoch)))
        # G.load_state_dict(torch.load("saved_models/%s/G_%d.pth" % (opt.exp_name, epoch)))
        # real_sty_Enc.load_state_dict(torch.load("saved_models/%s/real_sty_Enc_%d.pth" % (opt.exp_name, epoch)))
        # syn_sty_Enc.load_state_dict(torch.load("saved_models/%s/syn_sty_Enc_%d.pth" % (opt.exp_name, epoch)))
        # T.load_state_dict(torch.load("saved_models/%s/T_%d.pth" % (opt.exp_name, epoch)))

        if testing == 'test_REAL_image':
            test_REAL_image(epoch=29)
        elif testing == 'test_SYN_image':
            test_SYN_image()
        elif testing == 'test_plot_latent_tsne':
            test_plot_latent_tsne()
        elif testing == 'test_latent_manipulation':
            test_latent_manipulation()
        elif testing == 'test_samples':
            test_samples()
