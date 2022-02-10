import argparse
import datetime
import itertools
import time

import torch
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import *
from loss import SSIM, VGGPerceptualLoss, TVLoss, DCPLoss
from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default=r'D:\scene_other\UNIT\DAUW', help="dataset")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=35, help="number of epochs of training")
parser.add_argument("--exp_name", type=str, default="Refactor testing", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=20, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=40, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
parser.add_argument("--gpu", type=str, default='1', help="set GPU")
parser.add_argument("--seed", type=int, default=123, help="Random state")
opt = parser.parse_args()
# print(opt)
cuda = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


def set_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train():
    # Create sample and checkpoint directories
    os.makedirs("./images/%s" % opt.exp_name, exist_ok=True)
    os.makedirs("./saved_models/%s" % opt.exp_name, exist_ok=True)
    # torch.cuda.empty_cache()
    set_seed(opt.seed, cuda)

    criterion_recon = torch.nn.L1Loss()
    ssim_loss = SSIM()
    tv_loss = TVLoss(1)
    perceptual_loss = VGGPerceptualLoss()
    # dcp_loss = DCPLoss(1)

    if cuda:
        criterion_recon = criterion_recon.cuda()
        ssim_loss = ssim_loss.cuda()
        tv_loss = tv_loss.cuda()
        perceptual_loss = perceptual_loss.cuda()
        # dcp_loss = dcp_loss.cuda()


    # Initialize encoders, generators and discriminators
    c_Enc = ContentEncoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual)
    G = Generator(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
    real_sty_Enc = StyleEncoder(dim=opt.dim, n_downsample=opt.n_downsample, style_dim=opt.style_dim)
    syn_sty_Enc = StyleEncoder(dim=opt.dim, n_downsample=opt.n_downsample, style_dim=opt.style_dim)
    T = StyleTransformUnit(dim=opt.dim, style_dim=opt.style_dim)
    D = MultiDiscriminator()

    if cuda:
        c_Enc = c_Enc.cuda()
        G = G.cuda()
        real_sty_Enc = real_sty_Enc.cuda()
        syn_sty_Enc = syn_sty_Enc.cuda()
        T = T.cuda()
        D = D.cuda()
        criterion_recon.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        c_Enc.load_state_dict(torch.load("saved_models/%s/c_Enc_%d.pth" % (opt.exp_name, opt.epoch)))
        G.load_state_dict(torch.load("saved_models/%s/G_%d.pth" % (opt.exp_name, opt.epoch)))
        real_sty_Enc.load_state_dict(torch.load("saved_models/%s/real_sty_Enc_%d.pth" % (opt.exp_name, opt.epoch)))
        syn_sty_Enc.load_state_dict(torch.load("saved_models/%s/syn_sty_Enc_%d.pth" % (opt.exp_name, opt.epoch)))
        T.load_state_dict(torch.load("saved_models/%s/T_%d.pth" % (opt.exp_name, opt.epoch)))
        D.load_state_dict(torch.load("saved_models/%s/D_%d.pth" % (opt.exp_name, opt.epoch)))

    else:
        # Initialize weights
        c_Enc.apply(weights_init_normal)
        G.apply(weights_init_normal)
        real_sty_Enc.apply(weights_init_normal)
        syn_sty_Enc.apply(weights_init_normal)
        T.apply(weights_init_normal)
        D.apply(weights_init_normal)

    # Loss weights
    lambda_gan = 1
    lambda_id = 10
    lambda_cyc = 1
    lambda_enhanced = 3.5 / 2  # 2.5
    lambda_ssim = 5.0 / 2  # 2.5
    lambda_tv = 0.3
    lambda_perceptual = 0.0005 / 2  # 0.0005
    lambda_enhanced_latent = 3
    # lambda_dcp = 1.5 / 2

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(c_Enc.parameters(), G.parameters(), real_sty_Enc.parameters(), syn_sty_Enc.parameters(),
                        T.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D1 = torch.optim.Adam(D.parameters(), lr=opt.lr * 5, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    # Configure dataloaders
    transforms_train = [
        transforms.ToTensor(),
    ]
    transforms_val = [
        transforms.Resize(size=(opt.img_height * 2, opt.img_width * 2), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ]

    set_seed(opt.seed, cuda)
    dataloader = DataLoader(
        EnhancedDataset(opt.data_root,
                        transforms_=transforms_train),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=True)
    set_seed(opt.seed, cuda)
    val_dataloader = DataLoader(
        EnhancedDataset(opt.data_root,
                        transforms_=transforms_val, mode="val"),
        batch_size=5,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=True,
    )

    def sample_images(batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        img_enhanceds_A = None
        img_enhanceds_B = None
        for imgA, imgB, label_B in zip(imgs["Real"], imgs["Syn"], imgs["label"]):
            with torch.no_grad():
                # Create copies of image
                XA = imgA.unsqueeze(0)
                XA = Variable(XA.type(Tensor)).cuda()
                XB = imgB.unsqueeze(0)
                XB = Variable(XB.type(Tensor)).cuda()

                c_code_A, s_code_A = c_Enc(XA), real_sty_Enc(XA)
                c_code_B, s_code_B = c_Enc(XB), syn_sty_Enc(XB)
                XAB = G(c_code_A, s_code_B)
                XBA = G(c_code_B, s_code_A)

                en_s_code_A = T(s_code_A)
                en_s_code_B = T(s_code_B)

                enhanced_A = G(c_code_A, en_s_code_A)
                enhanced_B = G(c_code_B, en_s_code_B)

                # cycle consistent
                c_code_BA, s_code_BA = c_Enc(XBA), real_sty_Enc(XBA)
                c_code_AB, s_code_AB = c_Enc(XAB), syn_sty_Enc(XAB)
                XABA = G(c_code_AB, s_code_A) if lambda_cyc > 0 else 0
                XBAB = G(c_code_BA, s_code_B) if lambda_cyc > 0 else 0

                # Reconstruct images
                XAA = G(c_code_A, s_code_A)
                XBB = G(c_code_B, s_code_B)

                # Concatenate samples horisontally
                item_list = [XBB, XBA, XBAB, enhanced_B, label_B.cuda().unsqueeze(0)]
                imgB = imgB.cuda().unsqueeze(0)
                for item in item_list:
                    imgB = torch.cat((imgB, item), -1)

                item_list = [XAA, XAB, XABA, enhanced_A]
                imgA = imgA.cuda().unsqueeze(0)
                for item in item_list:
                    imgA = torch.cat((imgA, item), -1)

                # Concatenate with previous samples vertically
                img_enhanceds_A = imgA if img_enhanceds_A is None else torch.cat((img_enhanceds_A, imgA), -2)
                img_enhanceds_B = imgB if img_enhanceds_B is None else torch.cat(
                    (img_enhanceds_B, imgB), -2)

        save_image(img_enhanceds_A, "images/%s/%s_I2I_Enhanced_A.png" % (opt.exp_name, batches_done), nrow=5,
                   normalize=True, range=(0, 1))
        save_image(img_enhanceds_B, "images/%s/%s_I2I_Enhanced_B.png" % (opt.exp_name, batches_done), nrow=5,
                   normalize=True, range=(0, 1))

    # ----------
    #  Training
    # ----------
    # Adversarial ground truths
    valid = 1
    fake = 0

    prev_time = time.time()
    sample_images(0)
    for epoch in range(opt.epoch + 1 if opt.epoch > 0 else 0, opt.n_epochs + 1):
        for i, batch in enumerate(dataloader):
            optimizer_G.zero_grad()
            optimizer_D1.zero_grad()

            # Set model input
            XA = Variable(batch["Real"].type(Tensor))
            XB = Variable(batch["Syn"].type(Tensor))
            labelB = Variable(batch["label"].type(Tensor))
            # -------------------------------
            #  Train Encoders and Generators
            # -------------------------------

            # Get shared latent representation
            c_code_A, s_code_A = c_Enc(XA), real_sty_Enc(XA)
            c_code_B, s_code_B = c_Enc(XB), syn_sty_Enc(XB)

            # Reconstruct images
            XAA = G(c_code_A, s_code_A)
            XBB = G(c_code_B, s_code_B)

            # Translate images
            XBA = G(c_code_B, s_code_A)
            XAB = G(c_code_A, s_code_B)

            # Cycle translation
            c_code_BA, s_code_BA = c_Enc(XBA), real_sty_Enc(XBA)
            c_code_AB, s_code_AB = c_Enc(XAB), syn_sty_Enc(XAB)
            XABA = G(c_code_AB, s_code_A) if lambda_cyc > 0 else 0
            XBAB = G(c_code_BA, s_code_B) if lambda_cyc > 0 else 0

            # Enhanced
            en_s_code_A = T(s_code_A)
            en_s_code_B = T(s_code_B)

            en_A = G(c_code_B, en_s_code_A)
            en_B = G(c_code_B, en_s_code_B)

            # -----------------------
            #  Train Discriminator 1 (real fake)
            # -----------------------
            optimizer_D1.zero_grad()

            loss_D1 = D.compute_loss(XA, valid) + D.compute_loss(XBA.detach(), fake) + D.compute_loss(XB,
                                                                                                      valid) + D.compute_loss(
                XAB.detach(), fake)

            loss_D1.backward()
            optimizer_D1.step()

            # -----------------------
            #  Train Generator
            # -----------------------
            optimizer_G.zero_grad()

            # Losses
            loss_GAN_1 = lambda_gan * D.compute_loss(XBA, valid) + D.compute_loss(XAB, valid)
            loss_ID_1 = lambda_id * criterion_recon(XAA, XA)
            loss_ID_2 = lambda_id * criterion_recon(XBB, XB)
            loss_cyc_1 = lambda_cyc * criterion_recon(XABA, XA)
            loss_cyc_2 = lambda_cyc * criterion_recon(XBAB, XB)
            loss_enhanced = lambda_enhanced * (criterion_recon(en_A, labelB) + criterion_recon(en_B, labelB))
            loss_ssim = lambda_ssim * ((1 - ssim_loss(en_A, labelB)) + (1 - ssim_loss(en_B, labelB)))
            loss_perceptual = lambda_perceptual * (perceptual_loss(en_A, labelB) + perceptual_loss(en_B, labelB))
            loss_enhanced_latent = lambda_enhanced_latent * (criterion_recon(en_s_code_B, en_s_code_A))
            loss_tv = lambda_tv * (tv_loss(en_B) + tv_loss(en_A))
            # loss_dcp = lambda_dcp * (dcp_loss(en_B) + dcp_loss(en_A))
            # Total loss
            loss_G = (
                    loss_GAN_1
                    + loss_ID_1
                    + loss_ID_2
                    + loss_cyc_1
                    + loss_cyc_2
                    + loss_enhanced
                    + loss_ssim
                    + loss_enhanced_latent
                    + loss_perceptual
                    + loss_tv
                    # + loss_dcp
            )
            loss_G.backward()
            optimizer_G.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            if i % 1000 == 0:
                E = loss_enhanced.item() + loss_ssim.item() + loss_enhanced_latent.item() + loss_perceptual.item() + loss_tv.item()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                    " [G loss: %f -- {loss_GAN: %f, loss_Identity: %f, Cycle Consistant: %f} ]"
                    " [Enhanced loss %f: [L1: %f, ssim: %f, latent: %f, perceptual: %f, tv: %f]] ETA: %s"
                    % (epoch, opt.n_epochs, i, len(dataloader), (loss_D1).item(),
                       loss_G.item(), (loss_GAN_1.item()),
                       (loss_ID_1.item() + loss_ID_2.item()),
                       (loss_cyc_1.item() + loss_cyc_2.item()),
                       E, loss_enhanced.item(), loss_ssim.item(), loss_enhanced_latent.item(),
                       loss_perceptual.item(), loss_tv.item(), time_left)
                )

        # If at sample interval save image
        if epoch % opt.sample_interval == 0:
            sample_images(epoch)
            print("Snapshot %d" % epoch)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D1.step()

        if epoch % opt.checkpoint_interval == 0 or epoch >= 25:
            # Save model checkpoints
            torch.save(c_Enc.state_dict(), "saved_models/%s/c_Enc_%d.pth" % (opt.exp_name, epoch))
            torch.save(G.state_dict(), "saved_models/%s/G_%d.pth" % (opt.exp_name, epoch))
            torch.save(real_sty_Enc.state_dict(), "saved_models/%s/real_sty_Enc_%d.pth" % (opt.exp_name, epoch))
            torch.save(syn_sty_Enc.state_dict(), "saved_models/%s/syn_sty_Enc_%d.pth" % (opt.exp_name, epoch))
            torch.save(T.state_dict(), "saved_models/%s/T_%d.pth" % (opt.exp_name, epoch))
            torch.save(D.state_dict(), "saved_models/%s/D_%d.pth" % (opt.exp_name, epoch))


if __name__ == '__main__':
    train()
