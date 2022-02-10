import os
import random

import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg", ".JPG", ".PNG"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def get_patch(imgs, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = imgs[0].size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale
    if ix == -1:
        # random.seed(123)
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        # random.seed(123)
        iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    outs = []
    for img in imgs:
        outs.append(img.crop((ty, tx, ty + tp, tx + tp)))

    return tuple(outs)


def augmentation(imgs, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    # random.seed(123)
    if random.random() < 0.5 and flip_h:
        for i in range(len(imgs)):
            imgs[i] = ImageOps.flip(imgs[i])
        info_aug['flip_h'] = True
    if rot:
        # random.seed(123)
        if random.random() < 0.5:
            for i in range(len(imgs)):
                imgs[i] = ImageOps.mirror(imgs[i])
            info_aug['flip_v'] = True
        # random.seed(123)
        if random.random() < 0.5:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].rotate(180)
        info_aug['trans'] = True
    return tuple(imgs), info_aug


class EnhancedDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", patch_size=128):
        super(EnhancedDataset, self).__init__()
        self.transform = transforms.Compose(transforms_)
        if mode == 'train':
            self.filesA = [os.path.join(os.path.join(root, 'trainA'), x) for x in
                           os.listdir(os.path.join(root, 'trainA')) if is_image_file(x)]
            self.filesB = [os.path.join(os.path.join(root, 'trainB'), x) for x in
                           os.listdir(os.path.join(root, 'trainB')) if is_image_file(x)]
            self.labelB = [os.path.join(os.path.join(root, 'trainB_label'), x) for x in
                           os.listdir(os.path.join(root, 'trainB_label')) if is_image_file(x)]
        else:
            self.filesA = [os.path.join(os.path.join(root, 'testA'), x) for x in
                           os.listdir(os.path.join(root, 'testA')) if is_image_file(x)]
            # sorted(glob.glob( + "/*.*"))
            self.filesB = [os.path.join(os.path.join(root, 'testB'), x) for x in
                           os.listdir(os.path.join(root, 'testB')) if is_image_file(x)]
            self.labelB = [os.path.join(os.path.join(root, 'testB_label'), x) for x in
                           os.listdir(os.path.join(root, 'testB_label')) if is_image_file(x)]

        # if mode == 'train':
        #     self.filesA = [os.path.join(os.path.join(root, 'trainA_UIEBwithRUIE'), x) for x in  # _UIEBwithRUIE
        #                    os.listdir(os.path.join(root, 'trainA_UIEBwithRUIE')) if is_image_file(x)]
        #     self.filesB = [os.path.join(os.path.join(root, 'trainB_NYU'), x) for x in
        #                    os.listdir(os.path.join(root, 'trainB_NYU')) if is_image_file(x)]
        #     self.labelB = [os.path.join(os.path.join(root, 'trainB_NYU_label'), x) for x in
        #                    os.listdir(os.path.join(root, 'trainB_NYU_label')) if is_image_file(x)]
        # else:
        #     self.filesA = [os.path.join(os.path.join(root, 'testA_UIEBwithRUIE'), x) for x in
        #                    os.listdir(os.path.join(root, 'testA_UIEBwithRUIE')) if is_image_file(x)]
        #     # sorted(glob.glob( + "/*.*"))
        #     self.filesB = [os.path.join(os.path.join(root, 'testB'), x) for x in
        #                    os.listdir(os.path.join(root, 'testB')) if is_image_file(x)]
        #     self.labelB = [os.path.join(os.path.join(root, 'testB_label'), x) for x in
        #                    os.listdir(os.path.join(root, 'testB_label')) if is_image_file(x)]
        self.patch_size = patch_size
        self.mode = mode

    def __getitem__(self, index):
        img_A = load_img(self.filesA[index % len(self.filesA)])
        img_B = load_img(self.filesB[index])
        label_B = load_img(self.labelB[index])

        if self.mode == 'train':
            img_A = get_patch(imgs=[img_A], patch_size=self.patch_size, scale=1)[0]
            (img_B, label_B) = get_patch(imgs=[img_B, label_B],
                                         patch_size=self.patch_size, scale=1)

            (img_A, img_B, label_B), _ = augmentation(imgs=[img_A, img_B, label_B])
        if self.mode == 'val':
            w, h = img_A.size
            new_w, new_h = w, h
            if (w / 4) % 1 != 0:
                new_w = w // 4 * 4
            if (h / 4) % 1 != 0:
                new_h = h // 4 * 4
            if new_w != w or new_h != h:
                img_A = img_A.resize((new_w, new_h))

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        label_B = self.transform(label_B)

        return {"Real": img_A, "Syn": img_B, "label": label_B}

    def __len__(self):
        if self.mode == 'val':
            return len(self.filesA)
        else:
            return len(self.filesB)


class EnhancedValDataset(Dataset):
    def __init__(self, transforms_=None, dataset_path="train", patch_size=128):
        super(EnhancedValDataset, self).__init__()
        self.transform = transforms.Compose(transforms_)
        self.files = [os.path.join(dataset_path, x) for x in
                       os.listdir(dataset_path) if is_image_file(x)]
        # sorted(glob.glob( + "/*.*"))
        self.patch_size = patch_size

    def __getitem__(self, index):

        img = load_img(self.files[index % len(self.files)])

        w, h = img.size
        new_w, new_h = w, h
        if (w / 4) % 1 != 0:
            new_w = w // 4 * 4
        if (h / 4) % 1 != 0:
            new_h = h // 4 * 4
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h))

        img = self.transform(img)

        return {"img": img, 'name': self.files[index % len(self.files)]}

    def __len__(self):
        return len(self.files)
