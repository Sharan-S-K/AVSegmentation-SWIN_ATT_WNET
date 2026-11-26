import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import random
from scipy.ndimage import gaussian_filter

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_dataset_paths(config_path, split='val'):
    from pathlib import Path
    config = load_config(config_path)
    base = Path(config_path).resolve().parent
    splits = config.get('dataset_splits')
    if not splits or 'train' not in splits or split not in splits:
        raise ValueError("Configure separate train, val and test directories in dataset_splits")
    all_pairs = {}
    seen = set()
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    for name, directories in splits.items():
        image_dir = (base / directories['images']).resolve()
        label_dir = (base / directories['labels']).resolve()
        def by_stem(folder):
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]
            result = {p.stem: p for p in files}
            if len(result) != len(files):
                raise ValueError("Duplicate filename stems in " + str(folder))
            return result
        images = by_stem(image_dir)
        labels = by_stem(label_dir)
        if not images or images.keys() != labels.keys():
            raise ValueError("Each split needs nonempty image/mask pairs with matching stems: " + name)
        resolved = {str(p.resolve()) for p in images.values()}
        if seen & resolved:
            raise ValueError("An image appears in more than one split")
        seen.update(resolved)
        all_pairs[name] = [(str(images[stem]), str(labels[stem])) for stem in sorted(images)]
    train = all_pairs['train']
    evaluation = all_pairs[split]
    return ([p[0] for p in train], [p[1] for p in train],
            [p[0] for p in evaluation], [p[1] for p in evaluation])


class DataAugmentation:


    def __init__(self, config):
        self.aug_config = config.get('augmentation', {})
        self.enabled = self.aug_config.get('enabled', False)

    def warp_together(self, transform, image, label, original, *args):
        if original is None:
            image, label = transform(image, label, *args)
            return image, label, None
        if image.ndim == 2:
            image = image[:, :, None]
        channels = image.shape[2]
        combined = np.concatenate([image, original], axis=2)
        combined, label = transform(combined, label, *args)
        return combined[:, :, :channels], label, combined[:, :, channels:]

    def elastic_transform(self, image, label, alpha=50, sigma=5):

        shape = image.shape[:2]

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (np.clip(y + dy, 0, shape[0]-1).astype(int),
                   np.clip(x + dx, 0, shape[1]-1).astype(int))

        if len(image.shape) == 3:
            transformed_image = np.zeros_like(image)
            for c in range(image.shape[2]):
                transformed_image[:, :, c] = image[:, :, c][indices]
        else:
            transformed_image = image[indices]

        if len(label.shape) == 3:
            transformed_label = np.zeros_like(label)
            for c in range(label.shape[2]):
                transformed_label[:, :, c] = label[:, :, c][indices]
        else:
            transformed_label = label[indices]

        return transformed_image, transformed_label

    def grid_distortion(self, image, label, num_steps=5, distort_limit=0.3):

        h, w = image.shape[:2]

        xx = np.linspace(0, w, num_steps)
        yy = np.linspace(0, h, num_steps)

        xx_new = xx + np.random.uniform(-distort_limit, distort_limit, num_steps) * w / num_steps
        yy_new = yy + np.random.uniform(-distort_limit, distort_limit, num_steps) * h / num_steps

        xx_new = np.clip(xx_new, 0, w-1)
        yy_new = np.clip(yy_new, 0, h-1)

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for i in range(h):
            map_x[i, :] = np.interp(np.linspace(0, w, w), xx_new, xx)
        for j in range(w):
            map_y[:, j] = np.interp(np.linspace(0, h, h), yy_new, yy)

        transformed_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        transformed_label = cv2.remap(label, map_x, map_y, cv2.INTER_NEAREST)

        return transformed_image, transformed_label

    def coarse_dropout(self, image, label, max_holes=8, max_size=32):

        h, w = image.shape[:2]

        num_holes = np.random.randint(1, max_holes + 1)

        for _ in range(num_holes):
            hole_h = np.random.randint(max_size // 2, max_size)
            hole_w = np.random.randint(max_size // 2, max_size)

            y = np.random.randint(0, h - hole_h)
            x = np.random.randint(0, w - hole_w)

            image[y:y+hole_h, x:x+hole_w] = 0

        return image, label

    def __call__(self, image, label, orig_image=None):


        if not self.enabled:
            return image, label, orig_image


        if random.random() < self.aug_config.get('horizontal_flip', 0):
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
            if orig_image is not None:
                orig_image = np.fliplr(orig_image).copy()


        if random.random() < self.aug_config.get('vertical_flip', 0):
            image = np.flipud(image).copy()
            label = np.flipud(label).copy()
            if orig_image is not None:
                orig_image = np.flipud(orig_image).copy()


        rotation_degrees = self.aug_config.get('rotation_degrees', 0)
        if rotation_degrees > 0:
            angle = random.uniform(-rotation_degrees, rotation_degrees)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            if orig_image is not None:
                orig_image = cv2.warpAffine(orig_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


        scale_range = self.aug_config.get('scale_range', [1.0, 1.0])
        if scale_range[0] != 1.0 or scale_range[1] != 1.0:
            scale = random.uniform(scale_range[0], scale_range[1])
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)

            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if orig_image is not None:
                orig_image = cv2.resize(orig_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


            if scale > 1.0:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = image[start_h:start_h+h, start_w:start_w+w]
                label = label[start_h:start_h+h, start_w:start_w+w]
                if orig_image is not None:
                    orig_image = orig_image[start_h:start_h+h, start_w:start_w+w]
            else:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = np.pad(image, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)) + ((0,0),) * (len(image.shape)-2), mode='reflect')
                label = np.pad(label, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0,0)), mode='constant')
                if orig_image is not None:
                    orig_image = np.pad(orig_image, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0,0)), mode='reflect')


        if random.random() < self.aug_config.get('elastic_transform_prob', 0):
            alpha = self.aug_config.get('elastic_alpha', 50)
            sigma = self.aug_config.get('elastic_sigma', 5)
            image, label, orig_image = self.warp_together(self.elastic_transform, image, label, orig_image, alpha, sigma)


        if random.random() < self.aug_config.get('grid_distortion_prob', 0):
            image, label, orig_image = self.warp_together(self.grid_distortion, image, label, orig_image)


        if random.random() < self.aug_config.get('gaussian_blur_prob', 0):
            kernel_range = self.aug_config.get('gaussian_blur_kernel', [3, 7])
            kernel_size = random.choice(range(kernel_range[0], kernel_range[1]+1, 2))
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            if orig_image is not None:
                orig_image = cv2.GaussianBlur(orig_image, (kernel_size, kernel_size), 0)


        if random.random() < self.aug_config.get('coarse_dropout_prob', 0):
            max_holes = self.aug_config.get('coarse_dropout_max_holes', 8)
            max_size = self.aug_config.get('coarse_dropout_max_size', 32)
            image, label = self.coarse_dropout(image.copy(), label, max_holes, max_size)

        return image, label, orig_image


class AVSegDataset(Dataset):
    def __init__(self, image_paths, label_paths, config, is_train=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.is_train = is_train
        self.channel_mode = config.get('channels', 'rgb')
        self.use_lab_colorspace = config.get('use_lab_colorspace', False)

        hp_profile = config['hardware_profile']
        self.hp = config['hardware_profiles'][hp_profile]
        self.strategy = self.hp['image_processing_strategy']
        self.image_size = self.hp['image_size']


        self.augmentation = DataAugmentation(config) if is_train else None
        self.aug_config = config.get('augmentation', {})

    def __len__(self):
        return len(self.image_paths)

    def apply_color_augmentation(self, img):

        if not self.is_train or not self.aug_config.get('enabled', False):
            return img

        img = img.astype(np.float32)


        brightness = self.aug_config.get('brightness', 0)
        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            img = img * factor


        contrast = self.aug_config.get('contrast', 0)
        if contrast > 0:
            factor = 1.0 + random.uniform(-contrast, contrast)
            mean = img.mean()
            img = (img - mean) * factor + mean


        saturation = self.aug_config.get('saturation', 0)
        if saturation > 0 and len(img.shape) == 3 and img.shape[2] == 3:
            factor = 1.0 + random.uniform(-saturation, saturation)
            gray = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = np.stack([gray] * 3, axis=-1)
            img = img * factor + gray * (1 - factor)

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]


        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        lbl = cv2.imread(lbl_path)
        if lbl is None:
            raise ValueError(f"Could not read label: {lbl_path}")
        lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != lbl.shape[:2]:
            raise ValueError('Image and label dimensions differ: ' + img_path)


        if self.is_train:
            img = self.apply_color_augmentation(img)

        orig_img = img.copy()


        if self.channel_mode == 'green_only':
            img_green = img[:, :, 1]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_green = clahe.apply(img_green)
            img_processed = np.expand_dims(img_green, axis=-1)
        else:
            img_processed = img


        if self.augmentation is not None:
            img_processed, lbl, orig_img = self.augmentation(img_processed, lbl, orig_img)


        if img_processed.ndim == 2:
            img_processed = img_processed[:, :, None]
        img_tensor = TF.to_tensor(np.ascontiguousarray(img_processed))

        if self.use_lab_colorspace:
            lab_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2LAB)
            lab_tensor = TF.to_tensor(lab_img)
            img_tensor = torch.cat([img_tensor, lab_tensor], dim=0)

        lbl_tensor = TF.to_tensor(lbl)


        if self.strategy == 'resize':
            img_tensor = TF.resize(img_tensor, [self.image_size, self.image_size], antialias=True)
            lbl_tensor = TF.resize(lbl_tensor, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        elif self.strategy == 'dynamic_resize':
            h, w = img_tensor.shape[-2:]
            if h > self.image_size or w > self.image_size:
                img_tensor = TF.resize(img_tensor, [self.image_size, self.image_size], antialias=True)
                lbl_tensor = TF.resize(lbl_tensor, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
            else:
                pad_h = self.image_size - h
                pad_w = self.image_size - w
                img_tensor = TF.pad(img_tensor, [0, 0, pad_w, pad_h], fill=0)
                lbl_tensor = TF.pad(lbl_tensor, [0, 0, pad_w, pad_h], fill=0)

        elif self.strategy == 'patching':
            pad_h = max(0, self.image_size - img_tensor.shape[-2])
            pad_w = max(0, self.image_size - img_tensor.shape[-1])
            img_tensor = TF.pad(img_tensor, [0, 0, pad_w, pad_h])
            lbl_tensor = TF.pad(lbl_tensor, [0, 0, pad_w, pad_h])
            if self.is_train:
                i, j, h, w = random_crop_params(img_tensor, self.image_size)
                img_tensor = TF.crop(img_tensor, i, j, h, w)
                lbl_tensor = TF.crop(lbl_tensor, i, j, h, w)
            else:
                img_tensor = TF.center_crop(img_tensor, [self.image_size, self.image_size])
                lbl_tensor = TF.center_crop(lbl_tensor, [self.image_size, self.image_size])

        else:
            raise ValueError('Unknown image_processing_strategy: ' + self.strategy)


        vessel_present = (torch.sum(lbl_tensor, dim=0, keepdim=True) > 0).float()
        bg_present = 1.0 - vessel_present
        vessel_mask = torch.cat([bg_present, vessel_present], dim=0)


        red_mask = (lbl_tensor[0] > 0.5) & (lbl_tensor[1] < 0.5) & (lbl_tensor[2] < 0.5)
        blue_mask = (lbl_tensor[0] < 0.5) & (lbl_tensor[1] < 0.5) & (lbl_tensor[2] > 0.5)

        class_indices = torch.zeros((lbl_tensor.shape[1], lbl_tensor.shape[2]), dtype=torch.long)
        class_indices[red_mask] = 1
        class_indices[blue_mask] = 2

        lbl_one_hot = torch.nn.functional.one_hot(class_indices, num_classes=3).permute(2, 0, 1).float()

        uncertain = (vessel_present[0] > 0) & (class_indices == 0)
        lbl_one_hot[:, uncertain] = 0
        return img_tensor, vessel_mask, lbl_one_hot


def random_crop_params(img, output_size):
    h, w = img.shape[-2:]
    th, tw = output_size, output_size
    if w == tw and h == th:
        return 0, 0, h, w

    if h < th or w < tw:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


def get_dataloaders(config_path, split='val'):
    config = load_config(config_path)
    hp_profile = config['hardware_profile']
    batch_size = config['hardware_profiles'][hp_profile]['batch_size']
    num_workers = config['hardware_profiles'][hp_profile].get('num_workers', 0)
    pin_memory = config['hardware_profiles'][hp_profile].get('pin_memory', True)

    train_imgs, train_lbls, test_imgs, test_lbls = get_dataset_paths(config_path, split=split)

    print(f"Training images: {len(train_imgs)}")
    print(f"{split} images: {len(test_imgs)}")

    train_dataset = AVSegDataset(train_imgs, train_lbls, config, is_train=True)
    test_dataset = AVSegDataset(test_imgs, test_lbls, config, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
