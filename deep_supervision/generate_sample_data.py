import argparse
from pathlib import Path

import cv2
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('synthetic_data'))
    args = parser.parse_args(argv)
    for split_index, (split, count) in enumerate([('train', 6), ('val', 2), ('test', 2)]):
        image_dir = args.output / split / 'images'
        label_dir = args.output / split / 'labels'
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            rng = np.random.RandomState(261 + split_index * 100 + index)
            size = 64
            yy, xx = np.mgrid[:size, :size]
            field = (xx - 31.5) ** 2 + (yy - 31.5) ** 2 < 30 ** 2
            image = np.zeros((size, size, 3), dtype=np.uint8)
            texture = rng.randint(0, 14, (size, size))
            image[:, :, 0] = np.where(field, 110 + texture, 0)
            image[:, :, 1] = np.where(field, 48 + texture, 0)
            image[:, :, 2] = np.where(field, 25 + texture, 0)
            labels = np.zeros_like(image)
            for vessel_type in [1, 2]:
                mask = np.zeros((size, size), dtype=np.uint8)
                x0 = int(rng.randint(15, 27) if vessel_type == 1 else rng.randint(38, 49))
                points = np.array([[x0, 7], [x0 + int(rng.randint(-4, 5)), 25],
                                   [x0 + int(rng.randint(-6, 7)), 43], [x0 + int(rng.randint(-5, 6)), 57]], np.int32)
                cv2.polylines(mask, [points], False, 255, int(rng.randint(2, 4)))
                branch = np.array([points[1], points[1] + [(-9 if vessel_type == 1 else 9), 9],
                                   points[1] + [(-14 if vessel_type == 1 else 14), 15]], np.int32)
                cv2.polylines(mask, [branch], False, 255, 1)
                region = (mask > 0) & field
                labels[region] = [255, 0, 0] if vessel_type == 1 else [0, 0, 255]
                image[region] = [180, 110, 70] if vessel_type == 1 else [45, 25, 40]
            stem = '{}_{:02d}.png'.format(split, index)
            if not cv2.imwrite(str(image_dir / stem), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError('Could not write image')
            if not cv2.imwrite(str(label_dir / stem), cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)):
                raise OSError('Could not write label')
    print('Synthetic example pairs:', args.output)


if __name__ == '__main__':
    main()
