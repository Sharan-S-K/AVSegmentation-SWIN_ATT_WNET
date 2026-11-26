import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

from dataset import get_dataloaders, load_config
from runtime import build_model, output_directory, setup


def scores(counts):
    tp, fp, fn, tn = [int(x) for x in counts]
    def divide(numerator, denominator):
        return numerator / denominator if denominator else None
    return {'dice': divide(2 * tp, 2 * tp + fp + fn),
            'iou': divide(tp, tp + fp + fn), 'sensitivity': divide(tp, tp + fn),
            'specificity': divide(tn, tn + fp), 'accuracy': divide(tp + tn, tp + tn + fp + fn),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def count(prediction, target, valid):
    prediction, target = prediction[valid], target[valid]
    return np.array([(prediction & target).sum(), (prediction & ~target).sum(),
                     (~prediction & target).sum(), (~prediction & ~target).sum()], dtype=np.int64)


def evaluate_model(model, loader, device, mixed_precision=False, prediction_directory=None):
    model.eval()
    totals = {key: np.zeros(4, dtype=np.int64) for key in
              ['vessel', 'artery_end_to_end', 'vein_end_to_end', 'artery_on_gt_vessels', 'vein_on_gt_vessels']}
    image_count = 0
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    with torch.no_grad():
        for images, vessel_targets, labels in loader:
            with autocast(enabled=mixed_precision and device.type == 'cuda'):
                stage1, stage2 = model(images.to(device), return_deep_sup=False)
            pred_vessel = stage1.argmax(1).cpu().numpy() == 1
            pred_type = (stage2[:, 1:3].argmax(1) + 1).cpu().numpy()
            pred_av = pred_type * pred_vessel
            gt_vessel = vessel_targets.argmax(1).numpy() == 1
            gt_av = labels.argmax(1).numpy()
            valid_av = labels.sum(1).numpy() > 0
            for index in range(len(images)):
                totals['vessel'] += count(pred_vessel[index], gt_vessel[index], np.ones_like(gt_vessel[index], dtype=bool))
                for name, class_id in [('artery', 1), ('vein', 2)]:
                    totals[name + '_end_to_end'] += count(pred_av[index] == class_id, gt_av[index] == class_id, valid_av[index])
                    totals[name + '_on_gt_vessels'] += count(pred_type[index] == class_id, gt_av[index] == class_id,
                                                            valid_av[index] & gt_vessel[index])
                if prediction_directory is not None:
                    rgb = palette[pred_av[index]]
                    cv2.imwrite(str(prediction_directory / '{:03d}_av.png'.format(image_count)), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(prediction_directory / '{:03d}_vessel.png'.format(image_count)), pred_vessel[index].astype(np.uint8) * 255)
                image_count += 1
    return {'images': image_count, 'aggregation': 'pooled pixel counts on preprocessed images',
            'end_to_end': {key: scores(totals[key]) for key in ['vessel', 'artery_end_to_end', 'vein_end_to_end']},
            'conditional_on_ground_truth_vessels': {key: scores(totals[key]) for key in ['artery_on_gt_vessels', 'vein_on_gt_vessels']}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = output_directory(args.config, config)
    checkpoint_path = args.checkpoint or output / 'best.pth'
    if not checkpoint_path.is_file():
        parser.error('Checkpoint not found: ' + str(checkpoint_path))
    device = setup(config, args.device)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = {key[7:] if key.startswith('module.') else key: value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    _, loader = get_dataloaders(args.config, split=args.split)
    prediction_dir = output / (args.split + '_predictions') if args.save_predictions else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    hw = config['hardware_profiles'][config['hardware_profile']]
    results = evaluate_model(model, loader, device, hw.get('use_mixed_precision', False), prediction_dir)
    results['split'] = args.split
    results['checkpoint_epoch'] = checkpoint['epoch']
    destination = output / (args.split + '_metrics.json')
    destination.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    print('Results:', destination)
    return results


if __name__ == '__main__':
    main()
