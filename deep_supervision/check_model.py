


import argparse
import json
from pathlib import Path
import time
import unittest

import torch
import torch.nn.functional as F

from models import AttentionWNetDeepSup


FULL_SIZE = False


def nll(probabilities, target):

    return F.nll_loss(probabilities.clamp_min(1e-7).log(), target)


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(261)
        cls.model = AttentionWNetDeepSup(in_channels=4)

    def assert_probabilities(self, tensor, shape):
        self.assertEqual(tuple(tensor.shape), shape)
        self.assertTrue(torch.isfinite(tensor).all().item())
        self.assertTrue(((tensor >= 0) & (tensor <= 1)).all().item())
        self.assertLess((tensor.sum(1) - 1).abs().max().item(), 1e-5)

    def assert_gradient(self, gradient):
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_01_output_contract(self):
        self.model.eval()
        with torch.no_grad():
            stage1, stage2, aux1, aux2 = self.model(torch.randn(2, 4, 64, 80))
        self.assertEqual(len(aux1), 3)
        self.assertEqual(len(aux2), 3)
        for output in [stage1] + aux1:
            self.assert_probabilities(output, (2, 2, 64, 80))
        for output in [stage2] + aux2:
            self.assert_probabilities(output, (2, 3, 64, 80))

    def test_02_each_auxiliary_head_reaches_shared_encoder(self):
        self.model.train()
        outputs = self.model(torch.randn(2, 4, 32, 48))
        for stage, aux, classes in [(self.model.unet1, outputs[2], 2),
                                    (self.model.unet2, outputs[3], 3)]:
            target = torch.randint(classes, (2, 32, 48))
            for name, prediction in zip(('ds_conv4', 'ds_conv3', 'ds_conv2'), aux):
                with self.subTest(stage_classes=classes, head=name):
                    gradients = torch.autograd.grad(
                        nll(prediction, target),
                        (getattr(stage, name).weight, stage.Conv1.conv[0].weight),
                        retain_graph=True, allow_unused=True)
                    for gradient in gradients:
                        self.assert_gradient(gradient)

    def test_03_stage2_backpropagates_through_stage1(self):
        self.model.train()
        stage2 = self.model(torch.randn(2, 4, 32, 48))[1]
        gradient = torch.autograd.grad(
            nll(stage2, torch.randint(3, (2, 32, 48))),
            self.model.unet1.Conv1.conv[0].weight, allow_unused=True)[0]
        self.assert_gradient(gradient)

    def test_04_weighted_deep_supervision_optimizer_step(self):
        self.model.train()
        first, second, aux1, aux2 = self.model(torch.randn(2, 4, 32, 48))
        target2 = torch.randint(3, (2, 32, 48))
        target1 = (target2 > 0).long()
        weights = (1.0, 0.5, 0.25, 0.125)
        loss1 = sum(w * nll(p, target1) for w, p in zip(weights, [first] + aux1))
        loss2 = sum(w * nll(p, target2) for w, p in zip(weights, [second] + aux2))
        loss = 0.5 * loss1 + loss2
        self.assertTrue(torch.isfinite(loss).item())
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()
                  if 'ds_conv' in name and name.endswith('weight')}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        optimizer.step()
        self.assertEqual(len(before), 6)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all().item())
            if name in before:
                self.assertFalse(torch.equal(before[name], parameter.detach()), name)

    def test_05_disabling_auxiliary_returns_preserves_main_predictions(self):
        self.model.eval()
        image = torch.randn(1, 4, 32, 48)
        with torch.no_grad():
            full = self.model(image, return_deep_sup=True)
            main = self.model(image, return_deep_sup=False)
        self.assertEqual(len(full), 4)
        self.assertEqual(len(main), 2)
        for first, second in zip(full[:2], main):
            self.assertTrue(torch.equal(first, second))

    def test_06_constant_input(self):
        self.model.eval()
        with torch.no_grad():
            stage1, stage2, aux1, aux2 = self.model(torch.zeros(1, 4, 32, 48))
        for output in [stage1] + aux1:
            self.assert_probabilities(output, (1, 2, 32, 48))
        for output in [stage2] + aux2:
            self.assert_probabilities(output, (1, 3, 32, 48))

    def test_07_background_only_targets_have_finite_loss(self):
        self.model.eval()
        with torch.no_grad():
            first, second = self.model(torch.randn(1, 4, 32, 32), return_deep_sup=False)
            target = torch.zeros(1, 32, 32, dtype=torch.long)
            self.assertTrue(torch.isfinite(nll(first, target) + nll(second, target)).item())

    def test_08_default_rgb_constructor(self):
        model = AttentionWNetDeepSup().eval()
        with torch.no_grad():
            first, second = model(torch.randn(1, 3, 32, 32), return_deep_sup=False)
        self.assert_probabilities(first, (1, 2, 32, 32))
        self.assert_probabilities(second, (1, 3, 32, 32))

    def test_09_actual_deep_supervision_objective(self):
        from objective import CombinedDeepSupLoss
        settings1 = {'alpha': .5, 'gamma': 2, 'dice_weight': .8, 'boundary_weight': .1}
        settings2 = {'alpha': 1., 'gamma': 3, 'dice_weight': .5, 'tv_weight': .01, 'boundary_weight': .15}
        weights = [1., .5, .25, .125]
        criterion = CombinedDeepSupLoss(settings1, settings2, .5, 1., weights)
        self.model.train()
        outputs = self.model(torch.randn(2, 4, 32, 32))
        target = torch.randint(3, (2, 32, 32))
        labels = F.one_hot(target, 3).permute(0, 3, 1, 2).float()
        vessels = F.one_hot((target > 0).long(), 2).permute(0, 3, 1, 2).float()
        total, first, second = criterion(*outputs, vessels, labels)
        mask = vessels[:, 1:2]
        expected1 = sum(w * criterion.criterion1(p, vessels) for w, p in zip(weights, [outputs[0]] + outputs[2]))
        expected2 = sum(w * criterion.criterion2(p * mask, labels * mask, mask=mask)
                        for w, p in zip(weights, [outputs[1]] + outputs[3]))
        self.assertTrue(torch.allclose(total, .5 * expected1 + expected2))
        gradients = torch.autograd.grad(total, [self.model.unet1.ds_conv4.weight,
                                               self.model.unet2.ds_conv4.weight])
        for gradient in gradients:
            self.assert_gradient(gradient)

    def test_11_boundary_loss_has_nonzero_prediction_gradients(self):
        from losses import BoundaryLoss
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = F.one_hot(torch.randint(3, (2, 32, 32)), 3).permute(0, 3, 1, 2).float()
        loss = BoundaryLoss()(logits.softmax(1), target)
        self.assertTrue(loss.requires_grad)
        self.assert_gradient(torch.autograd.grad(loss, logits)[0])

    def test_12_random_spatial_transforms_preserve_channel_alignment(self):
        import numpy as np
        from dataset import DataAugmentation
        config = {'augmentation': {'enabled': True, 'elastic_transform_prob': 1,
                                  'grid_distortion_prob': 1, 'elastic_alpha': 4,
                                  'elastic_sigma': 2}}
        image = np.random.RandomState(13).randint(0, 256, (64, 64, 3)).astype(np.uint8)
        result, _, original = DataAugmentation(config)(image[:, :, :1].copy(), image.copy(), image.copy())
        np.testing.assert_array_equal(result[:, :, 0], original[:, :, 0])

    def test_13_dataset_splits_and_shapes(self):
        from dataset import get_dataset_paths, get_dataloaders, load_config
        config_path = Path(__file__).with_name('config.json')
        train, _, val, _ = get_dataset_paths(config_path)
        _, _, test, _ = get_dataset_paths(config_path, split='test')
        self.assertTrue(set(train).isdisjoint(val))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(val).isdisjoint(test))
        loader, _ = get_dataloaders(config_path)
        image, vessel, label = next(iter(loader))
        settings = load_config(config_path)
        size = settings['hardware_profiles'][settings['hardware_profile']]['image_size']
        self.assertEqual(tuple(image.shape), (2, 4, size, size))
        self.assertEqual(tuple(vessel.shape), (2, 2, size, size))
        self.assertEqual(tuple(label.shape), (2, 3, size, size))

    def test_14_unknown_vessel_colors_are_not_artery_vein_background_targets(self):
        import cv2
        import tempfile
        import numpy as np
        from dataset import AVSegDataset, load_config
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            label = image.copy()
            label[10, 10] = [0, 255, 0]
            cv2.imwrite(str(directory / 'image.png'), image)
            cv2.imwrite(str(directory / 'mask.png'), label)
            config = load_config(Path(__file__).with_name('config.json'))
            config['hardware_profiles'][config['hardware_profile']]['image_size'] = 64
            dataset = AVSegDataset([str(directory / 'image.png')], [str(directory / 'mask.png')], config, is_train=False)
            _, vessels, classes = dataset[0]
            self.assertEqual(vessels[1, 10, 10].item(), 1.)
            self.assertEqual(classes[:, 10, 10].sum().item(), 0.)

    def test_10_full_608_square_inference(self):
        if not FULL_SIZE:
            self.skipTest('Use --full-size to include 608x608 CPU inference.')
        self.model.eval()
        with torch.no_grad():
            first, second = self.model(torch.randn(1, 4, 608, 608), return_deep_sup=False)
        self.assert_probabilities(first, (1, 2, 608, 608))
        self.assert_probabilities(second, (1, 3, 608, 608))


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test.id())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic CPU checks for Attention-WNet')
    parser.add_argument('--full-size', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    FULL_SIZE = args.full_size
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ModelChecks))
    if args.report:
        report = {'device': 'cpu', 'pytorch': torch.__version__, 'seed': 261,
                  'data': 'synthetic only', 'checkpoint_loaded': False,
                  'objective': 'NLL of model probabilities, for gradient checks only',
                  'full_size_enabled': FULL_SIZE, 'tests_run': result.testsRun,
                  'passed': result.successes,
                  'skipped': [(t.id(), why) for t, why in result.skipped],
                  'failures': [(t.id(), why) for t, why in result.failures],
                  'errors': [(t.id(), why) for t, why in result.errors],
                  'seconds': round(time.perf_counter() - start, 3)}
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
