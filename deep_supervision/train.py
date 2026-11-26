import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders, load_config
from objective import CombinedDeepSupLoss
from runtime import build_model, output_directory, setup


def compute_loss(model, images, vessels, labels, criterion, deep_supervision):
    outputs = model(images, return_deep_sup=deep_supervision)
    if deep_supervision:
        return criterion(*outputs, vessels, labels)
    return criterion(outputs[0], outputs[1], [], [], vessels, labels)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    hp = config['hyperparameters']
    if args.epochs is not None:
        hp['epochs'] = args.epochs
    if hp['epochs'] < 1:
        raise ValueError('epochs must be positive')
    device = setup(config, args.device)
    output = output_directory(args.config, config)
    train_loader, val_loader = get_dataloaders(args.config, split='val')
    if not len(train_loader) or not len(val_loader):
        raise ValueError('Training and validation loaders must be nonempty')
    model = build_model(config).to(device)
    hw = config['hardware_profiles'][config['hardware_profile']]
    if hw.get('use_data_parallel', False) and device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    deep_supervision = hp.get('deep_supervision', {}).get('enabled', True)
    weights = hp.get('deep_supervision', {}).get('weights', [1, .5, .25, .125])
    if len(weights) != 4 or weights[0] <= 0 or any(w < 0 for w in weights):
        raise ValueError('Provide four nonnegative supervision weights with positive main weight')
    criterion = CombinedDeepSupLoss(hp['stage1_loss_kwargs'], hp['stage2_loss_kwargs'],
                                    hp['stage1_weight'], hp['stage2_weight'], weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5)
    accumulation = hp.get('gradient_accumulation_steps', 1)
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError('gradient_accumulation_steps must be a positive integer')
    mixed_precision = hw.get('use_mixed_precision', False) and device.type == 'cuda'
    scaler = GradScaler(enabled=mixed_precision)
    writer = SummaryWriter(str(output / 'tensorboard'))
    history = []
    optimizer_steps = 0
    best_loss = float('inf')
    patience = 0
    print('Device:', device, '| Parameters:', sum(p.numel() for p in model.parameters()))
    try:
        for epoch in range(hp['epochs']):
            model.train()
            optimizer.zero_grad()
            train_total = 0.0
            for index, (images, vessels, labels) in enumerate(tqdm(train_loader, desc='Epoch {}'.format(epoch + 1))):
                images, vessels, labels = [t.to(device) for t in (images, vessels, labels)]
                group_start = (index // accumulation) * accumulation
                group_size = min(accumulation, len(train_loader) - group_start)
                with autocast(enabled=mixed_precision):
                    loss, _, _ = compute_loss(model, images, vessels, labels, criterion, deep_supervision)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss / group_size).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    optimizer_steps += 1
                train_total += loss.item()
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for images, vessels, labels in val_loader:
                    images, vessels, labels = [t.to(device) for t in (images, vessels, labels)]
                    with autocast(enabled=mixed_precision):
                        loss, _, _ = compute_loss(model, images, vessels, labels, criterion, deep_supervision)
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError('Nonfinite validation loss')
                    val_total += loss.item()
            train_loss = train_total / len(train_loader)
            val_loss = val_total / len(val_loader)
            scheduler.step(val_loss)
            row = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                   'optimizer_steps': optimizer_steps, 'learning_rate': optimizer.param_groups[0]['lr']}
            history.append(row)
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Loss/validation', val_loss, epoch + 1)
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint = {'epoch': epoch + 1, 'state_dict': state, 'config': config,
                          'optimizer': optimizer.state_dict(), 'optimizer_steps': optimizer_steps,
                          'validation_loss': val_loss}
            torch.save(checkpoint, output / 'last.pth')
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                torch.save(checkpoint, output / 'best.pth')
            else:
                patience += 1
            (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(row))
            if patience >= hp.get('early_stopping_patience', 15):
                break
    finally:
        writer.close()
    print('Best checkpoint:', output / 'best.pth')
    return history


if __name__ == '__main__':
    main()
