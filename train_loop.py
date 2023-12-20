import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils.evaluate import evaluate
from utils.model import UNet
from utils.leaky_model import LeakyUNet
from utils.dataload import BasicDataset
from utils.dice_score import dice_loss

transforms = ['original', 'hue', 'contrast', 'brightness', 'saturation']

datasets = {
    'original': {
        'images': Path('./data/train_aug++_transf/train_original/images/'),
        'groundtruth': Path('./data/train_aug++_transf/train_original/groundtruth/')
    },
    'hue': {
        'images': Path('./data/train_aug++_transf/train_hue/images/'),
        'groundtruth': Path('./data/train_aug++_transf/train_hue/groundtruth/')
    },
    'contrast': {
        'images': Path('./data/train_aug++_transf/train_contrast/images/'),
        'groundtruth': Path('./data/train_aug++_transf/train_contrast/groundtruth/')
    },
    'brightness': {
        'images': Path('./data/train_aug++_transf/train_brightness/images/'),
        'groundtruth': Path('./data/train_aug++_transf/train_brightness/groundtruth/')
    },
    'saturation': {
        'images': Path('./data/train_aug++_transf/train_saturation/images/'),
        'groundtruth': Path('./data/train_aug++_transf/train_saturation/groundtruth/')
    },
    '': {
        'images': Path(f'./data/train/images/'),
        'groundtruth': Path(f'./data/train/groundtruth/')
    }
}


dir_img = Path(f'./data/training/images/')
dir_mask = Path(f'./data/training/groundtruth/')

dir_checkpoint = Path('./checkpoints/')

log_file = open('script_output.log', 'w')
sys.stdout = log_file

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        dirs: tuple = (dir_img, dir_mask),
        tran: str = 'original'
):

    # 1. Create dataset
    dataset = BasicDataset(dirs[0], dirs[1], img_scale)


    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    
    logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() # if model.n_classes > 1 else nn.BCEWithLogitsLoss()    # Change to the 2 class version only
    global_step = 0
    
    train_losses = []
    val_losses = []
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, 2).permute(0, 3, 1, 2).float()
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                logging.info(f'Train Loss: {loss.item()}, Step: {global_step}, Epoch: {epoch}')

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))

                        # For the image
                        image_tensor = images[0].cpu().permute(1, 2, 0).numpy()
                        if image_tensor.min()<0:
                            image_tensor = (image_tensor + 1)/2
                        image_tensor = image_tensor.astype(np.float32)
                        image_tensor = (image_tensor * 255).clip(0, 255).astype(np.uint8)
                        print("shape of it: ", image_tensor.shape)

                        # # For the true mask - directly converting to numpy array, as it is 2D
                        true_mask_tensor = true_masks[0].float().cpu().numpy().astype(np.uint8)

                        true_mask_tensor = true_masks[0]

                        # # For the predicted mask - directly converting to numpy array, as it is 2D
                        pred_mask_tensor = masks_pred.argmax(dim=1)[0].float().cpu().numpy().astype(np.uint8)

                        pred_mask_tensor = masks_pred.argmax(dim=1)[0]
                        # check_and_save_mask(pred_mask_tensor, f'images/pred_maskwow_{global_step}_{learning_rate}.png')
            train_loss = epoch_loss / n_train
            train_losses.append(train_loss)

            # Calculate validation loss
            model.eval()  # Set the model to evaluation mode
            val_loss = 0
            with torch.no_grad():  # No gradients needed for validation
                for batch in val_loader:
                    images, true_masks = batch['image'], batch['mask']
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks) + dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()
                    )
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
        # Plot and save the graph - Step 3
        ep = np.arange(1, epoch+1)
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([-0.2, 1])
        plt.title(f'Training and Validation Loss Per Epoch ({tran})')
        plt.legend()
        plt.savefig(f'training_validation_loss_plot_{tran}.png')
        plt.close()      

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}_{tran}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--leaky', action='store_true', default=False, help='Use leaky ReLU')
    parser.add_argument('--dataset', type=str, default='original', help='Type of image transformation to apply')

    return parser.parse_args()

def save_image(image_array, file_path):
    # Reshape the array to a proper image format if needed
    if len(image_array.shape) == 3:
        # Assuming the image is in the format (channels, height, width)
        if image_array.shape[0] in [1, 3, 4]:  # Grayscale or RGB or RGBA
            image_array = np.transpose(image_array, (1, 2, 0))  # to (height, width, channels)
        else:
            raise ValueError("Unsupported channel number: {}".format(image_array.shape[0]))
    elif len(image_array.shape) != 2:
        raise ValueError("Unsupported array shape: {}".format(image_array.shape))

    # Ensure the array is of type uint8
    image_array = (image_array * 255).astype(np.uint8)

    # Convert to image and save
    image = Image.fromarray(image_array)
    image.save(file_path)

def check_and_save_mask(tensor, file_name):
    print(f"Tensor Size: {tensor.size()}")
    print(f"Tensor Type: {tensor.dtype}")

    # Convert to numpy and check values
    tensor_numpy = tensor.float().cpu().numpy()
    unique_values = np.unique(tensor_numpy)
    print(f"Unique values in tensor: {unique_values}")

    # Scaling if necessary (example: scale to 0-255 range)
    if unique_values.max() < 255:
        tensor_numpy = (tensor_numpy / unique_values.max()) * 255

    # Ensure type is uint8
    tensor_numpy = tensor_numpy.astype(np.uint8)

    # Save the image
    Image.fromarray(tensor_numpy).save(file_name)

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    # model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear) # original
    if args.leaky:
        model = LeakyUNet(n_channels=3, n_classes=2, bilinear=False)
    else:
        model = UNet(n_channels=3, n_classes=2, bilinear=False)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                f'\t{model.n_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    for tran in transforms:
        directories = (datasets[tran]['images'], datasets[tran]['groundtruth'])
        start_time = datetime.now()
        try:

            train_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                dirs=directories,
                tran=tran
            )
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError! '
                        'Enabling checkpointing to reduce memory usage, but this slows down training. '
                        'Consider enabling AMP (--amp) for fast and memory efficient training')
            torch.cuda.empty_cache()
            model.use_checkpointing()
            train_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                dirs=directories,
                tran=tran
            )
        end_time = datetime.now()  # Capture end time
        duration = end_time - start_time  # Calculate duration

        # Formatting the duration into hh:mm:ss
        duration_formatted = str(duration).split('.')[0]

        print(f"Transformation: {tran}")
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration_formatted}")

sys.stdout = sys.__stdout__
log_file.close()