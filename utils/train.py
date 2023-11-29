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
#import wandb
from evaluate import evaluate
from model import UNet
from dataload import BasicDataset, CarvanaDataset
from dice_score import dice_loss

dir_img = Path('./data/training_augmented/images/')
dir_mask = Path('./data/training_augmented/groundtruth/')
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
        # img_scale: float = 0.5,
        img_scale: float = 1,
        amp: bool = False,                  # Maybe yeet
        weight_decay: float = 1e-8,         # Keeping for now, maybe motivate
        momentum: float = 0.999,            # for adam
        gradient_clipping: float = 1.0,     # what dis?
        verbose: bool = False,
):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)  # Remove
    # except (AssertionError, RuntimeError, IndexError):

    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    #experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    #experiment.config.update(
    #    dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #         val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    #)
    if verbose:
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
    
    mean_losses = []
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
                #save_image(np.array(images[0].cpu()), './images/tests/training_imgs.png')
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # print("Target labels (true_masks):", true_masks)
                # alex = np.array(true_masks.cpu())
                # unique_values = np.unique(alex)
                # print(unique_values)
                #threshold = 127
                #true_masks[true_masks > threshold] = 255
                #true_masks[true_masks <= threshold] = 0
                # print("Target labels (true_masks):", true_masks)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    # if model.n_classes == 1:
                    #     loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    #     loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    # else:
                    #     loss = criterion(masks_pred, true_masks)
                    #     loss += dice_loss(
                    #         F.softmax(masks_pred, dim=1).float(),
                    #         F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    #         multiclass=True
                    #     )
                    
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
                #experiment.log({
                #    'train loss': loss.item(),
                #    'step': global_step,
                #    'epoch': epoch
                #})
                if verbose:
                    logging.info(f'Train Loss: {loss.item()}, Step: {global_step}, Epoch: {epoch}')

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # Create directory for histograms and images if it doesn't exist
                        if not os.path.exists('histograms'):
                            os.makedirs('histograms')
                        if not os.path.exists('images'):
                            os.makedirs('images')

                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                plt.hist(value.data.cpu().numpy().flatten(), bins=50)
                                plt.title(f'Weights Histogram - {tag}')
                                plt.savefig(f'histograms/weights_{tag}_{global_step}.png')
                                plt.close()
                            if value.grad is not None:
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    plt.hist(value.grad.data.cpu().numpy().flatten(), bins=50)
                                    plt.title(f'Gradients Histogram - {tag}')
                                    plt.savefig(f'histograms/gradients_{tag}_{global_step}.png')
                                    plt.close()

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
                        #######################JUST PRINTING AND SAVING IMGs ##############################
                        # # Save images and masks
                        # print(images[0].cpu().numpy().shape)
                        # # save_image(np.array(images[0].cpu()), './images/tests/training_imgs.png')
                        # # For the image
                        # image_tensor = images[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                        # print("shape of it: ", image_tensor.shape)
                        # Image.fromarray(image_tensor).save(f'images/image_{global_step}.png')

                        # For the image
                        image_tensor = images[0].cpu().permute(1, 2, 0).numpy()
                        if image_tensor.min()<0:
                            image_tensor = (image_tensor + 1)/2
                        image_tensor = image_tensor.astype(np.float32)
                        image_tensor = (image_tensor * 255).clip(0, 255).astype(np.uint8)
                        print("shape of it: ", image_tensor.shape)
                        Image.fromarray(image_tensor).save(f'images/imagewow_{global_step}_{learning_rate}.png')

                        # # For the true mask - directly converting to numpy array, as it is 2D
                        true_mask_tensor = true_masks[0].float().cpu().numpy().astype(np.uint8)
                        # unique_value = np.unique(true_mask_tensor)
                        # print("aaaaayyyyy ", unique_value)
                        # Image.fromarray(true_mask_tensor).save(f'images/true_mask_{global_step}.png')

                        true_mask_tensor = true_masks[0]
                        check_and_save_mask(true_mask_tensor, f'images/true_maskwow_{global_step}_{learning_rate}.png')

                        # # For the predicted mask - directly converting to numpy array, as it is 2D
                        pred_mask_tensor = masks_pred.argmax(dim=1)[0].float().cpu().numpy().astype(np.uint8)
                        # Image.fromarray(pred_mask_tensor).save(f'images/pred_mask_{global_step}.png')

                        pred_mask_tensor = masks_pred.argmax(dim=1)[0]
                        check_and_save_mask(pred_mask_tensor, f'images/pred_maskwow_{global_step}_{learning_rate}.png')
            mean_loss = epoch_loss / n_train
            mean_losses.append(mean_loss)
        # Plot and save the graph - Step 3
        plt.figure(figsize=(10, 5))
        plt.plot(mean_losses, label='Mean Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Mean Loss Per Epoch')
        plt.legend()
        plt.savefig(f'loss_plot_epoch.png')  # Save the plot for the current epoch
        plt.close()        

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
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
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')               # REMOVE
    parser.add_argument('--verbose', action='store_true', default=False, help='Display progress information')

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
    
    if args.verbose:
        logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    if args.verbose:
        logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        if args.verbose:
            logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        print("Training model....")

        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            verbose=args.verbose
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
            amp=args.amp
        )

sys.stdout = sys.__stdout__
log_file.close()