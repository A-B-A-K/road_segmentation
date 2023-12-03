#  CS-433: Road Segmentation

## Setup

The training and testing data we use is in the `data`. 

The training directory (in our case `training_augmented`, we will talk about the augmentation later) has to have the following structure:
```bash
/training_set
    /ground_truth
    /images
```
The test directory (in our case `test_set_images`) has to have the following structure:
```bash
/test_set
    /test_i
        test_i.png
```

## Training

To perform the training the following line of code has to be run:
```bash
python train.py --amp
```
Additional flags:
- `--epochs $num_epochs`
- `--learning-rate $lr`

## Testing

After training the model and saving it to `MODEL.pth`, it can be tested through the following CLI commands:

Test single image (no image partition):
```bash
python predict.py -i image.jpg -o output.jpg
```

Test on multiple images (with image partition):
```bash
bash pipeline.sh
```
**Note**: 
- The order of the partitionning can be chosen in the `pipeline.sh` file by changing the `order` variable. 
- The model tested can also be chosen by changing the `model` variable.
- The name of the prediction file can be chosen through the `predict_output` variable (The result can be found in the `predictions` directory).

## Make submission

Run the `make_submission.py` script (right now the name of the prediction and resulting file is defined in the script should be given as arguments)

> The file containing the predictions should not include any numbers otherwise the resulting submission is going to be wrong.
