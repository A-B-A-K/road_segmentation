#  CS-433: Road Segmentation

## Setup

The training and testing data we use is located in the `data` directory. 

The training directory has to have the following structure:
```bash
/training_set
    /ground_truth
    /images
```

>**Warning!** If your ground truths are not binary, you need to binarize them. To do so, use the following command:
```bash 
python data/binarize_dir.py
```
>_Note that the directory to be binarized can be adjusted in the script._




The test directory (in our case `test_set_images`) has to have the following structure:
```bash
/test_set
    /test_i
        test_i.png
```



There is the possibility to split the data into training and validation, augment the data and generate variants of the original dataset, namely hue, contrast, saturation and brightness (used in our pipeline to refine our predictions through ensemble learning).

### Split data
```bash
python data/split_data.py --ratio 0.9
```
_Note that the names of the input and output directories can be adjusted through the script, and that the split ratio has a default value of 0.9._

### Data augmentation:
```bash
python data/augment_data.py
```
_Note that the names of the input and output directories can be adjusted through the script_

### Transform data
```bash
python data/transform_data.py
```
_Note that the names of the input and output directories can be adjusted through the script_

## Training

In oreder to train a single model the following command can be run:
```bash
python train.py 
```
Interesting flags:
- `--epochs`: Number of epochs
- `--batch-size`: Batch size
- `--learning-rate`: Learning rate
- `--leaky`: Use leaky ReLU instead of normal ReLU
- `--load`: Load model from a .pth file
- `--amp`: Use mixed precision

The recommended setup to use is the following:
```bash
python train.py --amp --epochs 150 --batch-size 32 --learning-rate 1e-5 
```
The paths for the images and groundtruths can be defined in the top of the training file (a preset structure to enhance user experience has been setup and can be commented out).

An alternative that can be used in a similar way, that runs a set of transformed datasets in a cascaded way is:
```bash
python train_loop.py --amp --epochs 150 --batch-size 32 --learning-rate 1e-5 
```

The resulting weights will be stored in the checkpoints folder. Ideally the weights of intereset should be moved to a new weights folder.

## Testing

After training the model and saving it to `MODEL.pth`, it can be tested through the following CLI commands:

Test single image (no image partition):
```bash
python predict.py -i image.jpg -o output.jpg
```

To evaluate the performance of the validation set the following bash file should be run:
```bash
bash pipeline_val.sh
```
Similarly to create predictions on the test set (without evaluating since we have no groundtruths for each image):
```bash
bash pipeline_test.sh
```
**Note that there are a few parameters that should be set before running this file**:
-  Multiple models can be evaluated at the same time, they are defined in the models structure where their name has to be defined in the square brackets and the path to their respective weights needs to be specified after the equal sign.
- The name of the folder hosting the validation set
- The threshold for removing small clusters through DBSCAN.
- The up-scaling and down-scaling factors (ideally the one should be inversely proportional to the other).

### Ensemble Learning

To create predictions using weighted averages one can use the following bash file:
```bash
bash ensemble_learning.sh 
```
**Note that there are a few parameters that should be set before running this file**:
- The variable that specifies whether we are looking at the validation (val) or ther test (test) set
    - In case of applying on validation specify the name of the validation folder.
- The models that need to be averaged and their respective weights (no need for the weights to sum up to 1).
- The binarization threshold.
- The name of the file that is going to host the results.

## Make submission

To create a submission file the following command should be executed:
```bash
python utils/make_submission.py ./predictions/test/ submission.csv
```

> **Warning!** The file containing the predictions should not include any numbers otherwise the resulting submission is going to be wrong.
