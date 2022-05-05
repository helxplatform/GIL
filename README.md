# GIL - Generalized Image Learning 

## Purpose
This repo houses the initial scripts for building a deep learning app on BioData Catalyst powered by Seven Bridges. These scripts will be used to test training scalability, among other issues.

## Main script
`train.py` creates a VGG-16 model for single-channel image classification.
### Input arguments:
| Arg | Description | Type | Default | Required |
| --- | ----------- | ---- | ------ | -------- |
| --data_csv | Path to CSV file pointing to images/labels | string |  | YES |
| --image_column | Column name for images | string |  | YES |
| --label_column | Column name for labels | string |  | YES |
| --test_ratio | Percentage for testing data | float | 0.3 |   |
| --epochs | Number of training epochs | int | 15 |   |
| --classes | Number of classes. If not specified, classes will be inferred from labels | int | None |   |
| --batch_size | Training batch size | int | 8 |   |
| --output | Specify file name for output | string | 'model' |   |
| --auto_resize | Auto-resize to min height/width of image set | store_true |   |   |
| --auto_batch | Auto-detect max batch size. Selecting this will override any specified batch size | store_true |   |   |
| --index_first | Set images to depth as the first index (uncommon) | store_true |   |   |

## Debug
`get_sizes.py --data_csv /path/to/file.csv --image_column image_path_column_name` will create a CSV containing the image name, SimpleITK image shape, and Numpy array shape. It will also print this information to the console.
