""" Train VGG on image data """
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vgg16 import VGG_16
from copdgene_data_generator import get_image_set_size, batch_generator
from utility import get_gpu_memory_usage, get_model_memory_usage

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('Tensorflow version: ' + tf.__version__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", required=True, metavar="CSV FILE", help="CSV file pointing to images" )
    parser.add_argument("--image_column", required=True, help="Column name for images")
    parser.add_argument("--label_column", required=True, help="Column name for labels")
    parser.add_argument("--test_ratio", help="Percentage for testing data. Default is 0.3 (30%)", type=float, default=0.3)
    parser.add_argument("--epochs", help="Number of epochs. Default is 15", type=int, default=15)
    parser.add_argument("--classes", help="Number of classes. If not specified, classes will be inferred from labels", type=int, default=None)
    parser.add_argument("--batch_size", help="Training batch size. Default is 8", type=int, default=8)
    parser.add_argument("--output", help="Specify file name for output. Default is 'model'", default='model')
    parser.add_argument("--auto_resize", help="Auto-resize to min height/width of image set", action="store_true")
    parser.add_argument("--auto_batch", help="Auto-detect max batch size. Selecting this will override any specified batch size", action="store_true")
    parser.add_argument('--index_first', help="Set images to depth as the first index", action="store_true")
    args = parser.parse_args()

    epochs = args.epochs
    classes = args.classes
    batch_size = args.batch_size
    output = args.output
    test_ratio = args.test_ratio
    auto_resize = args.auto_resize
    auto_batch = args.auto_batch
    index_first = args.index_first

    # Point to images
    image_list_file = args.data_csv
    image_column = args.image_column
    label_column = args.label_column

    # Pull the list of files
    train_df = pd.read_csv(image_list_file)
    images = train_df[image_column].to_list()
    labels = train_df[label_column].to_list()

    # Split test set
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_ratio, random_state=42)

    # FOR DEBUG REMOVE IT
    print(f"Train Shape: {len(train_images)}")
    print(f"Train Label Len: {len(train_labels)}")

    print(f"Test Shape: {len(test_images)}")
    print(f"Test Label Len: {len(test_labels)}")

        # Get total number of images in each set
    train_image_sizes, train_image_count, train_min_height, train_min_width = get_image_set_size(train_images, index_first=index_first)
    test_image_sizes, test_image_count, test_min_height, test_min_width = get_image_set_size(test_images, index_first=index_first)

    # FOR DEBUG REMOVE IT
    print(f"train_image_sizes: {train_image_sizes}")
    print(f"train_image_count: {train_image_count}")
    print(f"train_min_height: {train_min_height}")
    print(f"train_min_width: {train_min_width}")

    print(f"test_image_sizes: {test_image_sizes}")
    print(f"test_image_count: {test_image_count}")
    print(f"test_min_height: {test_min_height}")
    print(f"test_min_width: {test_min_width}")

    # Set input image shape
    if auto_resize:
        min_height = min([train_min_height, test_min_height])
        min_width = min([train_min_width, test_min_width])
        input_shape = (min_height, min_width, 1) # (height, width, channels)
    else:
        input_shape = (512, 512, 1) # (height, width, channels)

    # Create a mirrored strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # # Build the model
    if classes is None:
        classes = len(np.unique(labels))
    classifier_activation = 'sigmoid'
    loss_type = 'binary_crossentropy'
    lst_metrics = ['categorical_accuracy']
    lr_rate = 0.01

    with strategy.scope():
        model = VGG_16(input_shape=input_shape, classes=classes, classifier_activation=classifier_activation)
        opt = tf.keras.optimizers.SGD(learning_rate=lr_rate, momentum=0.9)
        model.compile(loss=loss_type, optimizer=opt, metrics=lst_metrics)

    # Print Model Summary
    print(model.summary())

    # Determine batch size if auto-batch enabled
    if auto_batch:
        gpu_free, gpu_used = get_gpu_memory_usage()
        model_mem = get_model_memory_usage(model)
        batch_size = gpu_used // model_mem
        print(f"GPU memory allocated: {gpu_used} bytes")
        print(f"Model size: {model_mem} bytes")
        print(f"Maximum batch size: {batch_size} images")

    # Initialize settings for training
    train_steps = train_image_count // batch_size
    val_steps = test_image_count // batch_size

    # FOR DEBUG REMOVE IT
    print(f"input_shape: {input_shape}")
    print(f"train_steps: {train_steps}")
    print(f"val_steps: {val_steps}")

    # # Create the data generators
    train_gen = batch_generator(train_images, train_labels, batch_size, input_shape)
    test_gen = batch_generator(test_images, test_labels, batch_size, input_shape)

    # Train the model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(output+'.h5', monitor='categorical_accuracy', verbose=1, save_best_only=True)
    H = model.fit(
        x=train_gen,
        steps_per_epoch=train_steps,
        validation_data=test_gen,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[model_checkpoint])

    # Save loss history
    loss_history = np.array(H.history['loss'])
    np.savetxt(output+'_loss.csv', loss_history, delimiter=",")

if __name__ == '__main__':
    main()
    