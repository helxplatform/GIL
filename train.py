""" Train VGG on image data """
import os
from datetime import datetime
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models import build_image_classifier
from src.utility import get_max_batch_size
from src.data_generator import ImageSet

def model_config():
    """ Parse arguments and pull selected Keras application """
    # Available Keras application models
    model_dict = {
        "densenet121": tf.keras.applications.DenseNet121,
        "densenet169": tf.keras.applications.DenseNet169,
        "densenet201": tf.keras.applications.DenseNet201,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
        "efficientnetb1": tf.keras.applications.EfficientNetB1,
        "efficientnetb2": tf.keras.applications.EfficientNetB2,
        "efficientnetb3": tf.keras.applications.EfficientNetB3,
        "efficientnetb4": tf.keras.applications.EfficientNetB4,
        "efficientnetb5": tf.keras.applications.EfficientNetB5,
        "efficientnetb6": tf.keras.applications.EfficientNetB6,
        "efficientnetb7": tf.keras.applications.EfficientNetB7,
        "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
        "inceptionv3": tf.keras.applications.InceptionV3,
        "mobilenet": tf.keras.applications.MobileNet,
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "nasnetlarge": tf.keras.applications.NASNetLarge,
        "nasnetmobile": tf.keras.applications.NASNetMobile,
        "resnet101": tf.keras.applications.ResNet101,
        "resnet101v2": tf.keras.applications.ResNet101V2,
        "resnet152": tf.keras.applications.ResNet152,
        "resnet152v2": tf.keras.applications.ResNet152V2,
        "resnet50": tf.keras.applications.ResNet50,
        "resnet50v2": tf.keras.applications.ResNet50V2,
        "vgg16": tf.keras.applications.VGG16,
        "vgg19": tf.keras.applications.VGG19,
        "xception": tf.keras.applications.Xception
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing images")
    parser.add_argument("--data_csv", required=True, metavar="CSV FILE", help="CSV file pointing to images")
    parser.add_argument("--image_column", required=True, help="Column name for images")
    parser.add_argument("--label_column", required=True, help="Column name for labels")
    parser.add_argument("--arch", required=True, choices=model_dict.keys(), help="Model architecture. Supports most Keras applications.")
    parser.add_argument("--test_ratio", help="Percentage for testing data. Default is 0.3 (30%)", type=float, default=0.3)
    parser.add_argument("--epochs", help="Number of epochs. Default is 15", type=int, default=15)
    parser.add_argument("--classes", help="Number of classes. If not specified, classes will be inferred from labels", type=int, default=None)
    parser.add_argument("--batch_size", help="Training batch size. Default is 8", type=int, default=8)
    parser.add_argument("--output", help="Specify file name for output. Default is 'model'", default='model')
    parser.add_argument("--auto_resize", help="Auto-resize to min height/width of image set", action="store_true")
    parser.add_argument("--auto_batch", help="Auto-detect max batch size. Selecting this will override any specified batch size", action="store_true")
    parser.add_argument('--index_first', help="Set images to depth as the first index (uncommon)", action="store_true")
    parser.add_argument("--run_eagerly", help="Run eagerly (for debug). Will lose performance.", action="store_true")
    ARGS = parser.parse_args()

    if ARGS.arch not in model_dict:
        model_arch = None
    else:
        model_arch = model_dict[ARGS.arch]

    return ARGS, model_arch

def split_and_resize(images, labels, classes, test_ratio, auto_resize, index_first):
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_ratio, random_state=42)

    training_set = ImageSet(train_images, train_labels, classes, index_first)
    testing_set = ImageSet(test_images, test_labels, classes, index_first)

    min_height = min([training_set.min_height, testing_set.min_height])
    min_width = min([training_set.min_width, testing_set.min_width])

    # Set input image shape
    if auto_resize:
        input_shape = (min_height, min_width, 1) # (height, width, channels)
    else:
        input_shape = (512, 512, 1)

    print(f"Training images: {training_set.count}")
    print(f"Validation images: {testing_set.count}")
    print(f"Min height: {min_height}")
    print(f"Min width: {min_width}")
    print(f"Input shape: {input_shape}")

    return training_set, testing_set, input_shape

def main():
    ini_time = datetime.now()
    print(f"Init time: {ini_time}")
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print('Tensorflow version: ' + tf.__version__)

    ARGS, base_model = model_config()
    if base_model is None:
        print(f"{ARGS.arch} not in Keras applications!")
        print(f"Use '--help' for list of supported options.")
        return

    # Pull the list of files
    train_df = pd.read_csv(ARGS.data_csv)
    images = [os.path.join(ARGS.data_dir, name) for name in train_df[ARGS.image_column].to_list()]
    labels = train_df[ARGS.label_column].to_list()

    if ARGS.classes is None:
        #classes = len(np.unique(labels))
        classes = max(labels) + 1
    else:
        classes = ARGS.classes

    # Split training/test sets
    # Returns ImageSet instances for each set and input shape
    training_set, testing_set, input_shape = split_and_resize(images, labels, classes, ARGS.test_ratio, ARGS.auto_resize, ARGS.index_first)

    # Create a mirrored strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # # Build the model
    classifier_activation = 'sigmoid'
    loss_type = 'categorical_crossentropy'
    lst_metrics = ['categorical_accuracy']
    lr_rate = 0.01

    with strategy.scope():
        model = build_image_classifier(
            base_model=base_model,
            classes=classes,
            input_shape=input_shape,
            classifier_activation=classifier_activation,
            dropout=0.1)

        opt = tf.keras.optimizers.SGD(learning_rate=lr_rate, momentum=0.9)

        model.compile(
            loss=loss_type,
            optimizer=opt,
            metrics=lst_metrics,
            run_eagerly=ARGS.run_eagerly)

    # Print Model Summary
    print(model.summary())

    # Determine batch size if auto-batch enabled
    # Auto-batch will not run if no GPU present
    if not ARGS.auto_batch or not tf.config.list_physical_devices('GPU'):
        batch_size = ARGS.batch_size
    else:
        batch_size = get_max_batch_size(model)

    # Initialize settings for training
    try:
        train_steps = training_set.count // batch_size
        val_steps = testing_set.count // batch_size
    except ZeroDivisionError:
        print("Returned batch size of 0. Setting batch size to 1")
        batch_size = 1

    # FOR DEBUG REMOVE IT
    print(f"input_shape: {input_shape}")
    print(f"batch_size: {batch_size}")
    print(f"train_steps: {train_steps}")
    print(f"val_steps: {val_steps}")

    # # Create the data generators
    #train_gen = batch_generator(train_images, train_labels, batch_size, input_shape)
    #test_gen = batch_generator(test_images, test_labels, batch_size, input_shape)

    train_start_time = datetime.now()
    print(f"Training start time: {train_start_time}")
    print(f"Elapsed: {train_start_time - ini_time}")
    # Train the model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './' + ARGS.output + '.h5',
        monitor='categorical_accuracy',
        verbose=1,
        save_best_only=True)

    H = model.fit(
        x=training_set.generate_batches(batch_size, input_shape),
        steps_per_epoch=train_steps,
        validation_data=testing_set.generate_batches(batch_size, input_shape),
        validation_steps=val_steps,
        epochs=ARGS.epochs,
        batch_size=batch_size,
        callbacks=[model_checkpoint])

    # Save loss history
    loss_history = np.array(H.history['loss'])
    np.savetxt('./' + ARGS.output + '_loss.csv', loss_history, delimiter=",")

    end_time = datetime.now()
    print(f"End time: {end_time}")
    print(f"Training time: {end_time - train_start_time}")
    print(f"Total elapsed: {end_time - ini_time}")

if __name__ == '__main__':
    main()
