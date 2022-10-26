""" Train VGG on image data """
import argparse
import tensorflow as tf
SM_FRAMEWORK=tf.keras
import segmentation_models as sm
from src.utility import get_max_batch_size

def main():
    """ Parse arguments and pull selected Keras application """
    # Available Keras application models
    model_dict = {
        "unet": sm.Unet,
        "fpn": sm.FPN,
        "linknet": sm.Linknet,
        "pspnet": sm.PSPNet
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_height", help="Height of images (pixels)", required=True, type=int, default=512)
    parser.add_argument("--image_width", help="Width of images (pixels)", required=True, type=int, default=512)
    parser.add_argument("--image_channels", help="Number of channels", required=True, type=int, default=1)
    parser.add_argument("--classes", help="Number of classes", type=int, default=2)
    parser.add_argument("--backbone", help="Backbone architecture", type=str, default="resnet50")
    parser.add_argument("--cross_dev_ops", help="Cross device operation to use for multi-GPU reduction. 'all' = NcclAllReduce, 'hierarchical' = HierarchicalCopyAllReduce, 'one' = ReductionToOneDevice", type=str, choices=["all", "hierarchical", "one"], default="hierarchical")
    ARGS = parser.parse_args()

    LOG = open("segment_model_sizes.txt", "w")
    LOG.write(f"Tensorflow version: {tf.__version__}\n")

    # Create a mirrored strategy
    cdo_dict = {
        "all": tf.distribute.NcclAllReduce(),
        "hierarchical": tf.distribute.HierarchicalCopyAllReduce(),
        "one": tf.distribute.ReductionToOneDevice(reduce_to_device="/gpu:0")
    }
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=cdo_dict[ARGS.cross_dev_ops])

    # Build the model
    classifier_activation = 'sigmoid'
    loss_type = sm.losses.bce_jaccard_loss
    lst_metrics = ['sparse_categorical_accuracy']
    lr_rate = 0.01
    input_shape = (ARGS.image_height, ARGS.image_width, ARGS.image_channels)

    for arch, base_model in model_dict.items():
        LOG.write(f"Model: {arch}\n")
        with strategy.scope():
            #model = build_image_classifier(
            #    base_model=base_model,
            #    classes=ARGS.classes,
            #    input_shape=input_shape,
            #    classifier_activation=classifier_activation,
            #    dropout=0.1)

            model = base_model(
                backbone_name=ARGS.backbone,
                encoder_weights=None,
                classes=classes,
                input_shape=input_shape,
                activation=classifier_activation)

            opt = tf.keras.optimizers.Adam(learning_rate=lr_rate)

            model.compile(
                loss=loss_type,
                optimizer=opt,
                metrics=lst_metrics)

        # Determine batch size if auto-batch enabled
        # Auto-batch will not run if no GPU present
        _ = get_max_batch_size(model, unit="mebi", log=LOG)

    LOG.close()

if __name__ == '__main__':
    main()
