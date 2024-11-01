import argparse

from inaturalist import FISH_CLASSES
from cnn_pipeline import CNNPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Boolean argument for "download"
    parser.add_argument(
        "--download", action="store_true", help="Flag to initiate download"
    )

    # String list argument for "classes"
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        help="List of class names",
        default=FISH_CLASSES,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        help="Dimension of image",
        default=128,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of items per batch",
        default=64,
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of epochs/iterations",
        default=20,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="Value of k to choose top chooses",
        default=5,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (alpha)",
        default=0.001,
    )

    parser.add_argument(
        "--job-id",
        type=str,
        help="ID of the job",
        default="latest",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args.download, args.classes, args.image_size, args.batch_size, args.learning_rate, args.num_epochs, args.top_k, args.job_id


if __name__ == "__main__":
    download, classes, image_size, batch_size, learning_rate, num_epochs, top_k, job_id = parse_args()

    pipeline = CNNPipeline(
        image_size=image_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        top_k=top_k,
    )

    pipeline.start_pipeline(job_id, download=download, classes=classes)