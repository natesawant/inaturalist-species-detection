import argparse
import time

from efficient_pipeline import EfficientNetPipeline
from inaturalist import FISH_CLASSES
from cnn_pipeline import CNNPipeline
from vit_pipeline import ViTPipeline

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

    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (cnn, efficient, vit)",
        default="cnn",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args.download, args.classes, args.image_size, args.batch_size, args.learning_rate, args.num_epochs, args.top_k, args.job_id, args.model


if __name__ == "__main__":
    download, classes, image_size, batch_size, learning_rate, num_epochs, top_k, job_id, model = parse_args()

    models = {"cnn": CNNPipeline,
              "efficient": EfficientNetPipeline,
              "vit": ViTPipeline}
    
    model_pipeline = models.get(model)

    if not model_pipeline:
        print("Invalid model option")
        exit(1)

    pipeline = model_pipeline(
                image_size=image_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                top_k=top_k,
                model_type=model
            )

    start = time.time()

    pipeline.start_pipeline(job_id, download=download, classes=classes)

    time_elapsed = time.time() - start

    time_formatted = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))

    print(f"Total time elapsed: {time_formatted}")