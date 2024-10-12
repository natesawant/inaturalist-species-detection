from pathlib import Path
import boto3
import os
from progress.spinner import Spinner

s3 = boto3.resource(
    "s3"
)  # assumes credentials & configuration are handled outside python in .aws directory or environment variables


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    s3_folder = str(s3_folder)
    bucket = s3.Bucket(bucket_name)
    with Spinner(f"Downloading from {bucket_name}/{s3_folder} ") as spinner:
        for obj in bucket.objects.filter(Prefix=s3_folder):
            target = (
                obj.key
                if local_dir is None
                else os.path.join(
                    local_dir / s3_folder, os.path.relpath(obj.key, s3_folder)
                )
            )
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == "/":
                continue
            bucket.download_file(obj.key, target)
            spinner.next()


def download_s3_object(bucket_name, s3_key, local_dir=None):
    bucket = s3.Bucket(bucket_name)
    target = (
        s3_key
        if local_dir is None
        else os.path.join(local_dir, os.path.relpath(s3_key))
    )
    if not os.path.exists(target):
        bucket.download_file(s3_key, target)


data_dir = Path("data")

train_dir = "train"
val_dir = "val"
bucket_name = "inaturalist-marine-only-dataset"

if __name__ == "__main__":
    download_s3_folder(bucket_name=bucket_name, s3_folder="train", local_dir=data_dir)
