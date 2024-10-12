import os
import boto3
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError
from progress.bar import Bar

# Initialize a session using Amazon S3
s3 = boto3.client("s3")


def upload_files(root_dir, folder, bucket_name):
    try:
        # Iterate through all files in the directory
        for i, info in enumerate(os.walk(root_dir / folder), start=1):
            root, dirs, files = info
            with Bar(f"{i}: Uploading {root}", max=len(files)) as bar:
                for file in files:
                    # Full file path
                    file_path = os.path.join(root, file)

                    # The relative path inside the bucket
                    s3_key = os.path.relpath(file_path, root_dir)

                    try:
                        s3.head_object(Bucket=bucket_name, Key=s3_key)
                    except ClientError as e:
                        # Upload the file
                        s3.upload_file(file_path, bucket_name, s3_key)

                    bar.next()

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except NoCredentialsError:
        print("Credentials not available")


def upload_file(root_dir, file_path, bucket_name):
    s3_key = os.path.relpath(file_path, root_dir)
    s3.upload_file(file_path, bucket_name, s3_key)


# Example usage
data_dir = Path("data")
bucket_name = "inaturalist-marine-only-dataset"

if __name__ == "__main__":
    upload_file(data_dir, data_dir / "train.json", bucket_name)
    upload_file(data_dir, data_dir / "val.json", bucket_name)

    upload_files(data_dir, data_dir / "train", bucket_name)
    upload_files(data_dir, data_dir / "val", bucket_name)
