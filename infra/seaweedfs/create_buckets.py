"""BUCKETS 환경 변수(쉼표 구분)의 버킷을 만든다. 이미 있으면 건너뛴다."""

import logging
import os
import sys

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("create_buckets")

EXISTING_BUCKET_ERRORS = {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}


def main() -> int:
    bucket_names = [name.strip() for name in os.environ["BUCKETS"].split(",") if name.strip()]
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1",
    )
    for name in bucket_names:
        try:
            s3.create_bucket(Bucket=name)
            logger.info("created bucket %s", name)
        except ClientError as error:
            if error.response["Error"]["Code"] not in EXISTING_BUCKET_ERRORS:
                logger.error("failed to create bucket %s: %s", name, error)
                return 1
            logger.info("bucket %s already exists", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
