try:
  import unzip_requirements
except ImportError:
  pass

import os
import json
import time

import boto3
import tensorflow as tf

import census_data

FILE_DIR = '/tmp/'
BUCKET = os.environ['BUCKET']


def uploadHandler(event, context):
    # Download data to local tmp directory
    census_data.download(FILE_DIR)

    # Upload files to S3
    epoch_now = str(int(time.time()))

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).Object(os.path.join(epoch_now,census_data.TRAINING_FILE)
        ).upload_file(FILE_DIR+census_data.TRAINING_FILE)

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).Object(os.path.join(epoch_now,census_data.EVAL_FILE)
        ).upload_file(FILE_DIR+census_data.EVAL_FILE)

    response = {
        "statusCode": 200,
        "body": json.dumps({'epoch': epoch_now})
    }

    return response
