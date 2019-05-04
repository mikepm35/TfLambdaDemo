try:
  import unzip_requirements
except ImportError:
  pass

import os
import json
import time
import functools
import tarfile

import boto3
import tensorflow as tf

import census_data

FILE_DIR = '/tmp/'
BUCKET = os.environ['BUCKET']
# FILE_DIR = './'
# BUCKET = 'tflambdademo'


def trainHandler(event, context):
    time_start = time.time()

    body = json.loads(event.get('body'))

    # Read in epoch
    epoch_files = body['epoch']

    # Download files from S3
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            os.path.join(epoch_files,census_data.TRAINING_FILE),
            FILE_DIR+census_data.TRAINING_FILE)

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            os.path.join(epoch_files,census_data.EVAL_FILE),
            FILE_DIR+census_data.EVAL_FILE)

    # Create feature columns
    wide_cols, deep_cols = census_data.build_model_columns()

    # Setup estimator
    classifier = tf.estimator.LinearClassifier(
                        feature_columns=wide_cols,
                        model_dir=FILE_DIR+'model_'+epoch_files+'/')

    # Create callable input function and execute train
    train_inpf = functools.partial(
                    census_data.input_fn,
                    FILE_DIR+census_data.TRAINING_FILE,
                    num_epochs=2, shuffle=True,
                    batch_size=64)

    classifier.train(train_inpf)

    # Create callable input function and execute evaluation
    test_inpf = functools.partial(
                    census_data.input_fn,
                    FILE_DIR+census_data.EVAL_FILE,
                    num_epochs=1, shuffle=False,
                    batch_size=64)

    result = classifier.evaluate(test_inpf)
    print('Evaluation result: %s' % result)

    # Zip up model files and store in s3
    with tarfile.open(FILE_DIR+'model.tar.gz', mode='w:gz') as arch:
        arch.add(FILE_DIR+'model_'+epoch_files+'/', recursive=True)

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).Object(os.path.join(epoch_files,'model.tar.gz')
        ).upload_file(FILE_DIR+'model.tar.gz')


    # Convert result from float32 for json serialization
    for key in result:
        result[key] = result[key].item()

    response = {
        "statusCode": 200,
        "body": json.dumps({'epoch': epoch_files,
                            'runtime': round(time.time()-time_start, 1),
                            'result': result})
    }

    return response
