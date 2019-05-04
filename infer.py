try:
  import unzip_requirements
except ImportError:
  pass

import json
import os
import tarfile

import boto3
import tensorflow as tf
import numpy as np

import census_data

FILE_DIR = '/tmp/'
BUCKET = os.environ['BUCKET']


def _easy_input_function(data_dict, batch_size=64):
    """
    data_dict = {
        '<csv_col_1>': ['<first_pred_value>', '<second_pred_value>']
        '<csv_col_2>': ['<first_pred_value>', '<second_pred_value>']
        ...
    }
    """

    # Convert input data to numpy arrays
    for col in data_dict:
        col_ind = census_data._CSV_COLUMNS.index(col)
        dtype = type(census_data._CSV_COLUMN_DEFAULTS[col_ind][0])
        data_dict[col] = np.array(data_dict[col],
                                        dtype=dtype)

    labels = data_dict.pop('income_bracket')

    ds = tf.data.Dataset.from_tensor_slices((data_dict, labels))
    ds = ds.batch(64)

    return ds


def inferHandler(event, context):
    body = json.loads(event.get('body'))

    # Read in prediction data as dictionary
    # Keys should match _CSV_COLUMNS, values should be lists
    predict_input = body['input']

    # Read in epoch
    epoch_files = body['epoch']

    # Download model from S3 and extract
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            os.path.join(epoch_files,'model.tar.gz'),
            FILE_DIR+'model.tar.gz')

    tarfile.open(FILE_DIR+'model.tar.gz', 'r').extractall(FILE_DIR)

    # Create feature columns
    wide_cols, deep_cols = census_data.build_model_columns()

    # Load model
    classifier = tf.estimator.LinearClassifier(
                    feature_columns=wide_cols,
                    model_dir=FILE_DIR+'tmp/model_'+epoch_files+'/',
                    warm_start_from=FILE_DIR+'tmp/model_'+epoch_files+'/')

    # Setup prediction
    predict_iter = classifier.predict(
                        lambda:_easy_input_function(predict_input))

    # Iterate over prediction and convert to lists
    predictions = []
    for prediction in predict_iter:
        for key in prediction:
            prediction[key] = prediction[key].tolist()

        predictions.append(prediction)

    response = {
        "statusCode": 200,
        "body": json.dumps(predictions,
                            default=lambda x: x.decode('utf-8'))
    }

    return response
