import os

# AWS Configuration
AWS_REGION = 'us-east-1'
AWS_PROFILE = 'default'

# S3 Configuration
S3_BUCKET_PREFIX = 'nyc-taxi-dataset-public'
RIDE_INFO_PATH = 'nyc-taxi-orig-cleaned-split-parquet-per-year-multiple-files/ride-info/'
RIDE_FARE_PATH = 'nyc-taxi-orig-cleaned-split-parquet-per-year-multiple-files/ride-fare/'

# SageMaker Configuration
SAGEMAKER_ROLE = 'SageMakerRole'  # This will be created during setup
SAGEMAKER_INSTANCE_TYPE = 'ml.t3.medium'
SAGEMAKER_INSTANCE_COUNT = 1
SAGEMAKER_VOLUME_SIZE = 5  # GB

# Model Configuration
TRAIN_TEST_SPLIT_RATIO = 0.7
RANDOM_STATE = 42

# Output Configuration
PROCESSED_DATA_BUCKET = 'nyc-taxi-processed-data'  # This will be created during setup
TRAINING_DATA_PATH = 'processed/training/'
VALIDATION_DATA_PATH = 'processed/validation/'
MODEL_OUTPUT_PATH = 'models/' 