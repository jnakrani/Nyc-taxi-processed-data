import boto3
import pandas as pd
import numpy as np
from config import *
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sagemaker
from sagemaker import get_execution_role
import tempfile
import gc
import json
import io

# Set up logging with basic configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_s3_paths():
    """Verify S3 paths and bucket existence."""
    s3 = boto3.client('s3')
    
    # Check if bucket exists
    try:
        s3.head_bucket(Bucket=PROCESSED_DATA_BUCKET)
        logger.info(f"Bucket {PROCESSED_DATA_BUCKET} exists")
    except s3.exceptions.ClientError as e:
        logger.error(f"Bucket {PROCESSED_DATA_BUCKET} does not exist or is not accessible: {str(e)}")
        raise
    
    # Check training data path
    training_files = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=PROCESSED_DATA_BUCKET, Prefix=TRAINING_DATA_PATH):
            if 'Contents' in page:
                training_files.extend([obj['Key'] for obj in page['Contents']])
    except Exception as e:
        logger.error(f"Error listing training files: {str(e)}")
        raise
    
    if not training_files:
        logger.error(f"No training files found in {TRAINING_DATA_PATH}")
        raise ValueError("No training data available")
    
    logger.info(f"Found {len(training_files)} training files")
    
    # Check validation data path
    validation_files = []
    try:
        for page in paginator.paginate(Bucket=PROCESSED_DATA_BUCKET, Prefix=VALIDATION_DATA_PATH):
            if 'Contents' in page:
                validation_files.extend([obj['Key'] for obj in page['Contents']])
    except Exception as e:
        logger.error(f"Error listing validation files: {str(e)}")
        raise
    
    if not validation_files:
        logger.error(f"No validation files found in {VALIDATION_DATA_PATH}")
        raise ValueError("No validation data available")
    
    logger.info(f"Found {len(validation_files)} validation files")
    
    return training_files, validation_files

def load_processed_data():
    """Load processed training and validation data from S3."""
    logger.info("Loading processed data from S3...")
    s3 = boto3.client('s3')
    
    # Verify S3 paths first
    training_files, validation_files = verify_s3_paths()
    
    # Load training data
    train_chunks = []
    for file_key in training_files:
        try:
            logger.info(f"Loading training chunk: {file_key}")
            response = s3.get_object(Bucket=PROCESSED_DATA_BUCKET, Key=file_key)
            
            # Read parquet file from memory using BytesIO
            buffer = io.BytesIO(response['Body'].read())
            chunk = pd.read_parquet(buffer)
            
            if chunk is not None and not chunk.empty:
                train_chunks.append(chunk)
                logger.info(f"Successfully loaded training chunk: {file_key} with shape {chunk.shape}")
            else:
                logger.warning(f"Empty chunk loaded from {file_key}")
        except Exception as e:
            logger.error(f"Error loading training chunk {file_key}: {str(e)}")
            continue
    
    if not train_chunks:
        raise ValueError("No training data was successfully loaded")
    
    # Combine training chunks
    train_df = pd.concat(train_chunks, ignore_index=True)
    del train_chunks
    gc.collect()
    
    logger.info(f"Combined training data shape: {train_df.shape}")
    
    # Load validation data
    val_chunks = []
    for file_key in validation_files:
        try:
            logger.info(f"Loading validation chunk: {file_key}")
            response = s3.get_object(Bucket=PROCESSED_DATA_BUCKET, Key=file_key)
            
            # Read parquet file from memory using BytesIO
            buffer = io.BytesIO(response['Body'].read())
            chunk = pd.read_parquet(buffer)
            
            if chunk is not None and not chunk.empty:
                val_chunks.append(chunk)
                logger.info(f"Successfully loaded validation chunk: {file_key} with shape {chunk.shape}")
            else:
                logger.warning(f"Empty chunk loaded from {file_key}")
        except Exception as e:
            logger.error(f"Error loading validation chunk {file_key}: {str(e)}")
            continue
    
    if not val_chunks:
        raise ValueError("No validation data was successfully loaded")
    
    # Combine validation chunks
    val_df = pd.concat(val_chunks, ignore_index=True)
    del val_chunks
    gc.collect()
    
    logger.info(f"Combined validation data shape: {val_df.shape}")
    
    return train_df, val_df

def prepare_features(train_df, val_df):
    """Prepare features for model training."""
    logger.info("Preparing features for model training...")
    
    # Print actual columns to help with debugging
    logger.info(f"Available columns in data: {train_df.columns.tolist()}")
    
    # Define feature columns based on actual data columns
    # Using the actual column names from your dataset
    if 'vendor_id' in train_df.columns:
        # Parse column names from the dataframe
        numeric_columns = train_df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove target column if it's in the list
        if 'fare_amount' in numeric_columns:
            numeric_columns.remove('fare_amount')
        if 'total_amount' in numeric_columns:
            numeric_columns.remove('total_amount')
            
        feature_columns = numeric_columns
        
        # Add time-based features if they exist
        if 'pickup_at' in train_df.columns and 'pickup_hour' not in train_df.columns:
            logger.info("Creating time-based features from pickup_at")
            # Extract time features
            train_df['pickup_hour'] = train_df['pickup_at'].dt.hour
            train_df['pickup_day'] = train_df['pickup_at'].dt.day
            train_df['pickup_month'] = train_df['pickup_at'].dt.month
            train_df['pickup_dayofweek'] = train_df['pickup_at'].dt.dayofweek
            
            # Do the same for validation data
            val_df['pickup_hour'] = val_df['pickup_at'].dt.hour
            val_df['pickup_day'] = val_df['pickup_at'].dt.day
            val_df['pickup_month'] = val_df['pickup_at'].dt.month
            val_df['pickup_dayofweek'] = val_df['pickup_at'].dt.dayofweek
            
            # Add these new features
            feature_columns.extend(['pickup_hour', 'pickup_day', 'pickup_month', 'pickup_dayofweek'])
        
        # Calculate trip duration if not already present
        if 'trip_duration' not in train_df.columns and 'pickup_at' in train_df.columns and 'dropoff_at' in train_df.columns:
            logger.info("Calculating trip duration")
            train_df['trip_duration'] = (train_df['dropoff_at'] - train_df['pickup_at']).dt.total_seconds() / 60
            val_df['trip_duration'] = (val_df['dropoff_at'] - val_df['pickup_at']).dt.total_seconds() / 60
            feature_columns.append('trip_duration')
    else:
        # Fallback to expected feature columns (for backwards compatibility)
        feature_columns = [
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude',
            'pickup_hour', 'pickup_day', 'pickup_month', 'pickup_dayofweek',
            'trip_duration'
        ]
    
    target_column = 'fare_amount'
    
    # Check if all required columns exist
    missing_columns = [col for col in feature_columns if col not in train_df.columns]
    if missing_columns:
        logger.warning(f"Missing columns in data: {missing_columns}")
        feature_columns = [col for col in feature_columns if col in train_df.columns]
        logger.info(f"Using available feature columns: {feature_columns}")
    
    if not feature_columns:
        raise ValueError("No usable feature columns found in the data")
    
    # Prepare feature matrices
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_val = val_df[feature_columns]
    y_val = val_df[target_column]
    
    # Log feature importance
    logger.info("\nFeature Statistics:")
    for col in feature_columns:
        logger.info(f"{col}:")
        logger.info(f"  Mean: {X_train[col].mean():.4f}")
        logger.info(f"  Std: {X_train[col].std():.4f}")
        logger.info(f"  Min: {X_train[col].min():.4f}")
        logger.info(f"  Max: {X_train[col].max():.4f}")
    
    return X_train, y_train, X_val, y_val, feature_columns, target_column

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with early stopping."""
    logger.info("Training XGBoost model...")
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set up parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'tree_method': 'hist'  # Use histogram-based algorithm for faster training
    }
    
    # Train model with early stopping
    num_rounds = 1000
    early_stopping_rounds = 50
    
    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10
    )
    
    return model

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluate model performance."""
    logger.info("Evaluating model performance...")
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train)
    dval = xgb.DMatrix(X_val)
    
    # Make predictions
    y_train_pred = model.predict(dtrain)
    y_val_pred = model.predict(dval)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    # Log metrics
    logger.info("\nModel Performance Metrics:")
    logger.info(f"Training RMSE: ${train_rmse:.2f}")
    logger.info(f"Validation RMSE: ${val_rmse:.2f}")
    logger.info(f"Training R2 Score: {train_r2:.4f}")
    logger.info(f"Validation R2 Score: {val_r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual Fare')
    plt.ylabel('Predicted Fare')
    plt.title('Actual vs Predicted Fare')
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    # Plot residuals
    plt.figure(figsize=(12, 6))
    residuals = y_val - y_val_pred
    plt.scatter(y_val_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Fare')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('residuals.png')
    plt.close()
    
    return {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2
    }

def save_model(model, feature_columns):
    """Save model and feature information."""
    logger.info("Saving model and feature information...")
    
    # Create temporary directory for model files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_path = os.path.join(temp_dir, 'model.json')
        model.save_model(model_path)
        
        # Save feature information
        feature_info = {
            'feature_columns': feature_columns,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'xgboost'
        }
        feature_path = os.path.join(temp_dir, 'feature_info.json')
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f)
        
        # Upload to S3
        s3 = boto3.client('s3')
        s3.upload_file(model_path, PROCESSED_DATA_BUCKET, f"{MODEL_OUTPUT_PATH}model.json")
        s3.upload_file(feature_path, PROCESSED_DATA_BUCKET, f"{MODEL_OUTPUT_PATH}feature_info.json")
    
    logger.info("Model and feature information saved to S3")

def main():
    """Main model training function."""
    start_time = datetime.now()
    logger.info("Starting model training...")
    
    try:
        # Load processed data
        train_df, val_df = load_processed_data()
        
        # Prepare features
        X_train, y_train, X_val, y_val, feature_columns, target_column = prepare_features(train_df, val_df)
        
        # Train model
        model = train_xgboost_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
        
        # Save model
        save_model(model, feature_columns)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Model training completed successfully! Duration: {duration}")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 