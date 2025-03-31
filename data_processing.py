import os
import gc
import time
import boto3
import json
import psutil
import logging
import tempfile
import numpy as np
import pandas as pd
from config import *
import pyarrow as pa
import seaborn as sns
from tqdm import tqdm
import dask.dataframe as dd
import pyarrow.parquet as pq
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
from pyarrow.parquet import ParquetFile
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants for chunked processing
CHUNK_SIZE = 1000000  # Process 1 million rows at a time
MAX_WORKERS = min(20, os.cpu_count() * 3)  # Optimize based on CPU cores
BATCH_SIZE = 25  # Process 25 files at a time
MAX_BATCH_WORKERS = min(5, os.cpu_count())  # Workers per batch
CHECKPOINT_FILE = 'processing_checkpoint.json'
FILE_PROGRESS_FILE = 'file_progress.json'
TESTING_MODE = True  # Set to False when ready for full processing
TEST_FILE_COUNT = 10  # Number of files to process in testing mode
S3_BUCKET_PREFIX="nyc-taxi-dataset-public"
RIDE_INFO_PATH = 'nyc-taxi-orig-cleaned-split-parquet-per-year-multiple-files/ride-info/'
RIDE_FARE_PATH = 'nyc-taxi-orig-cleaned-split-parquet-per-year-multiple-files/ride-fare/'
TRAIN_TEST_SPLIT_RATIO = 0.7
RANDOM_STATE = 42

# Output Configuration
PROCESSED_DATA_BUCKET = 'nyc-taxi-processed-data'  # This will be created during setup
TRAINING_DATA_PATH = 'processed/training/'
VALIDATION_DATA_PATH = 'processed/validation/'
MODEL_OUTPUT_PATH = 'models/' 

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / 1024 / 1024 / 1024
    logger.info(f"Memory usage: {memory_usage_gb:.2f} GB")

def save_file_progress(filename, total_rows, processed_rows):
    """Save progress for a specific file."""
    if os.path.exists(FILE_PROGRESS_FILE):
        with open(FILE_PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
    else:
        progress = {}
    
    progress[filename] = {
        'total_rows': total_rows,
        'processed_rows': processed_rows,
        'percentage': (processed_rows / total_rows * 100) if total_rows > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(FILE_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def save_checkpoint(processed_files, data_type, total_files, total_processed):
    """Save checkpoint of processed files."""
    checkpoint = {
        'processed_files': processed_files,
        'data_type': data_type,
        'timestamp': datetime.now().isoformat(),
        'total_files': total_files,
        'total_processed': total_processed,
        'percentage_complete': (total_processed / total_files * 100) if total_files > 0 else 0
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def load_checkpoint():
    """Load checkpoint of processed files."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def read_parquet_chunks(s3_client, bucket, file_key):
    """Read parquet file in chunks using pyarrow."""
    start_time = time.time()
    try:
        # Get the file from S3
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        buffer = pa.BufferReader(response['Body'].read())
        
        # Open parquet file
        pf = ParquetFile(buffer)
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        
        # Process each row group
        processed_chunks = []
        processed_rows = 0
        
        for i in range(pf.num_row_groups):
            try:
                # Read row group
                chunk = pf.read_row_group(i).to_pandas()
                row_count = len(chunk)
                processed_rows += row_count
                
                # Convert timestamps
                if 'pickup_datetime' in chunk.columns:
                    chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'])
                if 'dropoff_datetime' in chunk.columns:
                    chunk['dropoff_datetime'] = pd.to_datetime(chunk['dropoff_datetime'])
                
                # Extract time-based features
                if 'pickup_datetime' in chunk.columns:
                    chunk['pickup_hour'] = chunk['pickup_datetime'].dt.hour
                    chunk['pickup_day'] = chunk['pickup_datetime'].dt.day
                    chunk['pickup_month'] = chunk['pickup_datetime'].dt.month
                    chunk['pickup_dayofweek'] = chunk['pickup_datetime'].dt.dayofweek
                
                # Calculate trip duration
                if 'pickup_datetime' in chunk.columns and 'dropoff_datetime' in chunk.columns:
                    chunk['trip_duration'] = (chunk['dropoff_datetime'] - chunk['pickup_datetime']).dt.total_seconds() / 60
                    
                processed_chunks.append(chunk)
                
                # Update progress
                save_file_progress(file_key, total_rows, processed_rows)
                
                # Log progress
                logger.info(f"Processed row group {i+1}/{pf.num_row_groups} in {file_key} ({row_count} rows)")
                
                # Clear memory after each row group
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing row group {i} in file {file_key}: {str(e)}")
                continue
        
        if processed_chunks:
            result = pd.concat(processed_chunks, ignore_index=True)
            
            # Clear memory after processing file
            del processed_chunks
            gc.collect()
            
            elapsed_time = time.time() - start_time
            rows_per_second = processed_rows / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"Completed file {file_key}: {processed_rows} rows in {elapsed_time:.2f} seconds ({rows_per_second:.2f} rows/sec)")
            
            return result, processed_rows
    except Exception as e:
        logger.error(f"Error reading file {file_key}: {str(e)}")
    return None, 0

def process_file(file_key, s3_client, bucket, data_type):
    """Process a single file."""
    try:
        df, rows = read_parquet_chunks(s3_client, bucket, file_key)
        if df is not None and not df.empty:
            # Save intermediate result
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
            df.to_parquet(temp_file.name)
            
            # Log success
            logger.info(f"Successfully processed {data_type} file: {file_key} ({rows} rows)")
            log_memory_usage()
            
            # Clean up
            del df
            gc.collect()
            
            return temp_file.name, rows
    except Exception as e:
        logger.error(f"Error processing file {file_key}: {str(e)}")
    
    return None, 0

def process_batch(batch_files, s3_client, bucket, data_type):
    """Process a batch of files using ThreadPoolExecutor."""
    start_time = time.time()
    temp_files = []
    processed_files = []
    total_rows = 0
    
    process_file_partial = partial(process_file, s3_client=s3_client, bucket=bucket, data_type=data_type)
    
    with ThreadPoolExecutor(max_workers=MAX_BATCH_WORKERS) as executor:
        future_to_file = {executor.submit(process_file_partial, file_key): file_key for file_key in batch_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(batch_files), desc=f"Processing {data_type} batch"):
            file_key = future_to_file[future]
            try:
                temp_file, rows = future.result()
                if temp_file:
                    temp_files.append(temp_file)
                    processed_files.append(file_key)
                    total_rows += rows
            except Exception as e:
                logger.error(f"Error processing file {file_key}: {str(e)}")
    
    elapsed_time = time.time() - start_time
    files_per_second = len(processed_files) / elapsed_time if elapsed_time > 0 else 0
    rows_per_second = total_rows / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Batch completed: {len(processed_files)}/{len(batch_files)} files processed in {elapsed_time:.2f} seconds")
    logger.info(f"Performance: {files_per_second:.2f} files/sec, {rows_per_second:.2f} rows/sec")
    
    return temp_files, processed_files, total_rows

def load_data_from_s3():
    """Load data from S3 bucket using parallel processing and chunks."""
    start_time = time.time()
    s3 = boto3.client('s3')
    
    # In testing mode, ignore the previous checkpoint and progress
    if TESTING_MODE:
        if os.path.exists(CHECKPOINT_FILE):
            logger.info("TESTING MODE: Resetting checkpoint")
            os.remove(CHECKPOINT_FILE)
        if os.path.exists(FILE_PROGRESS_FILE):
            logger.info("TESTING MODE: Resetting file progress")
            os.remove(FILE_PROGRESS_FILE)
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    processed_files = set(checkpoint['processed_files']) if checkpoint else set()
    data_type = checkpoint['data_type'] if checkpoint else 'ride_info'
    
    # List files in the ride info directory
    logger.info("Listing ride info files...")
    ride_info_files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET_PREFIX, Prefix=RIDE_INFO_PATH):
        if 'Contents' in page:
            ride_info_files.extend([obj['Key'] for obj in page['Contents']])
    
    # List files in the ride fare directory
    logger.info("Listing ride fare files...")
    ride_fare_files = []
    for page in paginator.paginate(Bucket=S3_BUCKET_PREFIX, Prefix=RIDE_FARE_PATH):
        if 'Contents' in page:
            ride_fare_files.extend([obj['Key'] for obj in page['Contents']])
    
    # Limit files for testing mode
    if TESTING_MODE:
        logger.info(f"TESTING MODE: Limiting to {TEST_FILE_COUNT} files of each type")
        ride_info_files = ride_info_files[:TEST_FILE_COUNT]
        ride_fare_files = ride_fare_files[:TEST_FILE_COUNT]
    
    ride_info_count = len(ride_info_files)
    ride_fare_count = len(ride_fare_files)
    total_files = ride_info_count + ride_fare_count
    
    logger.info(f"Processing {ride_info_count} ride info files and {ride_fare_count} ride fare files")
    
    # Keep track of totals
    total_processed = len(processed_files)
    total_ride_info_rows = 0
    total_ride_fare_rows = 0
    
    # Process ride info files in batches
    ride_info_temp_files = []
    if data_type == 'ride_info' or data_type == 'both':
        logger.info("Processing ride info files...")
        
        # Create batches of unprocessed files
        unprocessed_ride_info = [f for f in ride_info_files if f not in processed_files]
        logger.info(f"Found {len(unprocessed_ride_info)} unprocessed ride info files")
        
        for i in range(0, len(unprocessed_ride_info), BATCH_SIZE):
            batch = unprocessed_ride_info[i:i + BATCH_SIZE]
            if not batch:
                continue
            
            total_batches = (len(unprocessed_ride_info) - 1) // BATCH_SIZE + 1
            current_batch = i // BATCH_SIZE + 1
            remaining_batches = total_batches - current_batch
            
            logger.info(f"Processing batch {current_batch}/{total_batches} of ride info files")
            logger.info(f"Remaining batches after this one: {remaining_batches}")
            log_memory_usage()
            
            temp_files, new_processed, batch_rows = process_batch(batch, s3, S3_BUCKET_PREFIX, 'ride_info')
            ride_info_temp_files.extend(temp_files)
            
            # Update totals
            processed_files.update(new_processed)
            total_processed += len(new_processed)
            total_ride_info_rows += batch_rows
            
            # Save checkpoint after each batch
            save_checkpoint(list(processed_files), 'ride_info', total_files, total_processed)
            
            # Progress summary
            percent_complete = total_processed / total_files * 100 if total_files > 0 else 0
            remaining_files = total_files - total_processed
            logger.info(f"PROGRESS SUMMARY:")
            logger.info(f"  Files: {total_processed}/{total_files} ({percent_complete:.2f}%)")
            logger.info(f"  Remaining files: {remaining_files}")
            logger.info(f"  Ride info rows processed so far: {total_ride_info_rows}")
            expected_remaining_time = (time.time() - start_time) / total_processed * remaining_files if total_processed > 0 else 0
            logger.info(f"  Estimated remaining time: {expected_remaining_time:.2f} seconds ({expected_remaining_time/60:.2f} minutes)")
            
            # Clear memory after each batch
            gc.collect()
    
    # Mark the transition to processing fare files
    save_checkpoint(list(processed_files), 'both', total_files, total_processed)
    
    # Process ride fare files in batches
    ride_fare_temp_files = []
    logger.info("Processing ride fare files...")
    
    # Create batches of unprocessed files
    unprocessed_ride_fare = [f for f in ride_fare_files if f not in processed_files]
    logger.info(f"Found {len(unprocessed_ride_fare)} unprocessed ride fare files")
    
    for i in range(0, len(unprocessed_ride_fare), BATCH_SIZE):
        batch = unprocessed_ride_fare[i:i + BATCH_SIZE]
        if not batch:
            continue
        
        total_batches = (len(unprocessed_ride_fare) - 1) // BATCH_SIZE + 1
        current_batch = i // BATCH_SIZE + 1
        remaining_batches = total_batches - current_batch
        
        logger.info(f"Processing batch {current_batch}/{total_batches} of ride fare files")
        logger.info(f"Remaining batches after this one: {remaining_batches}")
        log_memory_usage()
        
        temp_files, new_processed, batch_rows = process_batch(batch, s3, S3_BUCKET_PREFIX, 'ride_fare')
        ride_fare_temp_files.extend(temp_files)
        
        # Update totals
        processed_files.update(new_processed)
        total_processed += len(new_processed)
        total_ride_fare_rows += batch_rows
        
        # Save checkpoint after each batch
        save_checkpoint(list(processed_files), 'both', total_files, total_processed)
        
        # Progress summary
        percent_complete = total_processed / total_files * 100 if total_files > 0 else 0
        remaining_files = total_files - total_processed
        logger.info(f"PROGRESS SUMMARY:")
        logger.info(f"  Files: {total_processed}/{total_files} ({percent_complete:.2f}%)")
        logger.info(f"  Remaining files: {remaining_files}")
        logger.info(f"  Ride fare rows processed so far: {total_ride_fare_rows}")
        expected_remaining_time = (time.time() - start_time) / total_processed * remaining_files if total_processed > 0 else 0
        logger.info(f"  Estimated remaining time: {expected_remaining_time:.2f} seconds ({expected_remaining_time/60:.2f} minutes)")
        
        # Clear memory after each batch
        gc.collect()
    
    # Log completion of file processing
    file_processing_time = time.time() - start_time
    logger.info(f"All files processed in {file_processing_time:.2f} seconds")
    logger.info(f"Total ride info rows: {total_ride_info_rows}, Total ride fare rows: {total_ride_fare_rows}")
    
    # Combine temporary files
    logger.info("Combining ride info files...")
    log_memory_usage()
    
    # Process in batches to avoid memory issues
    ride_info_df = None
    for i in range(0, len(ride_info_temp_files), 10):  # Process 10 temp files at a time
        batch_files = ride_info_temp_files[i:i+10]
        batch_dfs = []
        
        for temp_file in batch_files:
            try:
                chunk = pd.read_parquet(temp_file)
                batch_dfs.append(chunk)
                os.unlink(temp_file)  # Delete temporary file
            except Exception as e:
                logger.error(f"Error reading temp file {temp_file}: {str(e)}")
        
        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            if ride_info_df is None:
                ride_info_df = batch_df
            else:
                ride_info_df = pd.concat([ride_info_df, batch_df], ignore_index=True)
            
            # Clean up
            del batch_dfs, batch_df
            gc.collect()
    
    logger.info(f"Ride info data shape: {ride_info_df.shape if ride_info_df is not None else 'No data'}")
    
    # Combine ride fare files
    logger.info("Combining ride fare files...")
    log_memory_usage()
    
    # Process in batches to avoid memory issues
    ride_fare_df = None
    for i in range(0, len(ride_fare_temp_files), 10):  # Process 10 temp files at a time
        batch_files = ride_fare_temp_files[i:i+10]
        batch_dfs = []
        
        for temp_file in batch_files:
            try:
                chunk = pd.read_parquet(temp_file)
                batch_dfs.append(chunk)
                os.unlink(temp_file)  # Delete temporary file
            except Exception as e:
                logger.error(f"Error reading temp file {temp_file}: {str(e)}")
        
        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            if ride_fare_df is None:
                ride_fare_df = batch_df
            else:
                ride_fare_df = pd.concat([ride_fare_df, batch_df], ignore_index=True)
            
            # Clean up
            del batch_dfs, batch_df
            gc.collect()
    
    logger.info(f"Ride fare data shape: {ride_fare_df.shape if ride_fare_df is not None else 'No data'}")
    
    # Join the datasets
    logger.info("Joining ride info and fare datasets...")
    log_memory_usage()
    
    if ride_info_df is None or ride_fare_df is None:
        raise ValueError("One or both datasets are empty")
    
    merged_df = pd.merge(ride_info_df, ride_fare_df, on='ride_id', how='inner')
    
    # Clean up memory
    del ride_info_df, ride_fare_df
    gc.collect()
    
    total_time = time.time() - start_time
    logger.info(f"Data loading completed in {total_time:.2f} seconds")
    logger.info(f"Final dataset shape: {merged_df.shape}")
    return merged_df

def analyze_data_structure(df):
    """Analyze data structure and identify columns to drop."""
    logger.info("Analyzing data structure...")
    
    # Analyze column types and missing values
    column_analysis = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_values': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_values': df.nunique()
    })
    
    # Save column analysis
    column_analysis.to_csv('column_analysis.csv')
    
    # Identify columns to drop
    columns_to_drop = []
    
    # Drop columns with high missing values (>50%)
    high_missing = column_analysis[column_analysis['missing_percentage'] > 50].index
    columns_to_drop.extend(high_missing)
    
    # Drop columns with only one unique value
    single_value = column_analysis[column_analysis['unique_values'] == 1].index
    columns_to_drop.extend(single_value)
    
    # Drop columns that are not relevant for fare prediction
    irrelevant_columns = ['ride_id', 'store_and_fwd_flag', 'payment_type']
    columns_to_drop.extend(irrelevant_columns)
    
    # Remove duplicates
    columns_to_drop = list(set(columns_to_drop))
    
    logger.info(f"Columns to drop: {columns_to_drop}")
    
    # Drop identified columns
    df = df.drop(columns=columns_to_drop)
    
    return df

def explore_data(df):
    """Perform exploratory data analysis."""
    logger.info("Starting exploratory data analysis...")
    
    # Calculate trip duration if it doesn't exist already
    if 'trip_duration' not in df.columns and 'pickup_at' in df.columns and 'dropoff_at' in df.columns:
        logger.info("Calculating trip duration...")
        df['trip_duration'] = (df['dropoff_at'] - df['pickup_at']).dt.total_seconds() / 60
    
    # Basic statistics
    logger.info("\nData Shape: %s", df.shape)
    logger.info("\nColumns: %s", df.columns.tolist())
    logger.info("\nData Types:\n%s", df.dtypes)
    logger.info("\nMissing Values:\n%s", df.isnull().sum())
    logger.info("\nBasic Statistics:\n%s", df.describe())
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Fare distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='fare_amount', bins=50)
    plt.title('Distribution of Fare Amounts')
    plt.savefig('fare_distribution.png')
    plt.close()
    
    # Fare vs. trip distance (instead of duration)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df.sample(n=10000, random_state=RANDOM_STATE), 
                   x='trip_distance', y='fare_amount', alpha=0.5)
    plt.title('Fare Amount vs. Trip Distance')
    plt.savefig('fare_vs_distance.png')
    plt.close()
    
    # If trip_duration was calculated, plot that too
    if 'trip_duration' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df.sample(n=10000, random_state=RANDOM_STATE), 
                       x='trip_duration', y='fare_amount', alpha=0.5)
        plt.title('Fare Amount vs. Trip Duration')
        plt.savefig('fare_vs_duration.png')
        plt.close()
    
    # Pickup hour (extract from pickup_at)
    if 'pickup_hour' not in df.columns and 'pickup_at' in df.columns:
        df['pickup_hour'] = df['pickup_at'].dt.hour
    
    # Fare by hour of day
    if 'pickup_hour' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='pickup_hour', y='fare_amount')
        plt.title('Fare Amount by Hour of Day')
        plt.savefig('fare_by_hour.png')
        plt.close()
    
    # Save summary statistics
    logger.info("Saving summary statistics...")
    with open('data_summary.txt', 'w') as f:
        f.write("Data Summary\n")
        f.write("============\n\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Average Fare: ${df['fare_amount'].mean():.2f}\n")
        f.write(f"Median Fare: ${df['fare_amount'].median():.2f}\n")
        f.write(f"Standard Deviation: ${df['fare_amount'].std():.2f}\n")
        if 'pickup_hour' in df.columns:
            f.write(f"\nFare Statistics by Hour:\n")
            f.write(df.groupby('pickup_hour')['fare_amount'].describe().to_string())

def prepare_data(df):
    """Prepare data for model training."""
    logger.info("Starting data preparation...")
    
    # Split the data
    logger.info("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(
        df,
        test_size=0.3,  # 30% validation
        random_state=RANDOM_STATE
    )
    
    # Save processed data in chunks
    logger.info("Saving processed data to S3...")
    s3 = boto3.client('s3')
    
    # Save training data in chunks
    for i, chunk in enumerate(np.array_split(train_df, max(1, len(train_df) // CHUNK_SIZE))):
        chunk.to_parquet(f's3://{PROCESSED_DATA_BUCKET}/{TRAINING_DATA_PATH}train_chunk_{i}.parquet')
    
    # Save validation data in chunks
    for i, chunk in enumerate(np.array_split(val_df, max(1, len(val_df) // CHUNK_SIZE))):
        chunk.to_parquet(f's3://{PROCESSED_DATA_BUCKET}/{VALIDATION_DATA_PATH}validation_chunk_{i}.parquet')
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Validation data shape: {val_df.shape}")
    
    # Define feature columns
    feature_columns = [
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'pickup_hour', 'pickup_day', 'pickup_month', 'pickup_dayofweek',
        'trip_duration'
    ]
    
    target_column = 'fare_amount'
    
    return train_df, val_df, feature_columns, target_column

def main():
    """Main data processing function."""
    start_time = datetime.now()
    logger.info("Starting data processing...")
    
    try:
        # Load data
        df = load_data_from_s3()
        
        # Analyze data structure
        df = analyze_data_structure(df)
        
        # Explore data
        explore_data(df)
        
        # Prepare data
        train_df, val_df, feature_columns, target_column = prepare_data(df)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Data processing completed successfully! Duration: {duration}")
        
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()