import os
import gc
import time
import boto3
from botocore.config import Config
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
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pyarrow.parquet import ParquetFile
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from dask.distributed import Client, LocalCluster
from dask import delayed
import dask

# Create output directories
OUTPUT_DIR = "output"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
STATS_DIR = os.path.join(OUTPUT_DIR, "statistics")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")

# Create directories if they don't exist
for directory in [OUTPUT_DIR, LOG_DIR, VISUALIZATION_DIR, STATS_DIR, CHECKPOINT_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set up logging to file in the LOG_DIR
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'data_processing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Time tracking dictionary to store timing information
timing_stats = {
    'start_time': None,
    'end_time': None,
    'total_duration': None,
    'stages': {}
}

def start_timer(stage_name):
    """Start timing a specific stage of processing."""
    if stage_name not in timing_stats['stages']:
        timing_stats['stages'][stage_name] = {
            'start_time': datetime.now(),
            'end_time': None,
            'duration': None
        }
    else:
        timing_stats['stages'][stage_name]['start_time'] = datetime.now()
    
    logger.info(f"Starting stage: {stage_name} at {timing_stats['stages'][stage_name]['start_time']}")
    return timing_stats['stages'][stage_name]['start_time']

def end_timer(stage_name):
    """End timing for a specific stage and log the duration."""
    if stage_name in timing_stats['stages']:
        timing_stats['stages'][stage_name]['end_time'] = datetime.now()
        duration = timing_stats['stages'][stage_name]['end_time'] - timing_stats['stages'][stage_name]['start_time']
        timing_stats['stages'][stage_name]['duration'] = duration
        
        logger.info(f"Completed stage: {stage_name} in {format_duration(duration)}")
        return duration
    return None

def format_duration(duration):
    """Format a timedelta object into a readable string."""
    total_seconds = duration.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"

def save_timing_report():
    """Save the timing statistics to a JSON file."""
    # Convert datetime objects to strings for JSON serialization
    json_compatible_stats = {
        'start_time': timing_stats['start_time'].isoformat() if timing_stats['start_time'] else None,
        'end_time': timing_stats['end_time'].isoformat() if timing_stats['end_time'] else None,
        'total_duration': str(timing_stats['total_duration']) if timing_stats['total_duration'] else None,
        'stages': {}
    }
    
    for stage, stats in timing_stats['stages'].items():
        json_compatible_stats['stages'][stage] = {
            'start_time': stats['start_time'].isoformat() if stats['start_time'] else None,
            'end_time': stats['end_time'].isoformat() if stats['end_time'] else None,
            'duration': str(stats['duration']) if stats['duration'] else None
        }
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_file = os.path.join(STATS_DIR, f'timing_report_{timestamp}.json')
    with open(timing_file, 'w') as f:
        json.dump(json_compatible_stats, f, indent=2)
    
    logger.info(f"Timing report saved to {timing_file}")
    
    # Also create a text summary for human readability
    summary_file = os.path.join(STATS_DIR, f'timing_summary_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write("DATA PROCESSING TIMING SUMMARY\n")
        f.write("==============================\n\n")
        f.write(f"Started: {timing_stats['start_time']}\n")
        f.write(f"Completed: {timing_stats['end_time']}\n")
        f.write(f"Total Duration: {format_duration(timing_stats['total_duration'])}\n\n")
        
        f.write("Stage Durations:\n")
        f.write("--------------\n")
        for stage, stats in timing_stats['stages'].items():
            if stats['duration']:
                f.write(f"{stage}: {format_duration(stats['duration'])}\n")
    
    logger.info(f"Timing summary saved to {summary_file}")

# Constants for chunked processing with updated file paths
CHUNK_SIZE = 2000000
MAX_WORKERS = min(32, os.cpu_count() * 4)
BATCH_SIZE = 25
MAX_BATCH_WORKERS = min(8, os.cpu_count())
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'processing_checkpoint.json')
FILE_PROGRESS_FILE = os.path.join(CHECKPOINT_DIR, 'file_progress.json')
TESTING_MODE = True
# Allow setting test file count from environment variable
TEST_FILE_COUNT = 100  # Default to 3 files in test mode
S3_BUCKET_PREFIX="nyc-taxi-dataset-public"
RIDE_INFO_PATH = 'nyc-taxi-orig-cleaned-split-parquet-per-year-multiple-files/ride-info/'
RIDE_FARE_PATH = 'nyc-taxi-orig-cleaned-split-parquet-per-year-multiple-files/ride-fare/'
TRAIN_TEST_SPLIT_RATIO = 0.7
RANDOM_STATE = 42

# Output Configuration
PROCESSED_DATA_BUCKET = 'nyc-taxi-processed-data'
TRAINING_DATA_PATH = 'processed-last/training/'
VALIDATION_DATA_PATH = 'processed-last/validation/'
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

def save_checkpoint(processed_files, data_type, total_files, total_processed, has_data=False):
    """Save checkpoint of processed files."""
    checkpoint = {
        'processed_files': processed_files,
        'data_type': data_type,
        'timestamp': datetime.now().isoformat(),
        'total_files': total_files,
        'total_processed': total_processed,
        'percentage_complete': (total_processed / total_files * 100) if total_files > 0 else 0,
        'has_data': has_data,
        'elapsed_time': str(datetime.now() - timing_stats['start_time']) if timing_stats['start_time'] else None
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def read_parquet_with_dask(s3_client, bucket, file_key):
    """Read parquet file using Dask for better memory management."""
    try:
        # Log start of processing
        logger.info(f"Processing file with Dask: {file_key}")
        
        # Get the S3 path
        s3_path = f"s3://{bucket}/{file_key}"
        
        # Read with Dask - this is lazy and doesn't load data yet
        ddf = dd.read_parquet(
            s3_path,
            storage_options={
                'use_ssl': True,
                'anon': False  # Use AWS credentials
            },
            engine='pyarrow',
            calculate_divisions=False  # Skip index calculation for speed
        )
        
        # Calculate the total number of rows across all partitions
        # This is still lazy and doesn't load the data
        row_count = len(ddf)
        
        # Add time-based features (these operations are still lazy)
        if 'pickup_datetime' in ddf.columns:
            # Extract time components
            ddf['pickup_hour'] = ddf['pickup_datetime'].dt.hour
            ddf['pickup_day'] = ddf['pickup_datetime'].dt.day
            ddf['pickup_month'] = ddf['pickup_datetime'].dt.month
            ddf['pickup_dayofweek'] = ddf['pickup_datetime'].dt.dayofweek
        
        # Calculate trip duration if relevant columns exist
        if 'pickup_datetime' in ddf.columns and 'dropoff_datetime' in ddf.columns:
            ddf['trip_duration'] = (ddf['dropoff_datetime'] - ddf['pickup_datetime']).dt.total_seconds() / 60
        
        # Log success (still haven't loaded the data)
        logger.info(f"Prepared Dask dataframe for {file_key} with {row_count} rows")
        
        return ddf, row_count
        
    except Exception as e:
        logger.error(f"Error reading file {file_key} with Dask: {str(e)}")
        return None, 0

def process_file(file_key, s3_client, bucket, data_type):
    """Process a single file using Dask."""
    try:
        ddf, rows = read_parquet_with_dask(s3_client, bucket, file_key)
        if ddf is not None:
            # Save intermediate result to our custom TEMP_DIR
            temp_filename = os.path.join(TEMP_DIR, f"{data_type}_{os.path.basename(file_key)}.parquet")
            
            # Use Dask's to_parquet - this triggers computation but with better memory management
            ddf.to_parquet(
                temp_filename,
                compression="snappy",
                write_index=False,
                engine="pyarrow"
            )
            
            # Log success
            logger.info(f"Successfully processed {data_type} file: {file_key} ({rows} rows)")
            log_memory_usage()
            
            # Clean up
            del ddf
            gc.collect()
            
            return temp_filename, rows
    except Exception as e:
        logger.error(f"Error processing file {file_key}: {str(e)}")
    
    return None, 0

def process_batch(batch_files, s3_client, bucket, data_type):
    """Process a batch of files using ThreadPoolExecutor."""
    start_time = time.time()
    temp_files = []
    processed_files = []
    total_rows = 0
    
    # Check disk space and clean up if necessary before batch processing
    check_disk_space(TEMP_DIR)
    
    # Reduce batch size dynamically if needed based on disk space
    free_gb, used_percent = check_disk_space(TEMP_DIR)
    if free_gb is not None and free_gb < 5.0:
        # Reduce number of concurrent workers when disk space is low
        effective_workers = max(1, min(MAX_BATCH_WORKERS // 2, 2))
        logger.info(f"Reducing worker count due to low disk space: {effective_workers} workers")
    else:
        effective_workers = MAX_BATCH_WORKERS
    
    # We need to modify how we pass the s3_client
    def process_file_with_new_client(file_key):
        # Create a new S3 client inside the worker
        s3 = boto3.client('s3')
        return process_file(file_key, s3, bucket, data_type)
    
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_file = {executor.submit(process_file_with_new_client, file_key): file_key 
                         for file_key in batch_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(batch_files), 
                          desc=f"Processing {data_type} batch"):
            file_key = future_to_file[future]
            try:
                temp_file, rows = future.result()
                if temp_file:
                    temp_files.append(temp_file)
                    processed_files.append(file_key)
                    total_rows += rows
                    
                    # Check disk space after each file and clean up if necessary
                    if len(temp_files) % 5 == 0:  # Check every 5 files
                        check_disk_space(TEMP_DIR)
            except Exception as e:
                logger.error(f"Error processing file {file_key}: {str(e)}")
    
    elapsed_time = time.time() - start_time
    files_per_second = len(processed_files) / elapsed_time if elapsed_time > 0 else 0
    rows_per_second = total_rows / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Batch completed: {len(processed_files)}/{len(batch_files)} files processed in {elapsed_time:.2f} seconds")
    logger.info(f"Performance: {files_per_second:.2f} files/sec, {rows_per_second:.2f} rows/sec")
    
    return temp_files, processed_files, total_rows

def combine_dask_dataframes(temp_files, batch_size=10):
    """Combine files using Dask for better memory efficiency."""
    try:
        # Create a list to store Dask dataframes
        dask_dfs = []
        processed_count = 0
        
        # Process each file or batch of files
        for i in range(0, len(temp_files), batch_size):
            # Check disk space before processing a new batch
            check_disk_space(TEMP_DIR)
            
            batch_files = temp_files[i:i+batch_size]
            batch_dfs = []
            
            # Process each temp file
            for temp_file in batch_files:
                try:
                    if temp_file and os.path.exists(temp_file):
                        # Read with Dask instead of pandas - lazy loading
                        ddf = dd.read_parquet(temp_file, engine='pyarrow')
                        batch_dfs.append(ddf)
                        
                        processed_count += 1
                        if processed_count % 5 == 0:
                            logger.info(f"Processed {processed_count}/{len(temp_files)} temporary files")
                except Exception as e:
                    logger.error(f"Error reading temp file {temp_file}: {str(e)}")
            
            if batch_dfs:
                # Combine the batch - this is still lazy
                batch_combined = dd.concat(batch_dfs)
                dask_dfs.append(batch_combined)
                
                # Clean up
                del batch_dfs
                gc.collect()
        
        if dask_dfs:
            # Combine all batches - still lazy
            result_ddf = dd.concat(dask_dfs)
            
            # Optimize before computing
            result_ddf = result_ddf.repartition(
                npartitions=min(100, max(1, result_ddf.npartitions // 2))
            )
            
            # Trigger computation with better memory management
            result_df = result_ddf.compute()
            
            # Clean up
            del dask_dfs, result_ddf
            gc.collect()
            
            return result_df
        
        return None
    except Exception as e:
        logger.error(f"Error combining Dask dataframes: {str(e)}")
        return None

def load_data_from_s3():
    """Load data from S3 bucket using Dask for better memory management."""
    stage_name = "data_loading"
    start_timer(stage_name)
    
    # Set up Dask client
    client = setup_dask_client()
    
    start_time = time.time()
    
    # Modify the tempfile module's temporary directory to use our custom directory
    tempfile.tempdir = TEMP_DIR
    logger.info(f"Set temporary directory to: {TEMP_DIR}")
    
    # Ensure TEMP_DIR exists and is empty
    if os.path.exists(TEMP_DIR):
        # Clean up any old temp files
        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(filepath):
                try:
                    os.unlink(filepath)
                    logger.info(f"Cleaned up old temp file: {filepath}")
                except Exception as e:
                    logger.error(f"Error cleaning up {filepath}: {str(e)}")
    
    # Check available disk space before starting
    check_disk_space(TEMP_DIR)
    
    # Initialize session with optimized config
    session = boto3.Session()
    s3 = session.client('s3', config=Config(
        max_pool_connections=50,
        retries={'max_attempts': 10}
    ))
    
    # Force clear checkpoint if needed
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            # Check if we have actual processed data
            if not checkpoint.get('has_data', False):
                logger.info("Previous run didn't produce data, clearing checkpoint")
                os.remove(CHECKPOINT_FILE)
    
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
    
    # Process a small subset for testing in notebook
    if TESTING_MODE:
        ride_info_files = ride_info_files[:3]  # Only process 3 files when in notebook
        ride_fare_files = ride_fare_files[:3]
    
    # FORCE CLEAR: Treat all files as unprocessed if no data exists
    if not os.path.exists(CHECKPOINT_FILE) or not processed_files:
        logger.info("No valid checkpoint found, treating all files as unprocessed")
        processed_files = set()
    
    ride_info_count = len(ride_info_files)
    ride_fare_count = len(ride_fare_files)
    total_files = ride_info_count + ride_fare_count
    
    logger.info(f"Processing {ride_info_count} ride info files and {ride_fare_count} ride fare files")
    
    # Keep track of totals
    total_processed = len(processed_files)
    total_ride_info_rows = 0
    total_ride_fare_rows = 0
    
    # Process ride info files with Dask
    ride_info_temp_files = []
    if data_type == 'ride_info' or data_type == 'both':
        logger.info("Processing ride info files with Dask...")
        
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
            
            # Process batch
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
    
    # Process ride fare files with Dask
    ride_fare_temp_files = []
    logger.info("Processing ride fare files with Dask...")
    
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
        
        # Process batch
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
    
    # Use Dask to combine all the files more efficiently
    logger.info("Combining ride info files with Dask...")
    log_memory_usage()
    ride_info_df = combine_dask_dataframes(ride_info_temp_files)
    
    logger.info("Combining ride fare files with Dask...")
    log_memory_usage()
    ride_fare_df = combine_dask_dataframes(ride_fare_temp_files)
    
    # Optimize merge operation using Dask
    logger.info("Joining ride info and fare datasets with Dask...")
    log_memory_usage()
    
    if ride_info_df is None or ride_fare_df is None:
        raise ValueError("One or both datasets are empty")
    
    # Convert to Dask dataframes for memory-efficient merge
    dask_info = dd.from_pandas(ride_info_df, npartitions=max(1, os.cpu_count()))
    dask_fare = dd.from_pandas(ride_fare_df, npartitions=max(1, os.cpu_count()))
    
    # Use Dask's merge which is more memory-efficient
    merged_ddf = dask_info.merge(dask_fare, on='ride_id', how='inner')
    
    # Compute the result
    merged_df = merged_ddf.compute()
    
    # Clean up memory
    del ride_info_df, ride_fare_df, dask_info, dask_fare, merged_ddf
    gc.collect()
    
    total_time = time.time() - start_time
    logger.info(f"Data loading completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Final dataset shape: {merged_df.shape}")
    
    # Save final checkpoint with data status and timing info
    save_checkpoint(list(processed_files), 'both', total_files, total_processed, 
                  has_data=True)
    
    # End timer for this stage
    end_timer(stage_name)
    
    # Shut down the Dask client
    client.close()
    
    return merged_df

def analyze_data_structure(df):
    """Analyze data structure and identify columns to drop."""
    stage_name = "data_analysis"
    start_timer(stage_name)
    
    logger.info("Analyzing data structure...")
    
    # Analyze column types and missing values
    column_analysis = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_values': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_values': df.nunique()
    })
    
    # Save column analysis to statistics directory
    column_analysis.to_csv(os.path.join(STATS_DIR, 'column_analysis.csv'))
    
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
    
    end_timer(stage_name)
    return df

def explore_data(df):
    """Perform exploratory data analysis."""
    stage_name = "exploratory_analysis"
    start_timer(stage_name)
    
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
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'fare_distribution.png'))
    plt.close()
    
    # Fare vs. trip distance (instead of duration)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df.sample(n=10000, random_state=RANDOM_STATE), 
                   x='trip_distance', y='fare_amount', alpha=0.5)
    plt.title('Fare Amount vs. Trip Distance')
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'fare_vs_distance.png'))
    plt.close()
    
    # If trip_duration was calculated, plot that too
    if 'trip_duration' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df.sample(n=10000, random_state=RANDOM_STATE), 
                       x='trip_duration', y='fare_amount', alpha=0.5)
        plt.title('Fare Amount vs. Trip Duration')
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'fare_vs_duration.png'))
        plt.close()
    
    # Pickup hour (extract from pickup_at)
    if 'pickup_hour' not in df.columns and 'pickup_at' in df.columns:
        df['pickup_hour'] = df['pickup_at'].dt.hour
    
    # Fare by hour of day
    if 'pickup_hour' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='pickup_hour', y='fare_amount')
        plt.title('Fare Amount by Hour of Day')
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'fare_by_hour.png'))
        plt.close()
    
    # Save summary statistics to statistics directory
    with open(os.path.join(STATS_DIR, 'data_summary.txt'), 'w') as f:
        f.write("Data Summary\n")
        f.write("============\n\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Average Fare: ${df['fare_amount'].mean():.2f}\n")
        f.write(f"Median Fare: ${df['fare_amount'].median():.2f}\n")
        f.write(f"Standard Deviation: ${df['fare_amount'].std():.2f}\n")
        if 'pickup_hour' in df.columns:
            f.write(f"\nFare Statistics by Hour:\n")
            f.write(df.groupby('pickup_hour')['fare_amount'].describe().to_string())
    
    end_timer(stage_name)

def prepare_data(df):
    """Prepare data for model training using Dask for better memory efficiency."""
    stage_name = "data_preparation"
    start_timer(stage_name)
    
    logger.info("Starting data preparation with Dask...")
    
    # Convert to Dask DataFrame for more efficient operations
    ddf = dd.from_pandas(df, npartitions=max(1, os.cpu_count()))
    
    # Optimize: Use more efficient train/test split with Dask
    logger.info("Splitting data into training and validation sets...")
    
    # Generate a random mask for splitting
    # This uses numpy's random state which is more memory-efficient
    np.random.seed(RANDOM_STATE)
    mask = np.random.rand(len(df)) < TRAIN_TEST_SPLIT_RATIO
    
    # Split based on mask
    train_df = df[mask].reset_index(drop=True)
    val_df = df[~mask].reset_index(drop=True)
    
    # Convert to Dask dataframes for more efficient processing
    train_ddf = dd.from_pandas(train_df, npartitions=max(1, os.cpu_count()))
    val_ddf = dd.from_pandas(val_df, npartitions=max(1, os.cpu_count()))
    
    # Optimize: Save processed data with Dask
    logger.info("Saving processed data to S3...")
    
    # Create a Dask S3 write function for better memory management
    def save_dask_to_s3(ddf, bucket, key_prefix, npartitions=10):
        # Repartition to optimize file sizes
        ddf = ddf.repartition(npartitions=npartitions)
        
        # Save to S3 using Dask's to_parquet
        path = f"s3://{bucket}/{key_prefix}"
        
        ddf.to_parquet(
            path,
            engine='pyarrow',
            compression='snappy',
            write_index=False,
            storage_options={
                'anon': False,  # Use AWS credentials
            }
        )
        return path
    
    # Save train and validation data with Dask
    train_path = save_dask_to_s3(train_ddf, PROCESSED_DATA_BUCKET, TRAINING_DATA_PATH)
    val_path = save_dask_to_s3(val_ddf, PROCESSED_DATA_BUCKET, VALIDATION_DATA_PATH)
    
    logger.info(f"Training data saved to: {train_path}")
    logger.info(f"Validation data saved to: {val_path}")
    
    # Convert back to pandas for further processing if needed
    train_df = train_ddf.compute()
    val_df = val_ddf.compute()
    
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
    
    end_timer(stage_name)
    return train_df, val_df, feature_columns, target_column

def load_checkpoint():
    """Load checkpoint of processed files."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading checkpoint file: {str(e)}")
            return None
    return None

def main():
    """Main data processing function."""
    # Set the global start time
    timing_stats['start_time'] = datetime.now()
    logger.info(f"Starting data processing at {timing_stats['start_time']}")
    
    try:
        # Load data with timing
        stage_name = "overall_data_loading"
        start_timer(stage_name)
        df = load_data_from_s3()
        end_timer(stage_name)
        
        # Analyze data structure with timing
        stage_name = "overall_data_analysis"
        start_timer(stage_name)
        df = analyze_data_structure(df)
        end_timer(stage_name)
        
        # Explore data with timing
        stage_name = "overall_data_exploration"
        start_timer(stage_name)
        explore_data(df)
        end_timer(stage_name)
        
        # Prepare data with timing
        stage_name = "overall_data_preparation"
        start_timer(stage_name)
        train_df, val_df, feature_columns, target_column = prepare_data(df)
        end_timer(stage_name)
        
        # Record the end time and calculate total duration
        timing_stats['end_time'] = datetime.now()
        timing_stats['total_duration'] = timing_stats['end_time'] - timing_stats['start_time']
        
        # Log completion
        logger.info(f"Data processing completed successfully!")
        logger.info(f"Started: {timing_stats['start_time']}")
        logger.info(f"Finished: {timing_stats['end_time']}")
        logger.info(f"Total Duration: {format_duration(timing_stats['total_duration'])}")
        
        # Save timing report
        save_timing_report()
        
    except Exception as e:
        # Record failure time
        timing_stats['end_time'] = datetime.now()
        timing_stats['total_duration'] = timing_stats['end_time'] - timing_stats['start_time']
        
        logger.error(f"Error during data processing: {str(e)}", exc_info=True)
        
        # Save timing report even if there's an error
        save_timing_report()
        raise

# Helper function to check and manage disk space
def check_disk_space(directory):
    """Check and manage disk space in the given directory."""
    try:
        # Get disk stats
        disk = psutil.disk_usage(directory)
        free_gb = disk.free / (1024 * 1024 * 1024)  # Convert to GB
        
        # Log disk space status
        logger.info(f"Disk space check - Free: {free_gb:.2f} GB, Used: {disk.percent:.1f}%")
        
        # If less than 2GB free or >90% used, clean up temporary files
        if free_gb < 2.0 or disk.percent > 90:
            logger.warning(f"Low disk space detected! Free: {free_gb:.2f} GB, Used: {disk.percent:.1f}%")
            
            # Clean up oldest temporary files in TEMP_DIR
            if os.path.exists(TEMP_DIR):
                temp_files = []
                for filename in os.listdir(TEMP_DIR):
                    filepath = os.path.join(TEMP_DIR, filename)
                    if os.path.isfile(filepath) and filepath.endswith('.parquet'):
                        temp_files.append((filepath, os.path.getmtime(filepath)))
                
                # Sort by modification time (oldest first)
                temp_files.sort(key=lambda x: x[1])
                
                # Remove up to 50% of temp files if we have a critical situation
                files_to_remove = len(temp_files) // 2 if disk.percent > 95 else len(temp_files) // 4
                for filepath, _ in temp_files[:files_to_remove]:
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed temporary file to free space: {filepath}")
                    except Exception as e:
                        logger.error(f"Failed to remove temporary file {filepath}: {str(e)}")
            
            # Check disk space again
            disk = psutil.disk_usage(directory)
            free_gb = disk.free / (1024 * 1024 * 1024)
            logger.info(f"After cleanup - Free: {free_gb:.2f} GB, Used: {disk.percent:.1f}%")
            
        return free_gb, disk.percent
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        return None, None

# Reduce batch size to process fewer files at once (less memory and disk space usage)
BATCH_SIZE = 25  # Reduced from 50

# Add a dynamic chunk size based on available memory
def get_optimal_chunk_size():
    """Determine optimal chunk size based on system memory."""
    try:
        # Get memory info
        memory_info = psutil.virtual_memory()
        total_gb = memory_info.total / (1024 * 1024 * 1024)
        available_gb = memory_info.available / (1024 * 1024 * 1024)
        
        # Calculate chunk size based on available memory
        # Use a more conservative approach when memory is limited
        if available_gb < 2.0:
            return 500000  # Very small chunks when memory is tight
        elif available_gb < 4.0:
            return 1000000  # Small chunks
        elif available_gb < 8.0:
            return 1500000  # Medium chunks
        else:
            return 2000000  # Original size
    except:
        # Default to a conservative size if we can't determine memory
        return 1000000

# Update CHUNK_SIZE to be dynamic
CHUNK_SIZE = get_optimal_chunk_size()
logger.info(f"Using dynamic chunk size: {CHUNK_SIZE} rows")

# Optimize Dask processing by creating a local cluster
def setup_dask_client():
    """Set up a local Dask cluster optimized for the available resources."""
    # Determine best worker count based on available resources
    cores = psutil.cpu_count(logical=False)
    ram_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    
    # Calculate optimal worker configuration
    # Use fewer workers with more memory if RAM is constrained
    if ram_gb < 8:
        n_workers = max(1, cores // 2)
        threads_per_worker = 1
    else:
        n_workers = max(1, cores - 1)  # Leave one core for system processes
        threads_per_worker = 2
    
    # Calculate memory limit per worker
    memory_limit = int((ram_gb * 0.8) / n_workers)
    memory_limit = f"{memory_limit}GB"
    
    logger.info(f"Setting up Dask with {n_workers} workers, {threads_per_worker} threads per worker, {memory_limit} RAM per worker")
    
    # Create cluster and client
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        local_directory=TEMP_DIR  # Use our temp directory for spilling
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    return client

# Add an efficient full Dask pipeline option
def process_with_full_dask_pipeline():
    """Process the entire pipeline with Dask for maximum efficiency."""
    logger.info("Setting up full Dask pipeline for end-to-end processing")
    
    # Set up Dask client
    client = setup_dask_client()
    
    try:
        # List files from S3
        s3 = boto3.client('s3')
        
        # Get ride info files
        logger.info("Listing ride info files...")
        ride_info_objects = []
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_BUCKET_PREFIX, Prefix=RIDE_INFO_PATH):
            if 'Contents' in page:
                ride_info_objects.extend(page['Contents'])
        ride_info_keys = [obj['Key'] for obj in ride_info_objects]
        
        # Get ride fare files
        logger.info("Listing ride fare files...")
        ride_fare_objects = []
        for page in paginator.paginate(Bucket=S3_BUCKET_PREFIX, Prefix=RIDE_FARE_PATH):
            if 'Contents' in page:
                ride_fare_objects.extend(page['Contents'])
        ride_fare_keys = [obj['Key'] for obj in ride_fare_objects]
        
        if TESTING_MODE:
            logger.info(f"TESTING MODE ENABLED: Processing only {TEST_FILE_COUNT} files from each dataset")
            ride_info_keys = ride_info_keys[:TEST_FILE_COUNT]
            ride_fare_keys = ride_fare_keys[:TEST_FILE_COUNT]
        
        # Create paths for Dask
        ride_info_paths = [f"s3://{S3_BUCKET_PREFIX}/{key}" for key in ride_info_keys]
        ride_fare_paths = [f"s3://{S3_BUCKET_PREFIX}/{key}" for key in ride_fare_keys]
        
        logger.info(f"Processing {len(ride_info_paths)} ride info files and {len(ride_fare_paths)} ride fare files")
        
        # Create Dask dataframes directly from S3
        logger.info("Reading ride info data with Dask...")
        ride_info_ddf = dd.read_parquet(
            ride_info_paths,
            engine='pyarrow',
            storage_options={'anon': False},
            calculate_divisions=False
        )
        
        logger.info("Reading ride fare data with Dask...")
        ride_fare_ddf = dd.read_parquet(
            ride_fare_paths,
            engine='pyarrow',
            storage_options={'anon': False},
            calculate_divisions=False
        )
        
        # Add time-based features
        logger.info("Adding time-based features...")
        if 'pickup_datetime' in ride_info_ddf.columns:
            ride_info_ddf['pickup_hour'] = ride_info_ddf['pickup_datetime'].dt.hour
            ride_info_ddf['pickup_day'] = ride_info_ddf['pickup_datetime'].dt.day
            ride_info_ddf['pickup_month'] = ride_info_ddf['pickup_datetime'].dt.month
            ride_info_ddf['pickup_dayofweek'] = ride_info_ddf['pickup_datetime'].dt.dayofweek
        
        if 'pickup_datetime' in ride_info_ddf.columns and 'dropoff_datetime' in ride_info_ddf.columns:
            ride_info_ddf['trip_duration'] = (ride_info_ddf['dropoff_datetime'] - ride_info_ddf['pickup_datetime']).dt.total_seconds() / 60
        
        # Optimize partitions for better performance
        logger.info("Optimizing Dask partitions...")
        ride_info_ddf = ride_info_ddf.repartition(npartitions=100)
        ride_fare_ddf = ride_fare_ddf.repartition(npartitions=100)
        
        # Merge dataframes
        logger.info("Merging datasets with Dask...")
        merged_ddf = ride_info_ddf.merge(ride_fare_ddf, on='ride_id', how='inner')
        
        # Prepare for train/test split
        logger.info("Preparing train/test split...")
        
        # Fix: Use sample() instead of random_sample() for Dask dataframes
        train_ddf = merged_ddf.sample(frac=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_STATE)
        
        # Alternative approach for validation set to avoid index issues
        # Generate a boolean mask for selecting validation rows
        logger.info("Creating validation dataset...")
        # Create a random boolean series with the same index as merged_ddf
        # Use a different random seed to ensure we get complementary sets
        val_ddf = merged_ddf.sample(frac=1-TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_STATE+1)
        
        # Get metadata about datasets
        logger.info("Computing dataset statistics...")
        # Calculate dataset sizes - this triggers computation only for metadata
        train_nrows = len(train_ddf)
        train_ncols = len(train_ddf.columns)
        val_nrows = len(val_ddf)
        val_ncols = len(val_ddf.columns)
        
        logger.info("="*50)
        logger.info("Training Dataset")
        logger.info("Please provide the number of rows and columns for your training dataset:")
        logger.info(f"Rows: {train_nrows:,}")
        logger.info(f"Columns: {train_ncols}")
        logger.info(f"Column names: {list(train_ddf.columns)}")
        logger.info("="*50)
        
        logger.info("Validation Dataset")
        logger.info("Please provide the number of rows and columns for your validation dataset:")
        logger.info(f"Rows: {val_nrows:,}")
        logger.info(f"Columns: {val_ncols}")
        logger.info(f"Column names: {list(val_ddf.columns)}")
        logger.info("="*50)
        
        # Save to S3 directly from Dask
        logger.info("Saving results to S3...")
        
        train_path = f"s3://{PROCESSED_DATA_BUCKET}/{TRAINING_DATA_PATH}"
        val_path = f"s3://{PROCESSED_DATA_BUCKET}/{VALIDATION_DATA_PATH}"
        
        # Save with optimal settings
        train_ddf.to_parquet(
            train_path, 
            compression='snappy',
            write_index=False,
            engine='pyarrow',
            storage_options={'anon': False}
        )
        
        val_ddf.to_parquet(
            val_path, 
            compression='snappy',
            write_index=False,
            engine='pyarrow',
            storage_options={'anon': False}
        )
        
        logger.info(f"Training data saved to {train_path}")
        logger.info(f"Validation data saved to {val_path}")
        
        # For analysis, we need some stats in pandas - let's sample a small portion
        logger.info("Computing sample statistics...")
        sample_df = merged_ddf.sample(frac=0.01, random_state=RANDOM_STATE).compute()
        
        logger.info(f"Sample shape: {sample_df.shape}")
        logger.info(f"Column list: {list(sample_df.columns)}")
        
        # Clean up
        client.close()
        
        return sample_df
    
    except Exception as e:
        logger.error(f"Error in Dask pipeline: {str(e)}")
        client.close()
        raise

def run_test_mode(num_files=5):
    """Run the pipeline in test mode with a specified number of files.
    
    Args:
        num_files: Number of files to process from each dataset
    """
    # Set testing mode environment variables
    os.environ['TESTING_MODE'] = 'True'
    os.environ['TEST_FILE_COUNT'] = str(num_files)
    
    logger.info(f"Running in TEST MODE with {num_files} files from each dataset")
    
    # Call the main processing function
    process_with_full_dask_pipeline()

if __name__ == "__main__":
    try:
        # Print configuration information
        logger.info("="*50)
        logger.info("NYC TAXI DATA PROCESSING CONFIGURATION")
        logger.info("="*50)
        logger.info(f"Testing Mode: {TESTING_MODE}")
        if TESTING_MODE:
            logger.info(f"Test File Count: {TEST_FILE_COUNT}")
        logger.info(f"Chunk Size: {CHUNK_SIZE:,} rows")
        logger.info(f"Batch Size: {BATCH_SIZE} files")
        logger.info(f"Max Workers: {MAX_WORKERS}")
        logger.info(f"Max Batch Workers: {MAX_BATCH_WORKERS}")
        logger.info(f"Train/Test Split Ratio: {TRAIN_TEST_SPLIT_RATIO:.2f}/{1-TRAIN_TEST_SPLIT_RATIO:.2f}")
        logger.info(f"Temporary Directory: {TEMP_DIR}")
        logger.info("="*50)
        
        # Set temporary directory for all tempfile operations
        tempfile.tempdir = TEMP_DIR
        logger.info(f"Set temporary directory to: {TEMP_DIR}")
        
        # Clean up any old temp files before starting
        if os.path.exists(TEMP_DIR):
            for filename in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, filename)
                if os.path.isfile(filepath):
                    try:
                        os.unlink(filepath)
                    except:
                        pass
        
        # Option 1: Use the original pipeline with Dask enhancements
        # main()
        
        # Option 2: Uncomment to use the full Dask pipeline instead
        logger.info("Starting full Dask pipeline processing...")
        process_with_full_dask_pipeline()
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)