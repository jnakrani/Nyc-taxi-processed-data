# NYC Taxi Fare Prediction

A comprehensive pipeline for processing NYC taxi data and training machine learning models to predict taxi fares.

## Project Overview

This project provides an end-to-end solution for processing large-scale NYC taxi datasets from S3, transforming the data, and training machine learning models to predict taxi fares. The pipeline is designed to handle large datasets efficiently using techniques like chunked processing, parallel execution, and memory management with Dask.

## Features

- **Efficient Data Processing**: 
  - Processes large S3 datasets in chunks with parallel execution
  - Memory-efficient processing using Dask for distributed computing
  - Checkpointing mechanism to resume processing from interruptions
  - Comprehensive logging and timing reports

- **Data Transformation**:
  - Feature engineering including time-based features
  - Data cleaning and preprocessing
  - Train/validation split
  - Visualization and statistical analysis

- **Model Training**:
  - XGBoost regression model for fare prediction
  - Model evaluation with RMSE and R² metrics
  - Feature importance analysis
  - Model persistence

## Project Structure

```
├── config.py               # Configuration parameters
├── data_processing.py      # Data ingestion and preprocessing pipeline
├── model_training.py       # XGBoost model training
├── output/                 # Generated outputs
│   ├── logs/               # Processing logs
│   ├── visualizations/     # Data visualizations
│   ├── statistics/         # Data statistics
│   ├── checkpoints/        # Processing checkpoints
│   └── temp/               # Temporary files for processing
```

## Requirements

- Python 3.7+
- AWS credentials configured
- Required Python packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - boto3
  - dask
  - pyarrow
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn
  - tqdm

## Setup

1. Configure AWS credentials:
```bash
aws configure
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Adjust configuration in `config.py` if needed.

## Usage

### Data Processing

Run the data processing pipeline to fetch and transform NYC taxi data:

```bash
python data_processing.py
```

This will:
1. Fetch NYC taxi ride data from S3
2. Process and transform the data
3. Split into training and validation sets
4. Save processed data back to S3

Options:
- Testing mode: Set `TESTING_MODE = True` in the code to process only a subset of files
- Adjust batch sizes and workers based on available resources

### Model Training

After data processing, train the XGBoost model:

```bash
python model_training.py
```

This will:
1. Load processed data from S3
2. Train an XGBoost regression model
3. Evaluate model performance
4. Save the model and feature information to S3

## Performance Considerations

- The pipeline is designed to handle large datasets efficiently through chunked processing
- Memory usage is actively managed through Dask and garbage collection
- Disk space is monitored and managed during processing
- Processing is parallelized across multiple workers

## Customization

- Adjust chunk sizes, batch sizes, and worker counts based on your hardware capabilities
- Modify feature engineering steps in `prepare_features()` function
- Tune XGBoost hyperparameters in `train_xgboost_model()` function

## License

[License information]