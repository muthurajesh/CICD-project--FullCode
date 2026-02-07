# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, help="Path to raw data")  # Specify the type for raw data (str)
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train data (str)
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test data (str)
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")  # Specify the type (float) and default value (0.2) for test-train ratio
    args = parser.parse_args()

    return args

def main(args):  # Write the function name for the main data preparation logic
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # ------- WRITE YOUR CODE HERE -------

    # Step 1: Perform label encoding to convert categorical features into numerical values for model compatibility.  
    # Step 2: Split the dataset into training and testing sets using train_test_split with specified test size and random state.  
    # Step 3: Save the training and testing datasets as CSV files in separate directories for easier access and organization.  
    # Step 4: Log the number of rows in the training and testing datasets as metrics for tracking and evaluation.  

    # Encoding the categorical 'Segment' column
    label_encoder = LabelEncoder()
    df['Segment'] = label_encoder.fit_transform(df['Segment'])

    # Log the first few rows of the dataframe
    logging.info(f"Transformed Data:\n{df.head()}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    # Log completion
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("test_size", len(test_df))

if __name__ == "__main__":
    # Start MLflow Run
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()  # Call the function to parse arguments

    lines = [
        f"Raw data path: {args.raw_data}",  # Print the raw_data path
        f"Train dataset output path: {args.train_data}",  # Print the train_data path
        f"Test dataset path: {args.test_data}",  # Print the test_data path
        f"Test-train ratio: {args.test_train_ratio}",  # Print the test_train_ratio
    ]

    for line in lines:
        print(line)
    
    main(args)

    # End MLflow Run
    mlflow.end_run()
