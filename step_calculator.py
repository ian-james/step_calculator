# Author: James Fraser
# Last Modified: 2023-11-21
# Description: This script reads a TRC file and calculates the step width and distance between two foot markers.
#              The script can be run from the command line using the following command:
#              python step_calculator.py --input_file ./trc_files/RDTO11_transformed.trc --output_file ./trc_files/RDTO11_transformed_modified.csv --left_markers LHEE_X42 LHEE_Z42 --right_markers RHEE_X35 RHEE_Z35 --calc_metrics STEP_LENGTH_X STEP_WIDTH_Z
# Step Calculator:

import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock

#Calculate the step width and distance between two foot markers using the specified distance function.
def calculate_step_metrics(marker_left, marker_right, distance_function):
    # Calculate the distance in 3D space using the specified distance function
    distance_3d = distance_function(marker_left, marker_right)

    # Calculate the step width using the distance function
    step_width = distance_function(marker_left[:2], marker_right[:2])

    return step_width, distance_3d


# This function reads the TRC file and returns a Pandas DataFrame.
def read_trc_file(file_path, delimiter='\t', header=None, skip_rows=0):
    """
    Opens a TRC file, reads it into a Pandas DataFrame, and returns the DataFrame.
    """
    try:
        # Read the TRC file into a Pandas DataFrame
        df = pd.read_csv(file_path, delimiter=delimiter,header=header, skiprows=skip_rows)

        return df

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file {file_path} is empty or contains no valid data.")
        return None

# This function renames the TRC file to a CSV file.
def rename_trc_to_csv(file_name, extra_text="_modified"):
    # Check if the file has a .trc extension
    if not file_name.endswith('.trc'):
        print("Invalid file extension. Please provide a file with .trc extension.")
        return
    
    # Generate the new CSV file name
    new_file_name = os.path.splitext(file_name)[0] + extra_text + '.csv'
    
    return new_file_name

# This function combines the names of the two row and check if the value exists in the first row.
def get_dataframe_category_columns(df):
    last = ""
    new_columns = []
    for column_value, next_value in zip(df.iloc[0], df.iloc[1]):
        column_value = str(column_value)

        if pd.isna(next_value):
            next_value = ""
        next_value = str(next_value)

        if pd.isna(column_value) or column_value == "nan":
            column_value = last
        else:
            last = column_value

        if next_value == "" or next_value is None:
            new_columns.append(column_value)
        else:
            new_columns.append(column_value + "_" + next_value)

    df.columns = new_columns
    df = df.drop([0,1])
    df = df.reset_index(drop=True)
    return df

# This function cleans the data frame by removing empty rows and converting the data types to float.
def clean_data_frame(df):
    # Delete empty rows and display the shape of the frame.    
    df.dropna(how='all', inplace=True)
    df = df.astype(float)    
    return df


# This function fills the missing values with the median of the closest two values.
def fill_missing_with_median(series):

    mask = series.isnull()
    
    # Iterate over the indices where values are missing
    i = 0
    while i < len(series):
        if mask[i]:
            # Find the closest non-null values before and after the missing value
            before = series[i-1::-1].dropna().iloc[0] if i > 0 else np.nan
            after = series[i:].dropna().iloc[0] if i < len(series) - 1 else np.nan
            
            # Fill the missing value with the median of the closest two values
            series[i] = np.nanmedian([before, after])
            
            # Move to the next non-null value
            i = i + 1 + (series[i:].notnull().idxmax() - i)
        else:
            i += 1

    if series.isnull().values.any() or series.isna().values.any():
        print("Series contains NaN values after filling missing values.")
    
    return series


# This function loads the arguments from the command line.
def load_arguments():
    try:
        # Create the argument parser
        parser = argparse.ArgumentParser(description="Your program description here.")

        # Add command-line arguments
        parser.add_argument("--input_file", help="Input file name", default="./trc_files/RDTO11_transformed.trc")    
        parser.add_argument("--output_file", help="Output file name")

        # Regex Names if we want all left and right heel markers
        # all_left_positions = "^LHEE"
        # all_right_positions = "^RHEE"

        # # Names if we want only the first left and right heel markers
        # left_heel_name = "LHEE_X42"
        # right_heel_name = "RHEE_X35"

        parser.add_argument("--left_markers", nargs='+', help="Left marker string", default=["LHEE_X42","LHEE_Z42"])
        parser.add_argument("--right_markers", nargs='+', help="Right marker string", default=["RHEE_X35","RHEE_Z35"])

        parser.add_argument("--delimiter", default='\t', help="Delimiter used in the TRC file")
        parser.add_argument("--skip_rows", default=3, help="Number of rows to skip in the TRC file")

        parser.add_argument("--distance_function", default="euclidean", help="Distance function to use")
        parser.add_argument("--calc_metrics", nargs='+', default=["STEP_LENGTH_X","STEP_HEIGHT_Y","STEP_WIDTH_Z"], help="Calculate step metrics")

        # Add arguments to allow only frames between a start and end frame to be processed in the analysis.
        parser.add_argument("--start_frame", default=0, help="Start frame to process")
        parser.add_argument("--end_frame", default=-1, help="End frame to process")

        # Add an argument to allow other columns to be included from the original data frame.
        parser.add_argument("--include_columns", nargs='+', default=["Frame#","Time"], help="Include columns from the original data frame")

        # Parse the command-line arguments
        args = parser.parse_args()

        return args
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
        return None
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        return None
    

def main():
    
    # File Location and Information    
    args = load_arguments()

    # Load all the arguments
    input_file = args.input_file
    output_file = args.output_file
    left_markers = args.left_markers
    right_markers = args.right_markers
    delimiter = args.delimiter
    skip_rows = args.skip_rows
    distance_function = args.distance_function
    calc_metrics = args.calc_metrics
    start_frame = int(args.start_frame)
    end_frame = int(args.end_frame)
    include_columns = args.include_columns

    print(f"Reading TRC file: {input_file}  with delimiter: {delimiter} and skipping rows: {skip_rows}")
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")        
        return


    # Read the TRC file into a Pandas DataFrame
    df = read_trc_file(input_file,delimiter,header=None, skip_rows=skip_rows)
    if df is None:
        print(f"Error reading TRC file: {input_file}")
        return
    
    # Get the columns that are categories
    print("Combining category names and cleaning the data frame.")
    df = get_dataframe_category_columns(df)

    # Combine the first two rows to get the column names
    # Check at least two rows exist
    if df.shape[0] < 2:
        print("Not enough rows to combine column names. Structure is not as expected.")
        return    
    
    df = clean_data_frame(df)

    # Remove the rows that are outside the start and end frame    
    if start_frame > 0:
        df = df[df["Frame#"] >= start_frame]
    if end_frame > 0:
        df = df[df["Frame#"] <= end_frame]
    df = df.reset_index(drop=True)


    key_columns_df = df[include_columns]
    # Iterate over the left and right markers
    for left_marker, right_marker, metric in zip(left_markers, right_markers, calc_metrics):
        print("\nSelecting the left and right heel markers and calculating the step metrics.")
        # Get the left and right heel markers
        left_cols = df.filter(regex=left_marker, axis=1) 
        right_cols = df.filter(regex=right_marker, axis=1)

        print("Cleaning the missing data in the columns.")    
        # Fill missing values with the median of the closest two values
        left_cols.transform(fill_missing_with_median, axis=0)
        right_cols.transform(fill_missing_with_median, axis=0)        

        hdf = pd.concat([left_cols, right_cols], axis=1)        
        
        # Calculate the distance between the two markers for each row
        print("Calculating the distance between the two markers for each row.")
        print(f"The markers are: {left_marker} and {right_marker}")
        print(f"Using the distance function: {distance_function}")
        distance_function = euclidean if distance_function == "euclidean" else cityblock    
        hdf[metric] = hdf.apply(lambda row: distance_function(row[left_cols.columns], row[right_cols.columns]), axis=1)

        # Replace the HDF columns in the original dataframe.
        df[left_cols.columns] = left_cols
        df[right_cols.columns] = right_cols
        df[metric] = hdf[metric]    

        key_columns_df = pd.concat([key_columns_df, hdf], axis=1)
        
    print("\nWriting the data frame to a CSV file.")        
    # Write the key columns that were selected
    renamed_file_name =  rename_trc_to_csv(input_file,"_key_columns")
    key_columns_df.to_csv(renamed_file_name, index=False)    
    print(f"Successfully wrote DataFrame to CSV file: {renamed_file_name}")

    # Save the summary to a CSV file
    summary = key_columns_df.describe()
    summary_file_name = rename_trc_to_csv(input_file,"_summary")
    summary.to_csv(summary_file_name)
    print(f"Successfully wrote DataFrame to CSV file: {summary_file_name}")

    # Write the original dataframe with the new columns to a CSV file
    renamed_file_name =  output_file if output_file is not None else rename_trc_to_csv(input_file,"_modified")
    df.to_csv(renamed_file_name, index=False)
    print(f"Successfully wrote DataFrame to CSV file: {renamed_file_name}")
    
if __name__ == "__main__":
    main()