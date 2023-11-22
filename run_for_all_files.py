# Author: James Fraser
# Last Modified: 2023-11-21
# Description: This script runs the step_calculator.py script on all .trc files in a directory.
import os
import subprocess
import argparse

# Define a function to load command-line arguments
def load_arguments():
    parser = argparse.ArgumentParser(description="Run the step_calculator.py script on all .trc files in a directory")

    # Add the directory argument
    #parser.add_argument("--directory", default=os.getcwd(), help="Specify the directory (default: current directory)")
    parser.add_argument("--directory", default="./trc_files/", help="Defaults to ../trc_files or current directory if ../trc_files does not exist")

    args = parser.parse_args()
    return args

def main():
    # Load command-line arguments
    args = load_arguments()

    # Access the directory argument
    target_directory = args.directory

    # Your program logic goes here
    print(f"Working with directory: {target_directory}")

    if not os.path.isdir(target_directory):
        print(f"Error: {target_directory} is not a directory.")
        return

    # Get a list of all files in the current directory
    files = os.listdir(target_directory)

    # Filter out the .trc files
    trc_files = [file for file in files if file.endswith('.trc')]

    # If there are no .trc files, exit the program
    if len(trc_files) == 0:
        print("Error: no .trc files found in the directory.")
        return

    # Run the step_calculator.py script on each .trc file
    for trc_file in trc_files:

        print(f"Running step_calculator.py on {trc_file}...")

        # Combine the directory and file name to get the full path
        trc_file = os.path.join(target_directory, trc_file)

        # Construct the command to run the script
        command = f'python step_calculator.py --input_file {trc_file}'

        # Alternative command if you want to specificy the start and end times
        # command = f'python step_calculator.py --input_file {trc_file} --start_frame 10 --end_frame 20'

        # Aternative command if you want to specificy the left and right foot markers
        #command = f'python step_calculator.py --input_file {trc_file} --left_markers LHEE_X42 LHEE_Z42 --right_markers RHEE_X42, RLHEE_Z35 --start_frame 10 --end_frame 20'
        
        # Run the script using subprocess
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()