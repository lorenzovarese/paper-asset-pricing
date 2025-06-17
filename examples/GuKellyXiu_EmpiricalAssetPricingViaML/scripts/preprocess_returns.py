import pandas as pd
import os

# --- Configuration ---
# Set the path to your input CSV file
input_csv_path = 'data/raw/crsp_returns_dec1956_dec2021_EOM.csv' 

# Set the path for the new, corrected output file
output_csv_path = 'data/raw/crsp_returns_dec1956_dec2021_EOM_decimal.csv'
# -------------------

def convert_returns_to_decimal(input_path, output_path):
    """
    Reads a CSV, converts the 'ret' column from percentage to decimal by 
    dividing by 100, and saves it to a new CSV file.
    """
    # Check if the input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    print(f"Reading data from '{input_path}'...")
    df = pd.read_csv(input_path)

    # Check if 'ret' column exists
    if 'ret' not in df.columns:
        print(f"Error: 'ret' column not found in the CSV file.")
        return

    print("\nOriginal data (first 5 rows):")
    print(df.head())
    print(f"\nOriginal 'ret' stats:\n{df['ret'].describe()}")

    # --- The core conversion step ---
    print("\nConverting 'ret' column from percentage to decimal (dividing by 100)...")
    df['ret'] = df['ret'] / 100.0
    # --------------------------------

    print("\nConverted data (first 5 rows):")
    print(df.head())
    print(f"\nNew 'ret' stats:\n{df['ret'].describe()}")

    # Save the corrected data to a new file
    print(f"\nSaving corrected data to '{output_path}'...")
    df.to_csv(output_path, index=False)

    print("\nConversion complete!")
    print(f"Please update your pipeline to use the new file: '{output_path}'")


if __name__ == "__main__":
    convert_returns_to_decimal(input_csv_path, output_csv_path)