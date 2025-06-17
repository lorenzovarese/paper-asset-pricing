import pandas as pd

# Define the input and output filenames
input_filename = 'portfolios/additional_datasets/risk_free_rate_1956_2021.csv'
output_filename = 'portfolios/additional_datasets/monthly_risk_free_rates_2.csv'

# --- Main Script ---

try:
    # 1. Load the data from the CSV file into a pandas DataFrame
    print(f"Reading data from '{input_filename}'...")
    df = pd.read_csv(input_filename)

    # 2. Rename the 'rf' column to 'annual_rf' for better readability
    df.rename(columns={'rf': 'annual_rf'}, inplace=True)

    # 3. Calculate the monthly risk-free rate
    # The formula to convert an annual rate to a monthly rate, accounting for compounding, is:
    # monthly_rf = (1 + annual_rf)^(1/12) - 1
    df['monthly_rf'] = (1 + df['annual_rf'])**(1/12) - 1

    # Drop the 'annual_rf' column as we only need the monthly rate
    df.drop(columns=['annual_rf'], inplace=True)

    # 4. Rename the 'monthly_rf' column to 'rf' for consistency with the expected output
    df.rename(columns={'monthly_rf': 'rf'}, inplace=True)

    # 4. Select and reorder columns for the final output
    output_df = df[['date', 'rf']]

    # 5. Save the results to a new CSV file
    # We use index=False to avoid writing the pandas DataFrame index as a column
    output_df.to_csv(output_filename, index=False)

    print(f"\nSuccessfully processed the data.")
    print(f"Output saved to '{output_filename}'")

    # Display the first few rows of the new data for verification
    print("\n--- First 5 rows of the output data: ---")
    print(output_df.head())

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script, or provide the full path.")
except Exception as e:
    print(f"An error occurred: {e}")