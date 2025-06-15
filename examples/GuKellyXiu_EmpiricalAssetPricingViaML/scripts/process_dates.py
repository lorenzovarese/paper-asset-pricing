import pandas as pd
import argparse
import os


def adjust_dates_to_eom(input_filename, date_format):
    """
    Reads a CSV, shifts dates in the 'date' column to the end of the month,
    and saves the result to a new file.

    Args:
        input_filename (str): The path to the input CSV file.
        date_format (str): The format string for parsing the date column.
    """
    # Automatically generate the output filename by adding a suffix
    base_name, extension = os.path.splitext(input_filename)
    output_filename = f"{base_name}_EOM{extension}"

    print(f"Reading data from '{input_filename}'...")

    try:
        # Load the CSV file into a pandas DataFrame
        # low_memory=False can prevent dtype mixing warnings with large files.
        df = pd.read_csv(input_filename, low_memory=False)

        print(f"Processing the 'date' column using format '{date_format}'...")

        # 1. Convert the 'date' column to datetime objects using the specified format.
        #    Using `errors='coerce'` will turn any unparseable dates into NaT (Not a Time),
        #    which is safer than letting the script fail.
        original_dates = pd.to_datetime(df["date"], format=date_format, errors="coerce")

        # 2. Shift the dates to the end of the month.
        #    The .dt accessor provides access to datetime properties.
        #    `pd.offsets.MonthEnd(0)` is an offset that snaps a date to the end
        #    of its current month.
        end_of_month_dates = original_dates + pd.offsets.MonthEnd(0)

        # 3. Convert the datetime objects back to the YYYYMMDD integer format.
        #    This format is kept consistent for the output.
        df["date"] = end_of_month_dates.dt.strftime("%Y%m%d").astype(int)

        # 4. Save the modified DataFrame to a new CSV file.
        #    `index=False` prevents pandas from writing the DataFrame index as a column.
        print(f"Saving the updated data to '{output_filename}'...")
        df.to_csv(output_filename, index=False)

        print("\nProcessing complete!")
        print(
            f"The file '{output_filename}' has been created with dates adjusted to the end of the month."
        )

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except KeyError:
        print(
            f"Error: A column named 'date' was not found in '{input_filename}'. Please check the header."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Adjusts dates in a CSV file to the end of the month.",
        formatter_class=argparse.RawTextHelpFormatter,  # For better help text formatting
    )

    # Add the required input file argument
    parser.add_argument(
        "input_filename", help="The path to the input CSV file (e.g., datashare.csv)."
    )

    # Add the optional date format argument
    parser.add_argument(
        "--date_format",
        default="%Y%m%d",
        help="The format of the date column.\n"
        "Examples:\n"
        "  '%%Y%%m%%d' for integers like 20211231 (this is the default).\n"
        "  '%%Y-%%m-%%d' for strings like '2021-12-31'.",
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    adjust_dates_to_eom(args.input_filename, args.date_format)
