import sys
from pathlib import Path
import logging

# Add the src directory to the Python path to allow importing paper_data
# This is a common practice for running scripts outside the installed package
# For a proper installed package, this might not be strictly necessary if
# the package is installed in editable mode (`uv pip install -e .`)
# Adjusted path: Path(__file__).parent is 'paper_data', .parent again is 'src'
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_data.manager import DataManager

# Define constants for logging
LOG_FILE_NAME = "logs.log"  # Consistent with paper-tools
LOG_LEVEL = "INFO"  # Default logging level for file

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL.upper())

# Clear existing handlers to prevent duplicate output if run multiple times
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Define the path to your data configuration file
config_file_name = "data-config.yaml"

# Define the root of your PAPER project
# Assuming this script is run from the monorepo root, where 'ThesisExample' is a direct child.
# Adjust this path if your execution context is different.
paper_project_root = Path("tmp/ThesisExample")

# Construct the full path to the config file
data_config_path = paper_project_root / "configs" / config_file_name
log_file_path = paper_project_root / LOG_FILE_NAME

# Ensure the project root and log directory exist before configuring logging
paper_project_root.mkdir(parents=True, exist_ok=True)
log_file_path.parent.mkdir(parents=True, exist_ok=True)

# Configure FileHandler: All logs go to the file
file_handler = logging.FileHandler(log_file_path, mode="a")  # 'a' for append mode
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Configure StreamHandler: Only errors/critical messages go to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(
    logging.ERROR
)  # Set level to ERROR to suppress INFO/WARNING on console
console_formatter = logging.Formatter(
    "%(levelname)s: %(message)s"
)  # Simpler format for console errors
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Initial messages to console (not logged to file by default, as they are direct prints)
print(f"Attempting to load config from: {data_config_path.resolve()}")
print(f"Using project root: {paper_project_root.resolve()}")
print(f"Detailed logs will be written to: {log_file_path.resolve()}")


def main():
    try:
        # Log the start of the pipeline to the file
        root_logger.info("Starting data pipeline execution via run_pipeline.py script.")
        root_logger.info(f"Config path: {data_config_path.resolve()}")
        root_logger.info(f"Project root: {paper_project_root.resolve()}")

        manager = DataManager(config_path=data_config_path)
        processed_datasets = manager.run(project_root=paper_project_root)

        # Final success message to console
        print(
            f"\nData pipeline completed successfully. Additional information in '{log_file_path.resolve()}'"
        )
        root_logger.info("Data pipeline completed successfully.")

        # Log information about the final processed datasets to the file
        root_logger.info("\n--- Final Processed Datasets ---")
        for name, df in processed_datasets.items():
            root_logger.info(f"Dataset '{name}':")
            root_logger.info(f"  Shape: {df.shape}")
            root_logger.info(f"  Columns: {df.columns}")
            root_logger.info(f"Head {df.head()}")
            root_logger.info("-" * 30)

    except FileNotFoundError as e:
        root_logger.error(
            f"Error: {e}. Please ensure the config file and data paths are correct.",
            exc_info=True,
        )
        print(
            f"Error: A required file was not found. Check logs for details: '{log_file_path.resolve()}'"
        )
        sys.exit(1)  # Exit with error code
    except ValueError as e:
        root_logger.error(f"Configuration Error: {e}", exc_info=True)
        print(
            f"Error: Configuration issue. Check logs for details: '{log_file_path.resolve()}'"
        )
        sys.exit(1)
    except NotImplementedError as e:
        root_logger.error(f"Feature Not Implemented: {e}", exc_info=True)
        print(
            f"Error: Feature not implemented. Check logs for details: '{log_file_path.resolve()}'"
        )
        sys.exit(1)
    except Exception as e:
        root_logger.exception(
            f"An unexpected error occurred: {e}"
        )  # Use exception for full traceback
        print(
            f"An unexpected error occurred. Check logs for details: '{log_file_path.resolve()}'"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
