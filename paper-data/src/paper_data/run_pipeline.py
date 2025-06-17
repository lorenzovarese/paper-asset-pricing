import sys
from pathlib import Path
import logging
import argparse


sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_data.manager import DataManager
from paper_data.config_parser import load_config

# Define constants that do not depend on the project path
LOG_LEVEL = "INFO"
CONFIG_FILE_NAME = "data-config.yaml"
LOG_FILE_NAME = "logs.log"


def main():
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Run the paper-data pipeline for a specific project directory."
    )
    parser.add_argument(
        "project_root",
        type=str,
        help="The root directory of the PAPER project (e.g., 'tmp/MyPaperProjectName').",
    )
    args = parser.parse_args()

    # --- 2. Configure Paths and Logging (inside main) ---
    paper_project_root = Path(args.project_root).resolve()

    # Validate the provided path
    if not paper_project_root.is_dir():
        print(
            f"Error: The provided project path is not a valid directory: {paper_project_root}"
        )
        sys.exit(1)

    data_config_path = paper_project_root / "configs" / CONFIG_FILE_NAME
    log_file_path = paper_project_root / LOG_FILE_NAME

    # Ensure the log directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL.upper())

    # Clear existing handlers to prevent duplicate output
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure FileHandler
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Configure StreamHandler for console errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Initial messages to console
    print(f"Using project root: {paper_project_root}")
    print(f"Attempting to load config from: {data_config_path}")
    print(f"Detailed logs will be written to: {log_file_path}")

    # --- 3. Run the Data Pipeline ---
    try:
        root_logger.info("Starting data pipeline execution via run_pipeline.py script.")
        root_logger.info(f"Config path: {data_config_path}")
        root_logger.info(f"Project root: {paper_project_root}")

        data_config = load_config(config_path=data_config_path)
        manager = DataManager(config=data_config)
        processed_datasets = manager.run(project_root=paper_project_root)

        print(
            f"\nData pipeline completed successfully. Additional information in '{log_file_path}'"
        )
        root_logger.info("Data pipeline completed successfully.")

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
            f"Error: A required file was not found. Check logs for details: '{log_file_path}'"
        )
        sys.exit(1)
    except ValueError as e:
        root_logger.error(f"Configuration Error: {e}", exc_info=True)
        print(f"Error: Configuration issue. Check logs for details: '{log_file_path}'")
        sys.exit(1)
    except NotImplementedError as e:
        root_logger.error(f"Feature Not Implemented: {e}", exc_info=True)
        print(
            f"Error: Feature not implemented. Check logs for details: '{log_file_path}'"
        )
        sys.exit(1)
    except Exception as e:
        root_logger.exception(f"An unexpected error occurred: {e}")
        print(
            f"An unexpected error occurred. Check logs for details: '{log_file_path}'"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
