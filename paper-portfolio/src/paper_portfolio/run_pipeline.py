import sys
from pathlib import Path
import logging
import argparse


# Add the src directory to the Python path to allow importing paper_portfolio
# This is crucial for running the script directly for development/testing.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_portfolio.manager import PortfolioManager
from paper_portfolio.config_parser import load_config

LOG_FILE_NAME = "logs.log"
LOG_LEVEL = "INFO"
DEFAULT_CONFIG_FILENAME = "portfolio-config.yaml"

# Get the root logger
root_logger = logging.getLogger()


def setup_logging(project_root: Path):
    """Configures logging for the pipeline run."""
    log_file_path = project_root / LOG_FILE_NAME
    root_logger.setLevel(LOG_LEVEL.upper())

    # Clear existing handlers
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


def main():
    """
    Main function to execute the portfolio evaluation pipeline.
    """
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Run the paper-portfolio pipeline for a specific project directory."
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

    # Set up logging to the project's log file
    setup_logging(paper_project_root)

    # Construct the full path to the config file
    portfolio_config_path = paper_project_root / "configs" / DEFAULT_CONFIG_FILENAME

    print(f"Using project root: {paper_project_root}")
    print(f"Attempting to load config from: {portfolio_config_path}")
    print(f"Detailed logs will be written to: {paper_project_root / LOG_FILE_NAME}")

    # --- 3. Run the Portfolio Pipeline ---
    try:
        root_logger.info("--- Starting Portfolio Pipeline Execution ---")
        root_logger.info(f"Config path: {portfolio_config_path}")
        root_logger.info(f"Project root: {paper_project_root}")

        portfolio_config = load_config(config_path=portfolio_config_path)
        manager = PortfolioManager(config=portfolio_config)
        manager.run(project_root=paper_project_root)

        print("\n✅ Portfolio pipeline completed successfully. See logs for details.")
        root_logger.info("--- Portfolio Pipeline Completed Successfully ---")

    except FileNotFoundError as e:
        error_msg = f"A required file was not found: {e}"
        root_logger.error(error_msg, exc_info=True)
        print(f"\n❌ ERROR: {error_msg}. Check logs for the full traceback.")
        sys.exit(1)
    except ValueError as e:
        error_msg = f"Configuration or data validation error: {e}"
        root_logger.error(error_msg, exc_info=True)
        print(f"\n❌ ERROR: {error_msg}. Check logs for the full traceback.")
        sys.exit(1)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        root_logger.exception(error_msg)
        print("\n❌ An unexpected error occurred. Check logs for the full traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
