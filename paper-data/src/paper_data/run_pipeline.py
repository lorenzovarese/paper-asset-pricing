import sys
from pathlib import Path

# Add the src directory to the Python path to allow importing paper_data
# This is a common practice for running scripts outside the installed package
# For a proper installed package, this might not be strictly necessary if
# the package is installed in editable mode (`uv pip install -e .`)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from paper_data.manager import DataManager

if __name__ == "__main__":
    # Define the path to your data configuration file
    # This path is relative to your PAPER/ThesisExample project root
    config_file_name = "data-config.yaml"

    # Define the root of your PAPER project
    # This is where 'configs', 'data', 'logs.log' etc. are located
    # Adjust this path based on where your 'PAPER' directory is relative to where you run this script
    paper_project_root = Path("PAPER/ThesisExample")

    # Construct the full path to the config file
    data_config_path = paper_project_root / "configs" / config_file_name

    print(f"Attempting to load config from: {data_config_path.resolve()}")
    print(f"Using project root: {paper_project_root.resolve()}")

    try:
        manager = DataManager(config_path=data_config_path)
        processed_datasets = manager.run(project_root=paper_project_root)

        print("\n--- Final Processed Datasets ---")
        for name, df in processed_datasets.items():
            print(f"Dataset '{name}':")
            print(df.head())
            print("-" * 30)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the config file and data paths are correct.")
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except NotImplementedError as e:
        print(f"Feature Not Implemented: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
