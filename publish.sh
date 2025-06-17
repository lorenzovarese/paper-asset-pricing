#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Define all possible packages
ALL_PACKAGES=("paper-data" "paper-model" "paper-portfolio" "paper-asset-pricing")
# This will hold the packages we actually want to publish
PACKAGES_TO_PUBLISH=()

# --- Helper Functions ---
function show_help() {
  echo "Usage: $0 [FLAGS]"
  echo "Build and publish selected P.A.P.E.R. packages to PyPI."
  echo ""
  echo "Flags:"
  echo "  --all         Build and publish all packages."
  echo "  --data        Build and publish paper-data."
  echo "  --model       Build and publish paper-model."
  echo "  --portfolio   Build and publish paper-portfolio."
  echo "  --cli         Build and publish paper-asset-pricing (the CLI)."
  echo "  -h, --help    Show this help message."
  echo ""
}

# --- Argument Parsing ---
# If no arguments are provided, show help and exit.
if [ $# -eq 0 ]; then
  show_help
  exit 0
fi

# Parse command-line arguments
for arg in "$@"; do
  case $arg in
    --all)
      PACKAGES_TO_PUBLISH=("${ALL_PACKAGES[@]}")
      # The --all flag overrides any other selections, so we can stop parsing.
      break
      ;;
    --data)
      PACKAGES_TO_PUBLISH+=("paper-data")
      ;;
    --model)
      PACKAGES_TO_PUBLISH+=("paper-model")
      ;;
    --portfolio)
      PACKAGES_TO_PUBLISH+=("paper-portfolio")
      ;;
    --cli)
      PACKAGES_TO_PUBLISH+=("paper-asset-pricing")
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$arg'"
      show_help
      exit 1
      ;;
  esac
done

# Ensure there are unique packages to publish
# This handles cases where a user might specify the same flag twice.
IFS=" " read -r -a PACKAGES_TO_PUBLISH <<< "$(echo "${PACKAGES_TO_PUBLISH[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' ')"

if [ ${#PACKAGES_TO_PUBLISH[@]} -eq 0 ]; then
    echo "No packages selected to publish. Use --help for options."
    exit 0
fi

# --- Main Logic ---
# Clean the root dist directory before starting
echo "--- Cleaning old builds ---"
rm -rf dist
mkdir dist

# Loop through each selected package
for PKG in "${PACKAGES_TO_PUBLISH[@]}"; do
  echo "--- Processing package: $PKG ---"

  # Check if the directory exists before trying to enter it
  if [ ! -d "$PKG" ]; then
    echo "Error: Directory $PKG not found! Skipping."
    continue
  fi

  cd "$PKG"

  # Build the package. The output will go to the root dist/ directory.
  echo "Building $PKG..."
  uv build --out-dir ../dist

  cd ..
  echo "--- Finished package: $PKG ---"
  echo ""
done

# Check if any packages were actually built
if [ -z "$(ls -A dist)" ]; then
    echo "No packages were built. Nothing to upload."
    exit 0
fi

# After all selected packages are built, upload them
echo "--- Uploading packages to PyPI ---"
echo "The following packages will be uploaded:"
ls -1 dist
echo ""

# Uncomment the following line to actually upload
# twine upload dist/*

# For demonstration, we'll just list them.
# To enable uploading, remove the 'echo' and uncomment the 'twine' line.
echo "Dry run: Would upload with 'twine upload dist/*'"


echo "All selected packages processed."
