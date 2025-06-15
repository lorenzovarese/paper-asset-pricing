#!/bin/bash

# A list of your packages
PACKAGES=("paper-data" "paper-model" "paper-portfolio" "paper-tools")

# Clean the root dist directory before starting
echo "--- Cleaning old builds ---"
rm -rf dist

# Loop through each package
for PKG in "${PACKAGES[@]}"; do
  echo "--- Processing package: $PKG ---"
  
  # Navigate into the package directory
  cd "$PKG" || { echo "Directory $PKG not found!"; exit 1; }
  
  # Build the package. The output will go to the root dist/ directory.
  echo "Building $PKG..."
  # This command builds the package and places it in the root dist/ folder
  uv build --out-dir ../dist
  
  # Navigate back to the root directory
  cd ..
  echo "--- Finished package: $PKG ---"
  echo ""
done

# After all packages are built, upload them in one go
echo "--- Uploading all packages to PyPI ---"
twine upload dist/*

echo "All packages processed and uploaded."