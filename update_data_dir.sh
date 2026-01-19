#!/bin/bash

# Absolute path where the script is executed
BASE_DIR="$(pwd)"
NEW_PATH="${BASE_DIR}/data"

SEARCH_DIR="experiments"

# Recursively find YAML files only inside experiments/
find "$SEARCH_DIR" -type f \( -name "*.yml" -o -name "*.yaml" \) | while read -r file; do
    sed -i -E "s|^([[:space:]]*data_dir:).*|\1 ${NEW_PATH}|g" "$file"
    echo "Updated: $file"
done