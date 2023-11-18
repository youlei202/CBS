#!/bin/bash

# Directory containing the files
dir="./scripts/study7"

# Loop through all the files in the directory
for file in "$dir"/*
do
  if [ -f "$file" ]; then
    echo "Submitting file: $file"
    bsub < "$file"
  fi
done

echo "All files submitted."

