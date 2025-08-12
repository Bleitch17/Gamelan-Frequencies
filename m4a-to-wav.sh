#!/bin/bash

# Set input and output directories
INPUT_DIR="m4a"
OUTPUT_DIR="wav"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Enable case-insensitive matching for .m4a extensions
shopt -s nocaseglob

# Loop through all .m4a files in the input directory
for file in "$INPUT_DIR"/*.m4a; do
    # Skip if no files are found
    [ -e "$file" ] || continue

    # Get the base filename without extension
    base=$(basename "$file")
    base="${base%.*}"

    # Convert to WAV
    ffmpeg -i "$file" -ar 44100 -ac 2 -sample_fmt s16 "$OUTPUT_DIR/${base}.wav"
done

# Disable case-insensitive matching
shopt -u nocaseglob
