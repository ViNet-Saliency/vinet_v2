#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <script_to_run> <output_filename>"
    exit 1
fi

SCRIPT_TO_RUN=$1
OUTPUT_FILE=$2

nohup "$SCRIPT_TO_RUN" > "$OUTPUT_FILE" 2>&1 &
