#!/bin/bash
# Reverse file content using pr, sort, and cut

if [ $# -ne 1 ]; then
    echo "Usage: $0 filename"
    exit 1
fi

pr -t -n "$1" | sort -nr | cut -f2-
