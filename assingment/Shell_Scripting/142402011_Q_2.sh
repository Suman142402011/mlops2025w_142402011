#!/bin/bash
# Traverse Documents directory and list contents

traverse_dir() {
    for item in "$1"/*; do
        [ -e "$item" ] || continue  # skip if no match
        if [ -d "$item" ]; then
            echo "Directory: $item"
            traverse_dir "$item"
        elif [ -f "$item" ]; then
            echo "File: $item"
        fi
    done
}

start_dir="$HOME/Documents"
traverse_dir "$start_dir"
