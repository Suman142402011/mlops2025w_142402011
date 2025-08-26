#!/bin/bash
# Reverse a string

read -p "Enter a string: " str

# Method 1: Using rev command
reversed=$(echo "$str" | rev)

echo "Reversed string: $reversed"
