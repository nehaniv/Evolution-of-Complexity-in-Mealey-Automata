#!/bin/bash
# Usage: ./run_gap_parallel.sh <manifest_file> <timestamp> [N_PARALLEL]
# Example: ./run_gap_parallel.sh gap_manifest_20240610.txt 20240610 8

set -e

MANIFEST="$1"
STAMP="$2"
N_PARALLEL="${3:-8}"  # Default to 8 parallel jobs if not specified

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest file $MANIFEST not found!"
  exit 1
fi

echo "Running GAP in parallel on files listed in $MANIFEST with $N_PARALLEL jobs..."

# Run all .g files in parallel
cat "$MANIFEST" | xargs -n 1 -P "$N_PARALLEL" -I {} sh -c 'gap < "{}" > "{}.out"'

echo "Concatenating outputs in manifest order..."
> gap_output_${STAMP}.txt
while read gfile; do
    cat "${gfile}.out" >> gap_output_${STAMP}.txt
done < "$MANIFEST"

# Optionally, clean up per-run outputs
rm -f *.g.out

echo "GAP parallel run complete. Output: gap_output_${STAMP}.txt"