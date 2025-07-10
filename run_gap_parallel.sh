set -e

MANIFEST="$1"
STAMP="$2"
N_PARALLEL="${3:-8}"

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

rm -f *.g.out

echo "GAP parallel run complete. Output: gap_output_${STAMP}.txt"