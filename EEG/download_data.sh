#!/bin/bash

set -e

BASE_URL="https://physionet.org/files/ucddb/1.0.0"
DATA_DIR="data/raw/ucd"

mkdir -p "$DATA_DIR"

SUBJECTS=(
    "002"
    "003"
    "006"
    "010"
    "014"
)

for SUBJECT in "${SUBJECTS[@]}"
do
    echo "Downloading subject $SUBJECT..."

    curl -L --fail \
        "$BASE_URL/ucddb${SUBJECT}.rec" \
        -o "$DATA_DIR/ucddb${SUBJECT}.rec"

    curl -L --fail \
        "$BASE_URL/ucddb${SUBJECT}_respevt.txt" \
        -o "$DATA_DIR/ucddb${SUBJECT}_respevt.txt"
done

echo ""
echo "Finished downloading UCD data."