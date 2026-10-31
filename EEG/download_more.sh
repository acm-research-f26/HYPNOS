#!/bin/bash

BASE_URL="https://physionet.org/files/ucddb/1.0.0"
DATA_DIR="data/raw/ucd"

mkdir -p "$DATA_DIR"

download_subject () {
    SUBJECT=$1

    echo "Downloading $SUBJECT"

    curl -sS -L --fail \
        "$BASE_URL/ucddb${SUBJECT}.rec" \
        -o "$DATA_DIR/ucddb${SUBJECT}.rec"

    curl -sS -L --fail \
        "$BASE_URL/ucddb${SUBJECT}_respevt.txt" \
        -o "$DATA_DIR/ucddb${SUBJECT}_respevt.txt"

    echo "Finished $SUBJECT"
}

export -f download_subject
export BASE_URL
export DATA_DIR

printf "%s\n" \
005 \
007 \
012 \
023 \
028 \
| xargs -P 5 -I {} bash -c 'download_subject "$@"' _ {}