#!/bin/bash
# HYPNOS EEG model: download open sleep-apnea EEG datasets into data/raw/
# Safe to re-run: finished files are skipped, partial downloads resume.
# PhysioNet files come from PhysioNet's AWS S3 open-data mirror (much faster), falling back to physionet.org.
cd "$(dirname "$0")" || exit 1
RAW="data/raw"; LOG="$RAW/download_log.txt"; mkdir -p "$RAW"
exec > >(tee -a "$LOG") 2>&1
echo "=== HYPNOS dataset download (v2, parallel) started $(date) ==="

fetch () {  # url [fallback-url] dest
  local url="$1" alt="$2" dest="$3"
  if [ -s "$dest" ] && [ ! -f "$dest.part" ]; then echo "  skip (have) $(basename "$dest")"; return 0; fi
  echo "  get  $(basename "$dest")"; touch "$dest.part"
  if curl -L --fail --retry 5 --retry-delay 5 -C - -sS "$url" -o "$dest" \
     || { [ -n "$alt" ] && curl -L --fail --retry 5 --retry-delay 5 -C - -sS "$alt" -o "$dest"; }; then
    rm -f "$dest.part"; echo "  done $(basename "$dest")"
  else echo "  !! FAILED $url"; fi
}
export -f fetch

physionet_db () {  # db version destdir [exclude-regex]
  local web="https://physionet.org/files/$1/$2" s3="https://physionet-open.s3.amazonaws.com/$1/$2"
  mkdir -p "$3"; echo "--- PhysioNet $1 -> $3"
  curl -sS -L --fail "$web/" | grep -o 'href="[^"/?#]*"' | sed 's/href="//;s/"$//' | sort -u \
   | grep -v -E '^(index\.html)?$' | { if [ -n "$4" ]; then grep -v -E "$4"; else cat; fi; } \
   | while read -r f; do printf '%s\0%s\0%s\0' "$s3/$f" "$web/$f" "$3/$f"; done \
   | xargs -0 -n 3 -P 6 bash -c 'fetch "$1" "$2" "$3"' _
}

# 1) UCDDB: 25 PSGs, C3-A2 + C4-A1 EEG, respiratory events + sleep stages (skips Holter ECG _lifecard.edf)
physionet_db ucddb 1.0.0 "$RAW/ucd" '_lifecard\.edf$'
# 2) MIT-BIH Polysomnographic DB: 18 records, EEG + sleep/apnea annotations
physionet_db slpdb 1.0.0 "$RAW/mitbih_slpdb"
# 3) DREAMS Apnea DB (Zenodo): 12 whole-night PSGs with expert-scored apneas/hypopneas (~750 MB)
D="$RAW/dreams_apnea"; mkdir -p "$D"; echo "--- DREAMS Apnea -> $D"
Z="https://zenodo.org/records/2650142/files"
fetch "$Z/readme.txt?download=1" "" "$D/readme.txt"
fetch "$Z/Dreams%20License.txt?download=1" "" "$D/Dreams_License.txt"
fetch "$Z/DatabaseApnea.rar?download=1" "" "$D/DatabaseApnea.rar"
if [ -s "$D/DatabaseApnea.rar" ] && [ ! -f "$D/DatabaseApnea.rar.part" ] && ! ls "$D" | grep -qi '\.edf$'; then
  echo "  extracting DatabaseApnea.rar"
  (cd "$D" && tar -xf DatabaseApnea.rar) || echo "  !! couldn't extract the .rar; run: brew install unar && unar $D/DatabaseApnea.rar"
fi

echo; echo "=== Done $(date) ==="; du -sh "$RAW"/* 2>/dev/null
echo "Failures this run are marked '!! FAILED' above. Re-run this file to retry anything missing."
echo "You can close this window."
