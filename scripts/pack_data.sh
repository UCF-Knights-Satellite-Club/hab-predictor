#!/usr/bin/env bash
# Pack data/ into a few shareable archives in out/share/ (layout mirrors data/).
#
#   scripts/pack_data.sh            # then upload out/share/ (see README, "Shared data")
#
# Unpack on another machine from the repo root:
#   cd data && for f in gfs/*.tar hrrr/*.tar *.tar *.tar.gz; do tar -xf "$f" && rm "$f"; done
set -euo pipefail
cd "$(dirname "$0")/../data"
out=../out/share
mkdir -p "$out/gfs" "$out/hrrr" "$out/masks" "$out/ne" "$out/nwi" "$out/osm"

# GFS: one tar per cycle month (files are already gzip-compressed netCDF)
for m in $(ls gfs | cut -c1-6 | sort -u); do
  f="$out/gfs/gfs_${m:0:4}-${m:4:2}.tar"
  [ -f "$f" ] || { tar -cf "$f.part" $(ls -d gfs/${m}*) && mv "$f.part" "$f"; }
done
[ -f "$out/hrrr/hrrr.tar" ] || { tar -cf "$out/hrrr/hrrr.tar.part" hrrr && mv "$out/hrrr/hrrr.tar.part" "$out/hrrr/hrrr.tar"; }
[ -f "$out/sondes.tar" ] || { tar -cf "$out/sondes.tar.part" sondes && mv "$out/sondes.tar.part" "$out/sondes.tar"; }
[ -f "$out/nlcd.tar.gz" ] || { tar -czf "$out/nlcd.tar.gz.part" nlcd && mv "$out/nlcd.tar.gz.part" "$out/nlcd.tar.gz"; }

# single files copied as-is
cp -u masks/*.npz "$out/masks/"
cp -u ne/*.zip "$out/ne/"
cp -u nwi/FL_geodatabase_wetlands.zip "$out/nwi/"
cp -u osm/florida-latest.osm.pbf "$out/osm/"
[ -f kaos1_spot.csv ] && cp -u kaos1_spot.csv "$out/"

(cd "$out" && find . -type f ! -name SHA256SUMS ! -name README.txt ! -name '*.part' -printf '%P\n' \
   | sort | xargs sha256sum > SHA256SUMS)
du -sh "$out"
