#!/bin/sh
# Pruebas de geometria (Zones y Cover): no necesitan el juego. Uso: sh test/run.sh
set -e
cd "$(dirname "$0")/.."
OUT=$(mktemp -d)
javac -d "$OUT" src/safesupport/Zones.java src/safesupport/Cover.java test/ZonesCheck.java test/ZonesWall.java test/CoverCheck.java
for t in ZonesCheck ZonesWall CoverCheck; do
  echo "== $t"
  java -cp "$OUT" "$t"
done
