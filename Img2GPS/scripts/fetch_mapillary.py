"""Fetch street-level images of the Img2GPS test region from Mapillary.

Mapillary publishes its imagery under CC-BY-SA and explicitly permits use
for "improvement, training, and development of products, services, [...]
algorithms, datasets" (Terms of Use §12), so it is a legal source of
extra training data for this project. Google Street View is NOT — its
ToS §3(c)(vii) forbids "use [of] Google Maps Content to improve machine
learning and artificial intelligence models, including to train, test,
validate or fine-tune the models".

What this script does:

1. probe   — given a bounding box, reports how many Mapillary images
             exist there and prints a few sample IDs/coords. Use this
             before any download to gauge whether it's worth the effort.
2. download — saves the 2048-px thumbnails as PNGs and writes a CSV in
              the same `image_path,latitude,longitude` format as
              `Img2GPS/metadata.csv`, so the rest of the pipeline can
              consume the new data unchanged.

Usage:

    export MAPILLARY_TOKEN="MLY|<your client token>"

    # 1) Quick coverage check around the spec test region.
    python Img2GPS/scripts/fetch_mapillary.py probe \
        --bbox=-75.19297,39.95026,-75.18949,39.95291

    # 2) Download up to 200 images into data/mapillary/.
    python Img2GPS/scripts/fetch_mapillary.py download \
        --bbox=-75.19297,39.95026,-75.18949,39.95291 \
        --out=data/mapillary --limit=200

Get a free client token at https://www.mapillary.com/dashboard/developers
(create an app → "Client Token" — starts with `MLY|`).

Attribution: any model trained on Mapillary imagery should credit
"© Mapillary contributors" and any redistribution must respect CC-BY-SA.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Iterable, List, Tuple


GRAPH_URL = "https://graph.mapillary.com/images"

DEFAULT_FIELDS = "id,thumb_2048_url,computed_geometry,geometry,captured_at,is_pano"


def _http_get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"OAuth {token}",
            "User-Agent": "img2gps-mapillary-fetch/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_bytes(url: str) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "img2gps-mapillary-fetch/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _iter_images(
    token: str,
    bbox: str,
    fields: str = DEFAULT_FIELDS,
    page_size: int = 200,
    max_total: int = 5000,
) -> Iterable[dict]:
    """Yield image entries inside `bbox` (min_lon,min_lat,max_lon,max_lat).

    Mapillary requires bbox queries to span < 0.01° square. This script
    does not split big bboxes — keep it small (the Img2GPS region is
    well under that limit).
    """
    params = {
        "fields": fields,
        "bbox": bbox,
        "limit": str(page_size),
    }
    url = f"{GRAPH_URL}?{urllib.parse.urlencode(params)}"
    seen = 0
    while url and seen < max_total:
        payload = _http_get_json(url, token)
        for item in payload.get("data", []):
            yield item
            seen += 1
            if seen >= max_total:
                return
        # Mapillary returns paging.cursors.after / paging.next.
        url = payload.get("paging", {}).get("next")


def _coords_of(entry: dict) -> Tuple[float, float] | None:
    """Prefer the map-matched (computed) geometry; fall back to raw."""
    for key in ("computed_geometry", "geometry"):
        geom = entry.get(key)
        if geom and geom.get("type") == "Point":
            lon, lat = geom["coordinates"][:2]
            return float(lat), float(lon)
    return None


def cmd_probe(token: str, bbox: str, sample: int) -> int:
    print(f"# probing bbox={bbox}")
    entries: List[dict] = []
    try:
        for i, entry in enumerate(_iter_images(token, bbox, page_size=200, max_total=sample * 5)):
            entries.append(entry)
            if i + 1 >= sample * 5:
                break
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.reason}", file=sys.stderr)
        return 2

    n = len(entries)
    pano = sum(1 for e in entries if e.get("is_pano"))
    print(f"# images found (capped at {sample*5}): {n}  (pano: {pano}, flat: {n - pano})")
    if not entries:
        print("# no Mapillary coverage in this box — try widening the bbox or use Flickr/Wikimedia")
        return 0
    print("# first samples (id  lat,lon  captured_at):")
    for entry in entries[:sample]:
        coords = _coords_of(entry)
        ts = entry.get("captured_at")
        if coords is None:
            continue
        print(f"  {entry['id']}  {coords[0]:.6f},{coords[1]:.6f}  {ts}")
    return 0


def cmd_download(
    token: str,
    bbox: str,
    out_dir: str,
    limit: int,
    skip_pano: bool,
    csv_path: str,
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    saved_rows: List[List[str]] = []
    fetched = 0
    skipped = 0

    # Paths in the CSV are written relative to the repo root (the parent of
    # the CSV's directory) so the resulting CSV concatenates cleanly with
    # `Img2GPS/metadata.csv`, which uses the same convention.
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(csv_path)) or ".", os.pardir)
    )

    for entry in _iter_images(token, bbox, page_size=200, max_total=limit * 5):
        if fetched >= limit:
            break
        if skip_pano and entry.get("is_pano"):
            skipped += 1
            continue
        coords = _coords_of(entry)
        url = entry.get("thumb_2048_url")
        if not (coords and url):
            skipped += 1
            continue

        image_id = entry["id"]
        out_path = os.path.join(out_dir, f"{image_id}.jpg")
        if not os.path.exists(out_path):
            try:
                blob = _http_get_bytes(url)
            except Exception as exc:  # noqa: BLE001
                print(f"! failed {image_id}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            with open(out_path, "wb") as fh:
                fh.write(blob)
            time.sleep(0.05)  # be polite

        rel_path = os.path.relpath(os.path.abspath(out_path), start=repo_root)
        saved_rows.append([rel_path, f"{coords[0]:.7f}", f"{coords[1]:.7f}"])
        fetched += 1
        if fetched % 25 == 0:
            print(f"  ... {fetched} downloaded")

    # Write/append CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["image_path", "latitude", "longitude"])
        w.writerows(saved_rows)

    print(f"done. fetched={fetched}  skipped={skipped}  csv={csv_path}")
    print("Remember: Mapillary imagery is © Mapillary contributors (CC-BY-SA).")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--bbox",
        required=True,
        help="min_lon,min_lat,max_lon,max_lat (Mapillary requires <0.01° square)",
    )
    common.add_argument(
        "--token",
        default=os.environ.get("MAPILLARY_TOKEN", ""),
        help="Mapillary client token (or env MAPILLARY_TOKEN)",
    )

    pp = sub.add_parser("probe", parents=[common], help="report coverage in a bbox")
    pp.add_argument("--sample", type=int, default=5)

    pd = sub.add_parser("download", parents=[common], help="download images + write CSV")
    pd.add_argument("--out", required=True, help="output directory for jpgs")
    pd.add_argument("--limit", type=int, default=200)
    pd.add_argument("--skip-pano", action="store_true", help="skip 360° panoramas")
    pd.add_argument(
        "--csv",
        default="Img2GPS/metadata_mapillary.csv",
        help="where to append image_path,latitude,longitude",
    )

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.token:
        print(
            "Set MAPILLARY_TOKEN or pass --token. Get one at "
            "https://www.mapillary.com/dashboard/developers",
            file=sys.stderr,
        )
        return 2

    if args.cmd == "probe":
        return cmd_probe(args.token, args.bbox, args.sample)
    if args.cmd == "download":
        return cmd_download(
            args.token, args.bbox, args.out, args.limit, args.skip_pano, args.csv
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
