import argparse
import hashlib
import os
import struct
import subprocess
import sys

import pandas as pd
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS, TAGS


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

IMAGE_FOLDER = os.path.join(_REPO_ROOT, "data", "images")
SHEILA_FOLDER = os.path.join(_REPO_ROOT, "data", "sheila")
CONVERTED_IMAGE_FOLDER = os.path.join(_REPO_ROOT, "data", "images_converted")
OUTPUT_CSV = os.path.join(_THIS_DIR, "metadata.csv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
CONVERT_HEIC_TO_IMAGE = True


def _rel_repo(abs_path: str) -> str:
    """Path relative to repo root with forward slashes (matches metadata.csv)."""
    rel = os.path.relpath(os.path.abspath(abs_path), _REPO_ROOT)
    return rel.replace("\\", "/")


def _ratio_to_float(value) -> float:
    if isinstance(value, tuple):
        return float(value[0]) / float(value[1])
    return float(value)


def _degrees(value) -> float:
    d, m, s = value
    return _ratio_to_float(d) + _ratio_to_float(m) / 60.0 + _ratio_to_float(s) / 3600.0


def get_gps_from_exif(exif_data):
    if not exif_data:
        return None, None

    gps_info = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == "GPSInfo":
            for gps_tag, gps_value in value.items():
                gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                gps_info[gps_tag_name] = gps_value

    if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
        return None, None

    lat = _degrees(gps_info["GPSLatitude"])
    lon = _degrees(gps_info["GPSLongitude"])

    if gps_info.get("GPSLatitudeRef") == "S":
        lat = -lat
    if gps_info.get("GPSLongitudeRef") == "W":
        lon = -lon

    return lat, lon


def get_gps_from_apple_location_xattr(path: str):
    """
    iPhone HEIC files copied through Photos/AirDrop may store location in the
    Apple extended attribute com.apple.assetsd.customLocation. The first two
    little-endian doubles are latitude and longitude.
    """
    try:
        result = subprocess.run(
            ["xattr", "-px", "com.apple.assetsd.customLocation", path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None

    hex_text = "".join(result.stdout.split())
    if len(hex_text) < 32:
        return None, None

    try:
        raw = bytes.fromhex(hex_text)
        lat, lon = struct.unpack("<dd", raw[:16])
    except (ValueError, struct.error):
        return None, None

    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return lat, lon
    return None, None


def get_gps_from_mdls(path: str):
    """macOS Spotlight: HEIC often has kMDItemLatitude / kMDItemLongitude when
    Pillow cannot read GPS EXIF (common for iPhone HEIC in git checkouts).
    """
    if sys.platform != "darwin":
        return None, None
    try:
        result = subprocess.run(
            ["mdls", "-name", "kMDItemLatitude", "-name", "kMDItemLongitude", path],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, None

    lat = lon = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("kMDItemLatitude"):
            try:
                lat = float(line.split("=", 1)[1].strip())
            except (IndexError, ValueError):
                pass
        elif line.startswith("kMDItemLongitude"):
            try:
                lon = float(line.split("=", 1)[1].strip())
            except (IndexError, ValueError):
                pass

    if lat is None or lon is None:
        return None, None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None
    return lat, lon


def extract_gps(path: str):
    try:
        with Image.open(path) as img:
            lat, lon = get_gps_from_exif(img.getexif())
            if lat is not None and lon is not None:
                return lat, lon, "exif"
    except UnidentifiedImageError as exc:
        image_error = f"{type(exc).__name__}: Pillow cannot decode this file"
    except Exception as exc:
        image_error = f"{type(exc).__name__}: {exc}"
    else:
        image_error = "no GPS EXIF found"

    lat, lon = get_gps_from_mdls(path)
    if lat is not None and lon is not None:
        return lat, lon, "mdls"

    lat, lon = get_gps_from_apple_location_xattr(path)
    if lat is not None and lon is not None:
        return lat, lon, "apple_xattr"

    print(f"No GPS: {os.path.basename(path)} ({image_error})")
    return None, None, None


def image_path_for_training(path: str, *, disambiguator: str | None = None) -> str:
    pl = path.lower()
    if not CONVERT_HEIC_TO_IMAGE or not (pl.endswith(".heic") or pl.endswith(".heif")):
        return path

    os.makedirs(CONVERTED_IMAGE_FOLDER, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    target_stem = f"{stem}_{disambiguator}" if disambiguator else stem
    output_path = os.path.join(CONVERTED_IMAGE_FOLDER, f"{target_stem}.png")
    if os.path.exists(output_path):
        return output_path

    try:
        subprocess.run(
            ["qlmanage", "-t", "-s", "1024", "-o", CONVERTED_IMAGE_FOLDER, path],
            check=True,
            capture_output=True,
            text=True,
        )
        qlmanage_path = os.path.join(CONVERTED_IMAGE_FOLDER, f"{os.path.basename(path)}.png")
        if os.path.exists(qlmanage_path):
            os.replace(qlmanage_path, output_path)
        return output_path
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Could not convert {os.path.basename(path)} to PNG ({exc}); using original path.")
        return path


def main_default() -> None:
    data = []
    for filename in sorted(os.listdir(IMAGE_FOLDER)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        filepath = os.path.join(IMAGE_FOLDER, filename)
        lat, lon, source = extract_gps(filepath)
        if lat is not None and lon is not None:
            training_path = image_path_for_training(filepath)
            print(f"GPS: {filename} -> {lat:.8f}, {lon:.8f} ({source})")
            data.append([_rel_repo(training_path), lat, lon])

    df = pd.DataFrame(data, columns=["image_path", "latitude", "longitude"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Extracted {len(df)} GPS locations to {OUTPUT_CSV}.")


def ingest_folder(folder: str, *, append: bool) -> None:
    """Extract GPS from every image in ``folder``, convert HEIC -> PNG under
    ``data/images_converted``, merge into ``metadata.csv`` when ``append``.

    Prints counts: eligible files, rows with GPS written, skipped (no GPS).
    """
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Not a directory: {folder}")

    # iPhone IMG_xxxx numbering recurs across shoots, so two ingest folders
    # often produce HEIC stems that collide with PNGs from earlier waves.
    # Tag every PNG with a short hash of the source folder's absolute path:
    # same folder -> same suffix (idempotent re-runs), different folder ->
    # different suffix (no silent overwrites of unrelated photos).
    disambiguator = hashlib.sha1(folder.encode("utf-8")).hexdigest()[:8]

    eligible = 0
    rows: list[list] = []
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            continue
        eligible += 1
        lat, lon, source = extract_gps(filepath)
        if lat is None:
            continue
        training_path = image_path_for_training(filepath, disambiguator=disambiguator)
        rel = _rel_repo(training_path)
        print(f"GPS: {filename} -> {lat:.8f}, {lon:.8f} ({source}) -> {rel}")
        rows.append([rel, lat, lon])

    no_gps = eligible - len(rows)
    new_df = pd.DataFrame(rows, columns=["image_path", "latitude", "longitude"])

    if append and os.path.exists(OUTPUT_CSV):
        old = pd.read_csv(OUTPUT_CSV)
        out = pd.concat([old, new_df], ignore_index=True)
        out.drop_duplicates(subset=["image_path"], keep="last", inplace=True)
        out.sort_values("image_path", kind="mergesort").reset_index(drop=True, inplace=True)
    else:
        out = new_df.sort_values("image_path", kind="mergesort").reset_index(drop=True)

    out.to_csv(OUTPUT_CSV, index=False)
    print(
        f"\nDone. Folder {folder}: {eligible} image files, "
        f"{len(rows)} with GPS, {no_gps} skipped (no GPS). "
        f"metadata.csv now has {len(out)} rows -> {OUTPUT_CSV}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Img2GPS/metadata.csv from EXIF / Apple xattr GPS.")
    parser.add_argument(
        "--ingest",
        metavar="FOLDER",
        help=f"Also scan this folder (e.g. {_rel_repo(SHEILA_FOLDER)}). "
        "Converts HEIC like the default pipeline and merges into metadata.csv.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="With --ingest: only write rows from that folder (replace entire CSV). Default is append/merge.",
    )
    args = parser.parse_args()

    if args.ingest:
        ingest_folder(args.ingest, append=not args.ingest_only)
        return
    main_default()


if __name__ == "__main__":
    main()
