import os
import struct
import subprocess

import pandas as pd
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS, TAGS


IMAGE_FOLDER = "data/images"
CONVERTED_IMAGE_FOLDER = "data/images_converted"
OUTPUT_CSV = "metadata.csv"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic")
CONVERT_HEIC_TO_IMAGE = True


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

    lat, lon = get_gps_from_apple_location_xattr(path)
    if lat is not None and lon is not None:
        return lat, lon, "apple_xattr"

    print(f"No GPS: {os.path.basename(path)} ({image_error})")
    return None, None, None


def image_path_for_training(path: str) -> str:
    if not CONVERT_HEIC_TO_IMAGE or not path.lower().endswith(".heic"):
        return path

    os.makedirs(CONVERTED_IMAGE_FOLDER, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(CONVERTED_IMAGE_FOLDER, f"{stem}.png")
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


def main() -> None:
    data = []
    for filename in sorted(os.listdir(IMAGE_FOLDER)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        filepath = os.path.join(IMAGE_FOLDER, filename)
        lat, lon, source = extract_gps(filepath)
        if lat is not None and lon is not None:
            training_path = image_path_for_training(filepath)
            print(f"GPS: {filename} -> {lat:.8f}, {lon:.8f} ({source})")
            data.append([training_path, lat, lon])

    df = pd.DataFrame(data, columns=["image_path", "latitude", "longitude"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Extracted {len(df)} GPS locations to {OUTPUT_CSV}.")


if __name__ == "__main__":
    main()
