#!/usr/bin/env python3
"""Helpers for fetching ArcGIS REST aerial backgrounds at render time.

`download_export` is used by `renderer.py` and `styled_renderer.py` in
this package; the module is also runnable as a CLI via `main()`.
"""
from math import pi, tan, log, atan, sinh
import argparse
import requests
from requests.adapters import HTTPAdapter, Retry

R = 6378137.0  # Web Mercator sphere

def ll_to_webmerc(lon, lat):
    # Clamp latitude to Web Mercator valid range
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = R * lon * pi / 180.0
    y = R * log(tan(pi/4.0 + (lat*pi/180.0)/2.0))
    return x, y

def webmerc_to_ll(x, y):
    lon = (x / R) * 180.0 / pi
    lat = (atan(sinh(y / R))) * 180.0 / pi
    return lon, lat

def make_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def build_export_url(
    service_export_url,
    lon, lat,
    buffer_x_m=500, buffer_y_m=None,
    width=2048, height=None,
    fmt="png",
    rotation=0,
    dpi=96,
    transparent=False,
    return_bbox4326=False
):
    """
    Build an ArcGIS REST /export URL centered at lon/lat with a buffer in meters.
    """
    if buffer_y_m is None:
        buffer_y_m = buffer_x_m
    if height is None:
        height = width

    # 1) Build bbox in EPSG:3857 (Web Mercator)
    cx, cy = ll_to_webmerc(lon, lat)
    xmin, ymin = cx - buffer_x_m, cy - buffer_y_m
    xmax, ymax = cx + buffer_x_m, cy + buffer_y_m

    # 2) Optionally provide bbox in 4326 by converting corners back
    if return_bbox4326:
        ll_min = webmerc_to_ll(xmin, ymin)
        ll_max = webmerc_to_ll(xmax, ymax)
        bbox_str = f"{ll_min[0]},{ll_min[1]},{ll_max[0]},{ll_max[1]}"
        bbox_sr = 4326
    else:
        bbox_str = f"{xmin},{ymin},{xmax},{ymax}"
        bbox_sr = 3857

    params = {
        "bbox": bbox_str,
        "bboxSR": bbox_sr,
        "imageSR": 3857,               # output CRS
        "size": f"{width},{height}",
        "format": fmt,
        "dpi": dpi,
        "transparent": "true" if transparent else "false",
        "rotation": rotation,
        "f": "image",                  # get raw image back
    }

    # Encode manually to keep it readable and avoid requests' %-encoding fuss here
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{service_export_url}?{query}"

def download_export(
    lon, lat,
    buffer_m=500, size=2048, fmt="png", rotation=0,
    out="image.png",
    service_export_url="https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/export",
    dpi=96, transparent=False,
    use_bbox4326=False,
    timeout=60,
    print_url=False
):
    url = build_export_url(
        service_export_url=service_export_url,
        lon=lon, lat=lat,
        buffer_x_m=buffer_m, buffer_y_m=buffer_m,
        width=size, height=size,
        fmt=fmt,
        rotation=rotation,
        dpi=dpi,
        transparent=transparent,
        return_bbox4326=use_bbox4326
    )
    if print_url:
        print("Export URL:", url)

    session = make_session()
    r = session.get(url, timeout=timeout)
    r.raise_for_status()

    # Basic content-type check
    ctype = r.headers.get("Content-Type", "")
    if "image" not in ctype.lower():
        # ArcGIS may return JSON/HTML if there’s an error; keep it for debugging
        debug_path = out + ".debug.txt"
        with open(debug_path, "wb") as f:
            f.write(r.content)
        raise RuntimeError(
            f"Expected image, got Content-Type={ctype}. "
            f"Saved response to {debug_path}"
        )

    with open(out, "wb") as f:
        f.write(r.content)
    return out

def main():
    p = argparse.ArgumentParser(
        description="Download a clipped image from an ArcGIS REST MapServer /export endpoint centered at a lat/long."
    )
    p.add_argument("--lon", type=float, required=True, help="Longitude (°)")
    p.add_argument("--lat", type=float, required=True, help="Latitude (°)")
    p.add_argument("--buffer-m", type=float, default=500, help="Half-width/height of AOI (meters).")
    p.add_argument("--size", type=int, default=2048, help="Image width and height (pixels).")
    p.add_argument("--fmt", default="png", help="Output format: png, jpg, tif, etc.")
    p.add_argument("--rotation", type=float, default=0, help="Rotation in degrees (clockwise).")
    p.add_argument("--dpi", type=int, default=100, help="Requested DPI.")
    p.add_argument("--transparent", action="store_true", help="Request a transparent background if format supports it.")
    p.add_argument("--use-bbox4326", action="store_true",
                   help="Send bbox in EPSG:4326 instead of 3857 (server will reproject).")
    p.add_argument("--out", default="image.png", help="Output file name.")
    p.add_argument("--print-url", action="store_true", help="Print the final export URL.")
    p.add_argument(
        "--service-export-url",
        default="https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/export",
        help="ArcGIS REST /export endpoint."
    )

    args = p.parse_args()

    path = download_export(
        lon=args.lon,
        lat=args.lat,
        buffer_m=args.buffer_m,
        size=args.size,
        fmt=args.fmt,
        rotation=args.rotation,
        out=args.out,
        service_export_url=args.service_export_url,
        dpi=args.dpi,
        transparent=args.transparent,
        use_bbox4326=args.use_bbox4326,
        print_url=args.print_url
    )
    print("Saved:", path)

if __name__ == "__main__":
    main()


'''python download_satellite_images.py \
  --lon -121.945800 --lat 37.404551 \
  --buffer-m 50 --size 2048 --fmt png --rotation 30 \
  --out santa_clara.png \
  --service-export-url "https://geo.sanjoseca.gov/server/rest/services/Imagery/DPW_ImageryCached2024/MapServer/export" \
  --print-url

  python download_satellite_images.py --lon -121.945800 --lat 37.404551 --buffer-m 500 --size 2048 --fmt png --out dc_usgs.png --print-url

  
  '''