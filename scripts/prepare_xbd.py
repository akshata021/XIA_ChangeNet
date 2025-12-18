"""Prepare xBD/xView2 dataset into training-ready structure."""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare xBD/xView2 data for XAl-ChangeNet")
    parser.add_argument("--src", type=Path, required=True, help="Source root containing images/labels/targets")
    parser.add_argument("--out", type=Path, required=True, help="Destination root (e.g., data/xbd)")
    return parser.parse_args()


def ensure_structure(src: Path) -> dict:
    parts = {
        "images": src / "images",
        "labels": src / "labels",
        "targets": src / "targets",
    }
    for key, path in parts.items():
        if key == "targets" and not path.exists():
            continue
        if not path.exists():
            raise FileNotFoundError(f"Expected directory missing: {path}")
    return parts


def discover_pairs(images_dir: Path) -> dict:
    pairs = defaultdict(dict)
    for img_path in images_dir.glob("*.*"):
        name = img_path.stem
        if "_pre_disaster" in name:
            tile_id = name.replace("_pre_disaster", "")
            pairs[tile_id]["pre"] = img_path
        elif "_post_disaster" in name:
            tile_id = name.replace("_post_disaster", "")
            pairs[tile_id]["post"] = img_path
    return {k: v for k, v in pairs.items() if "pre" in v and "post" in v}

def resolve_label_path(labels_dir: Path, tile_id: str) -> Path | None:
    """Handle label files that keep the pre/post suffix."""
    candidates = [
        labels_dir / f"{tile_id}.json",
        labels_dir / f"{tile_id}_pre_disaster.json",
        labels_dir / f"{tile_id}_post_disaster.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_wkt_polygon(wkt: str):
    wkt = wkt.strip()
    if "POLYGON" not in wkt.upper():
        return []
    body = wkt[wkt.find("((") + 2 : wkt.rfind("))")]
    rings = []
    for ring_str in body.split("),("):
        points = []
        for pair in ring_str.split(","):
            parts = pair.strip().split()
            if len(parts) != 2:
                continue
            x, y = map(float, parts)
            points.append((x, y))
        if points:
            rings.append(points)
    return rings


def load_polygons(label_path: Path):
    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    polygons = []
    features = data.get("features", [])
    if isinstance(features, dict):
        xy_features = features.get("xy", [])
        for feature in xy_features:
            wkt = feature.get("wkt")
            if not wkt:
                continue
            rings = parse_wkt_polygon(wkt)
            if rings:
                polygons.append([rings[0]])
    else:
        for feature in features:
            if isinstance(feature, str):
                continue
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [])
            if geom.get("type") == "Polygon":
                polygons.append(coords)
            elif geom.get("type") == "MultiPolygon":
                polygons.extend(coords)
    return polygons, data


def rasterize(polygons, size):
    mask = Image.new("L", size, 0)
    drawer = ImageDraw.Draw(mask)
    for polygon in polygons:
        if not polygon:
            continue
        points = [(float(x), float(y)) for x, y in polygon[0]]
        if len(points) < 3:
            continue
        drawer.polygon(points, outline=1, fill=1)
    return mask


def save_pair(out_root: Path, event: str, tile_id: str, pre_path: Path, post_path: Path,
              label_path: Path, target_dir: Path | None):
    event_img_dir = out_root / "images" / event
    pre_dir = event_img_dir / "pre"
    post_dir = event_img_dir / "post"
    mask_dir = out_root / "masks" / event
    ann_dir = out_root / "annotations" / event
    for d in (pre_dir, post_dir, mask_dir, ann_dir):
        d.mkdir(parents=True, exist_ok=True)

    pre_out = pre_dir / pre_path.name
    post_out = post_dir / post_path.name
    shutil.copy2(pre_path, pre_out)
    shutil.copy2(post_path, post_out)

    label_out = ann_dir / label_path.name
    shutil.copy2(label_path, label_out)

    mask_out = mask_dir / f"{tile_id}.png"
    if target_dir:
        candidate = target_dir / f"{tile_id}.png"
    else:
        candidate = None

    if candidate and candidate.exists():
        shutil.copy2(candidate, mask_out)
    else:
        polygons, _ = load_polygons(label_path)
        with Image.open(pre_path) as img:
            mask = rasterize(polygons, img.size)
        mask = mask.point(lambda p: 255 if p > 0 else 0)
        mask.save(mask_out)

    record = {
        "event": event,
        "tile_id": tile_id,
        "pre_image": str(pre_out.relative_to(out_root)),
        "post_image": str(post_out.relative_to(out_root)),
        "mask": str(mask_out.relative_to(out_root)),
        "annotation": str(label_out.relative_to(out_root)),
    }
    return record


def main():
    args = parse_args()
    src_parts = ensure_structure(args.src)
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(src_parts["images"])
    if not pairs:
        raise RuntimeError("No pre/post disaster pairs detected. Check filenames.")

    events = defaultdict(list)
    for tile_id, assets in pairs.items():
        parts = tile_id.split("_")
        event = "_".join(parts[:-1]) if len(parts) > 1 else tile_id
        events[event].append((tile_id, assets))

    manifest_paths = []
    for event, entries in tqdm(events.items(), desc="Events"):
        records = []
        for tile_id, assets in tqdm(entries, leave=False, desc=f"{event}"):
            label_path = resolve_label_path(src_parts["labels"], tile_id)
            if label_path is None:
                continue
            record = save_pair(
                out_root=args.out,
                event=event,
                tile_id=tile_id,
                pre_path=assets["pre"],
                post_path=assets["post"],
                label_path=label_path,
                target_dir=src_parts["targets"] if src_parts["targets"].exists() else None,
            )
            records.append(record)

        manifest_path = args.out / f"pairs_{event}.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        manifest_paths.append(manifest_path)

    print(f"Prepared {len(pairs)} pairs across {len(events)} events.")
    for path in manifest_paths:
        print(f"Manifest written to: {path}")


if __name__ == "__main__":
    main()
