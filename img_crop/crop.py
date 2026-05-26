import os
import json
import cv2
import copy
from glob import glob


# ============================================================
# 1. 路径配置
# ============================================================
INPUT_DIR = "/home/myneml/Videos/Tool_change/val2026/"          # 原始图像和 json 所在目录
OUTPUT_DIR = "/home/myneml/Videos/Tool_change_crop/val2026/"    # 输出目录

OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSON_DIR = os.path.join(OUTPUT_DIR, "jsons")

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)


# ============================================================
# 2. 参数配置
# ============================================================
STATION_LABEL = "station"

CROP_HEIGHT_RATIO = 1.0 / 3.0

KEEP_STATION_LABEL = True
# True  : crop 后仍然保留 station 标注
# False : station 只用于确定 crop 位置，不写入新 json

MIN_BOX_W = 1.0
MIN_BOX_H = 1.0

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


# ============================================================
# 3. 工具函数
# ============================================================
def get_rect_from_points(points):
    """
    LabelMe rectangle 可能是：
    1. 两个点: [[x1,y1], [x2,y2]]
    2. 四个点: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    统一转成:
        x1, y1, x2, y2
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    return float(x1), float(y1), float(x2), float(y2)


def rect_to_4points(x1, y1, x2, y2):
    """
    输出四点矩形格式。
    """
    return [
        [float(x1), float(y1)],
        [float(x2), float(y1)],
        [float(x2), float(y2)],
        [float(x1), float(y2)]
    ]


def find_all_station_rects(json_data):
    """
    找到所有 station 标注框。

    返回:
        [
            (x1, y1, x2, y2),
            ...
        ]
    """
    station_rects = []

    for shape in json_data.get("shapes", []):
        if shape.get("label") != STATION_LABEL:
            continue

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        x1, y1, x2, y2 = get_rect_from_points(points)
        station_rects.append((x1, y1, x2, y2))

    return station_rects


def compute_crop_range_by_all_stations(image_h, station_rects):
    """
    根据所有 station 的联合区域计算 crop 范围。

    规则：
    1. 如果所有 station 的高度范围 <= 原图高度的 1/3：
        crop 高度固定为 image_h / 3
    2. 如果所有 station 的高度范围 > 原图高度的 1/3：
        crop 高度动态变成 station 的完整联合高度
        也就是完整保留所有 station 区域
    """

    base_crop_h = int(round(image_h * CROP_HEIGHT_RATIO))
    base_crop_h = max(1, min(base_crop_h, image_h))

    station_y1 = min(rect[1] for rect in station_rects)
    station_y2 = max(rect[3] for rect in station_rects)

    station_union_h = station_y2 - station_y1
    station_center_y = 0.5 * (station_y1 + station_y2)

    # ============================================================
    # 情况 1：station 联合区域可以被 1/3 crop 覆盖
    # ============================================================
    if station_union_h <= base_crop_h:
        crop_h = base_crop_h

        # 默认以所有 station 的联合中心为中心
        crop_y1 = int(round(station_center_y - crop_h / 2.0))
        crop_y2 = crop_y1 + crop_h

        # 修正：确保 station_y1 ~ station_y2 完整落在 crop 内
        if crop_y1 > station_y1:
            crop_y1 = int(round(station_y1))
            crop_y2 = crop_y1 + crop_h

        if crop_y2 < station_y2:
            crop_y2 = int(round(station_y2))
            crop_y1 = crop_y2 - crop_h

    # ============================================================
    # 情况 2：station 联合区域超过 1/3 高度
    # 不再强制 1/3，直接完整保留 station 覆盖区域
    # ============================================================
    else:
        crop_y1 = int(round(station_y1))
        crop_y2 = int(round(station_y2))

        # 避免由于 round 导致边界被截掉
        crop_y1 = int(station_y1 // 1)
        crop_y2 = int(station_y2 + 0.999999)

    # ============================================================
    # 图像边界保护
    # ============================================================
    crop_y1 = max(0, crop_y1)
    crop_y2 = min(image_h, crop_y2)

    # 防止异常空 crop
    if crop_y2 <= crop_y1:
        crop_y1 = 0
        crop_y2 = image_h

    crop_h = crop_y2 - crop_y1

    return crop_y1, crop_y2, station_y1, station_y2, station_union_h


def clip_rect_to_crop(points, image_w, crop_y1, crop_y2):
    """
    对矩形标注进行裁剪。

    原图坐标:
        x: [0, image_w]
        y: [0, image_h]

    crop 区域:
        x: [0, image_w]
        y: [crop_y1, crop_y2]

    输出:
        crop 后图像坐标系下的四点矩形。
    """
    x1, y1, x2, y2 = get_rect_from_points(points)

    # 和 crop 区域求交集
    nx1 = max(x1, 0.0)
    ny1 = max(y1, float(crop_y1))
    nx2 = min(x2, float(image_w))
    ny2 = min(y2, float(crop_y2))

    nw = nx2 - nx1
    nh = ny2 - ny1

    if nw < MIN_BOX_W or nh < MIN_BOX_H:
        return None

    # 转到 crop 后坐标系
    ny1 = ny1 - crop_y1
    ny2 = ny2 - crop_y1

    return rect_to_4points(nx1, ny1, nx2, ny2)


def shift_and_clip_polygon(points, image_w, crop_y1, crop_y2):
    """
    对非 rectangle 的普通点集做简单处理：
    - 保留落在 crop 区域内的点；
    - y 坐标减 crop_y1；
    - 如果剩余点数量不足 2，则丢弃。

    注意：
    这里不是严格 polygon clipping。
    如果你的数据都是 rectangle，可以不用关心这个函数。
    """
    new_points = []

    crop_h = crop_y2 - crop_y1

    for x, y in points:
        if 0 <= x <= image_w and crop_y1 <= y <= crop_y2:
            new_x = float(x)
            new_y = float(y - crop_y1)

            new_x = max(0.0, min(new_x, float(image_w)))
            new_y = max(0.0, min(new_y, float(crop_h)))

            new_points.append([new_x, new_y])

    if len(new_points) < 2:
        return None

    return new_points


def process_shape(shape, image_w, crop_y1, crop_y2):
    """
    更新单个 shape 的 points。
    """
    label = shape.get("label", "")

    if label == STATION_LABEL and not KEEP_STATION_LABEL:
        return None

    points = shape.get("points", [])
    if len(points) < 2:
        return None

    new_shape = copy.deepcopy(shape)
    shape_type = shape.get("shape_type", "rectangle")

    if shape_type == "rectangle":
        new_points = clip_rect_to_crop(points, image_w, crop_y1, crop_y2)
    else:
        new_points = shift_and_clip_polygon(points, image_w, crop_y1, crop_y2)

    if new_points is None:
        return None

    new_shape["points"] = new_points
    return new_shape


def find_image_for_json(json_path, json_data):
    """
    优先使用 json 里的 imagePath。
    如果找不到，则用 json 同名图片。
    """
    json_dir = os.path.dirname(json_path)

    image_path_in_json = json_data.get("imagePath", None)

    if image_path_in_json:
        candidate = os.path.join(json_dir, image_path_in_json)
        if os.path.exists(candidate):
            return candidate

    base = os.path.splitext(os.path.basename(json_path))[0]

    for ext in IMAGE_EXTS:
        candidate = os.path.join(json_dir, base + ext)
        if os.path.exists(candidate):
            return candidate

    return None


# ============================================================
# 4. 主处理函数
# ============================================================
def crop_dataset_by_station_center(input_dir, output_image_dir, output_json_dir):
    json_paths = sorted(glob(os.path.join(input_dir, "*.json")))

    print(f"Found json files: {len(json_paths)}")

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        image_path = find_image_for_json(json_path, json_data)

        if image_path is None:
            print(f"[WARN] Image not found for json: {json_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        image_h, image_w = image.shape[:2]

        # ------------------------------------------------------------
        # 修改点：找到所有 station，而不是只找第一个 station
        # ------------------------------------------------------------
        station_rects = find_all_station_rects(json_data)

        if len(station_rects) == 0:
            print(f"[WARN] No station label found: {json_path}")
            continue

        crop_y1, crop_y2, station_y1, station_y2, station_union_h = \
            compute_crop_range_by_all_stations(
                image_h=image_h,
                station_rects=station_rects
            )

        cropped_image = image[crop_y1:crop_y2, :]

        new_json_data = copy.deepcopy(json_data)
        new_json_data["imageHeight"] = int(crop_y2 - crop_y1)
        new_json_data["imageWidth"] = int(image_w)
        new_json_data["imageData"] = None

        new_shapes = []

        for shape in json_data.get("shapes", []):
            new_shape = process_shape(
                shape=shape,
                image_w=image_w,
                crop_y1=crop_y1,
                crop_y2=crop_y2
            )

            if new_shape is not None:
                new_shapes.append(new_shape)

        new_json_data["shapes"] = new_shapes

        image_name = os.path.basename(image_path)
        json_name = os.path.basename(json_path)

        output_image_path = os.path.join(output_image_dir, image_name)
        output_json_path = os.path.join(output_json_dir, json_name)

        new_json_data["imagePath"] = image_name

        cv2.imwrite(output_image_path, cropped_image)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(new_json_data, f, ensure_ascii=False, indent=2)

        old_shape_num = len(json_data.get("shapes", []))
        new_shape_num = len(new_shapes)
        station_num = len(station_rects)

        print(
            f"[OK] {image_name} | "
            f"old_size=({image_w}, {image_h}) | "
            f"station_num={station_num} | "
            f"station_y_range=({station_y1:.2f}, {station_y2:.2f}) | "
            f"station_union_h={station_union_h:.2f} | "
            f"crop_y=({crop_y1}, {crop_y2}) | "
            f"new_size=({image_w}, {crop_y2 - crop_y1}) | "
            f"shapes={old_shape_num}->{new_shape_num}"
        )

        if station_union_h > (crop_y2 - crop_y1):
            print(
                f"[WARN] {image_name}: station union height "
                f"{station_union_h:.2f} > crop height {crop_y2 - crop_y1}. "
                f"Cannot fully keep all stations with 1/3 crop height."
            )


if __name__ == "__main__":
    crop_dataset_by_station_center(
        input_dir=INPUT_DIR,
        output_image_dir=OUTPUT_IMAGE_DIR,
        output_json_dir=OUTPUT_JSON_DIR
    )