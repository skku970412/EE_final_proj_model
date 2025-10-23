#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E Korean License Plate Detection & OCR
----------------------------------------
- YOLOv8 detector (Ultralytics)
- Plate crop -> robust warp/inner-crop
- Multi-binarization + multi-PSM Tesseract OCR
- Exports JSON/CSV, annotated images, and plate crops

Usage (single image):
    python e2e_korean_plate.py --image your.jpg --weights runs/license_plate_yolov8n/weights/best.pt --out out/

Usage (folder, recursive):
    python e2e_korean_plate.py --dir /path/to/images --weights runs/license_plate_yolov8n/weights/best.pt --out out/ --ext .jpg .png .jpeg

Requires:
    - Python 3.9+
    - pip install -r requirements.txt
    - Tesseract installed on system (tesseract command on PATH or --tesseract-cmd provided)
"""
import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except Exception as e:
    print("❌ Ultralytics YOLO import 실패:", e, file=sys.stderr)
    print("   pip install ultralytics 로 설치하세요.", file=sys.stderr)
    sys.exit(1)

try:
    import pytesseract
except Exception as e:
    print("❌ pytesseract import 실패:", e, file=sys.stderr)
    print("   pip install pytesseract 로 설치하고, 시스템에 Tesseract OCR도 설치되어 있어야 합니다.", file=sys.stderr)
    sys.exit(1)


# ------------------------ OCR & postprocess helpers ------------------------

PLATE_RE = re.compile(r"\b\d{2,3}[가-힣]-?\d{4}\b")
# 한국 번호판에 쓰이는 대표 한글자 (필요 시 추가)
ALLOWED_KOR = "가나다라마바사아자배허하호국영군임관학무"
SUBS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "D": "0", "—": "-"})

def safe_int(x: float) -> int:
    v = int(round(float(x)))
    return max(v, 0)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def draw_label(img: np.ndarray, text: str, x1: int, y1: int) -> None:
    """Draw a filled label above box corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    bx1, by1 = x1, max(0, y1 - th - 2*pad)
    bx2, by2 = x1 + tw + 2*pad, y1
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ------------------------ Warping utilities ------------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BL, BR]."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, bl, br], dtype=np.float32)

def warp_plate_from_crop(
    crop_bgr: np.ndarray,
    target_w: int = 900,
    inner: float = 0.08,
    canny: Tuple[int, int] = (40, 140),
    area_frac: Tuple[float, float] = (0.06, 0.98),
    ar_range: Tuple[float, float] = (2.2, 7.5),
    eps: float = 0.03,
    dilate_iter: int = 2,
) -> Tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """
    Try to find a quadrilateral plate within crop and perspective-warp it.
    Returns (warped, ok, edges, chosen_quad)
    """
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)

    med = np.median(g)
    auto_low = int(max(0, 0.66 * med))
    auto_high = int(min(255, 1.33 * med))
    low, high = (min(canny[0], auto_low), max(canny[1], auto_high))

    e = cv2.Canny(g, low, high)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=dilate_iter)

    cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    H, W = crop_bgr.shape[:2]
    area_img = H * W

    best = None
    best_score = -1.0

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        frac = area / (area_img + 1e-6)
        if not (area_frac[0] <= frac <= area_frac[1]):
            continue

        pts = approx.reshape(-1, 2).astype(np.float32)
        q = _order_quad(pts)

        w1 = np.hypot(*(q[1] - q[0]))
        w2 = np.hypot(*(q[3] - q[2]))
        h1 = np.hypot(*(q[2] - q[0]))
        h2 = np.hypot(*(q[3] - q[1]))
        w = max(w1, w2)
        h = max(h1, h2)
        if h <= 1:
            continue
        ar = w / h
        if not (ar_range[0] <= ar <= ar_range[1]):
            continue

        score = float(frac * ar)
        if score > best_score:
            best_score = score
            best = q

    if best is None:
        # fallback: minAreaRect box
        ys, xs = np.where(e > 0)
        if len(xs) == 0:
            return crop_bgr, False, e, None
        rect = cv2.minAreaRect(np.column_stack((xs, ys)))
        box = cv2.boxPoints(rect).astype(np.float32)
        best = _order_quad(box)

    target_h = int(target_w / 4.5)
    dst = np.float32([[0, 0], [target_w - 1, 0], [0, target_h - 1], [target_w - 1, target_h - 1]])
    M = cv2.getPerspectiveTransform(best, dst)
    warped = cv2.warpPerspective(crop_bgr, M, (target_w, target_h))

    pad = int(inner * min(target_w, target_h))
    warped = warped[pad:target_h - pad, pad:target_w - pad]

    return warped, True, e, best

def inner_warp_from_quad(img_bgr: np.ndarray, quad_src: np.ndarray, inner_ratio: float = 0.08, target_w: int = 900):
    """Shrink inside the quad and warp to reduce frame/bolts."""
    target_h = int(target_w / 4.5)
    dst_rect = np.float32([[0, 0], [target_w - 1, 0], [0, target_h - 1], [target_w - 1, target_h - 1]])
    H_src2dst = cv2.getPerspectiveTransform(quad_src, dst_rect)
    H_dst2src = np.linalg.inv(H_src2dst)

    r = float(inner_ratio)
    inner_dst = np.float32([
        [r * target_w, r * target_h],
        [(1 - r) * target_w - 1, r * target_h],
        [r * target_w, (1 - r) * target_h - 1],
        [(1 - r) * target_w - 1, (1 - r) * target_h - 1],
    ])

    inner_src = cv2.perspectiveTransform(inner_dst[None, :, :], H_dst2src)[0].astype(np.float32)
    M = cv2.getPerspectiveTransform(inner_src, dst_rect)
    warped = cv2.warpPerspective(img_bgr, M, (target_w, target_h))
    return warped, inner_src


# ------------------------ OCR ------------------------

def binarize_variants(gray: np.ndarray) -> Dict[str, np.ndarray]:
    """Create several binarized images."""
    bins: Dict[str, np.ndarray] = {}

    # upscale + CLAHE
    g = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    b3 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 4)
    k3 = np.ones((3, 3), np.uint8)
    b4 = cv2.morphologyEx(b2, cv2.MORPH_CLOSE, k3, iterations=1)

    bins["OTSU"] = b1
    bins["ADP31_C2"] = b2
    bins["ADP41_C4"] = b3
    bins["CLOSE_ADP"] = b4
    return bins

def ocr_once(bimg: np.ndarray, psm: int, lang: str) -> Tuple[str, int, str]:
    cfg = (f"--oem 1 --psm {psm} -l {lang} "
           f"-c tessedit_do_invert=0 -c load_system_dawg=0 -c load_freq_dawg=0 "
           f"-c tessedit_char_whitelist={ALLOWED_KOR}0123456789-")
    raw = pytesseract.image_to_string(bimg, config=cfg)
    txt = re.sub(r"[^0-9가-힣- ]", "", raw.translate(SUBS)).strip().replace(" ", "")
    score = 3 if PLATE_RE.fullmatch(txt) else (2 if PLATE_RE.search(txt) else 0)
    return txt, score, raw.strip()

@dataclass
class PlateCandidate:
    text: str
    score: int
    raw: str
    variant: str
    psm: int

@dataclass
class DetectionResult:
    image: str
    bbox: Tuple[int, int, int, int]
    conf: float
    text: str
    score: int
    raw: str
    crop_path: str = ""
    flat_path: str = ""
    bin_best_path: str = ""


# ------------------------ Main pipeline ------------------------

def process_crop_for_text(crop_bgr: np.ndarray, lang: str = "kor+eng") -> Tuple[PlateCandidate, Dict[str, Any]]:
    """Run warp + binarization + OCR voting, return best candidate and debug info."""
    # First warp attempt
    flat, ok, _edges, quad = warp_plate_from_crop(crop_bgr, target_w=900, inner=0.08,
                                                  canny=(40, 140), area_frac=(0.06, 0.98),
                                                  ar_range=(2.2, 7.5), eps=0.03, dilate_iter=2)
    # optional inner warp
    if quad is not None:
        try:
            flat, _inner_src = inner_warp_from_quad(crop_bgr, quad, inner_ratio=0.08, target_w=900)
        except Exception:
            pass

    # Light inner crop to avoid losing characters on sides
    h, w = flat.shape[:2]
    pad = int(0.06 * min(h, w))
    roi = flat[pad:h - pad, pad:w - pad]

    # White padding around
    p = int(0.08 * max(h, w))
    roi_pad = cv2.copyMakeBorder(roi, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    gray = cv2.cvtColor(roi_pad, cv2.COLOR_BGR2GRAY)
    variants = binarize_variants(gray)

    best = PlateCandidate(text="", score=-1, raw="", variant="", psm=7)

    for name, b in variants.items():
        for psm in (7, 8, 6, 13):
            t, s, r = ocr_once(b, psm, lang=lang)
            if s > best.score or (s == best.score and len(t) > len(best.text)):
                best = PlateCandidate(text=t, score=s, raw=r, variant=name, psm=psm)

    debug = {"roi_pad": roi_pad, "flat": flat, "best_variant": best.variant, "best_psm": best.psm}
    return best, debug

def annotate_and_save(orig_bgr: np.ndarray, boxes: np.ndarray, confs: np.ndarray,
                      texts: List[str], out_img_path: Path) -> None:
    """Draw YOLO boxes and recognized text, save image."""
    canvas = orig_bgr.copy()
    for (x1, y1, x2, y2), c, t in zip(boxes, confs, texts):
        x1, y1, x2, y2 = map(safe_int, (x1, y1, x2, y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{t or '—'} ({c:.2f})"
        draw_label(canvas, label, x1, y1)
    cv2.imwrite(str(out_img_path), canvas)

def discover_images(dir_path: Path, exts: List[str]) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(dir_path.rglob(f"*{ext}"))
        files.extend(dir_path.rglob(f"*{ext.upper()}"))
    return sorted(set(files))

def run_e2e(
    weights: Path,
    image: Path = None,
    dir: Path = None,
    out_dir: Path = Path("out"),
    conf: float = 0.5,
    iou: float = 0.5,
    device: str = "",
    save_crops: bool = True,
    save_flats: bool = True,
    save_annotated: bool = True,
    save_bin_best: bool = False,
    tesseract_cmd: str = "",
    lang: str = "kor+eng",
    exts: List[str] = None,
) -> Dict[str, Any]:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    if not weights.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {weights}")

    model = YOLO(str(weights))

    ensure_dir(out_dir)
    out_imgs = out_dir / "annotated"
    out_crops = out_dir / "crops"
    out_flats = out_dir / "flats"
    out_bins = out_dir / "bins"
    for p in [out_imgs, out_crops, out_flats, out_bins]:
        ensure_dir(p)

    if image is None and dir is None:
        raise ValueError("--image 또는 --dir 중 하나는 지정되어야 합니다.")

    paths: List[Path] = []
    if image is not None:
        paths = [image]
    else:
        if exts is None or len(exts) == 0:
            exts = [".jpg", ".jpeg", ".png", ".bmp"]
        paths = discover_images(dir, exts)

    all_results: List[DetectionResult] = []

    for p in paths:
        # YOLO inference (Ultralytics will handle reading)
        results = model.predict(source=str(p), conf=conf, iou=iou, device=device, verbose=False)
        for ri, result in enumerate(results):
            orig_bgr = result.orig_img  # BGR
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
            confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros((0,))
            # sort by confidence desc
            order = np.argsort(-confs) if len(confs) else []
            boxes = boxes[order] if len(order) else boxes
            confs = confs[order] if len(order) else confs

            texts_for_annot: List[str] = []
            for bi, (x1, y1, x2, y2) in enumerate(boxes):
                x1i, y1i, x2i, y2i = map(safe_int, (x1, y1, x2, y2))
                crop_bgr = orig_bgr[y1i:y2i, x1i:x2i].copy()

                # save raw crop
                crop_path = out_crops / f"{p.stem}_b{bi}.png"
                if save_crops:
                    cv2.imwrite(str(crop_path), crop_bgr)

                # OCR pipeline
                best, dbg = process_crop_for_text(crop_bgr, lang=lang)
                texts_for_annot.append(best.text)

                # save flat/roi best bin for debugging
                flat_path = ""
                bin_best_path = ""
                if save_flats:
                    flat_path = str(out_flats / f"{p.stem}_b{bi}_flat.png")
                    cv2.imwrite(flat_path, dbg["flat"])
                if save_bin_best and best.variant:
                    bin_best_path = str(out_bins / f"{p.stem}_b{bi}_{best.variant}_psm{best.psm}.png")
                    # Recreate the best bin to save it
                    gray = cv2.cvtColor(dbg["roi_pad"], cv2.COLOR_BGR2GRAY)
                    bins = binarize_variants(gray)
                    if best.variant in bins:
                        cv2.imwrite(bin_best_path, bins[best.variant])

                all_results.append(DetectionResult(
                    image=str(p),
                    bbox=(safe_int(x1), safe_int(y1), safe_int(x2), safe_int(y2)),
                    conf=float(confs[bi]) if len(confs) > bi else 0.0,
                    text=best.text,
                    score=int(best.score),
                    raw=best.raw,
                    crop_path=str(crop_path) if save_crops else "",
                    flat_path=str(flat_path) if save_flats else "",
                    bin_best_path=str(bin_best_path) if (save_bin_best and bin_best_path) else "",
                ))

            # annotated save (once per result)
            if save_annotated and len(boxes):
                out_img_path = out_imgs / f"{p.stem}_annotated.png"
                annotate_and_save(orig_bgr, boxes, confs, texts_for_annot, out_img_path)

    # export JSON & CSV
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "conf", "text", "score", "raw", "crop_path", "flat_path", "bin_best_path"])
        for r in all_results:
            x1, y1, x2, y2 = r.bbox
            w.writerow([r.image, x1, y1, x2, y2, r.conf, r.text, r.score, r.raw, r.crop_path, r.flat_path, r.bin_best_path])

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "count": len(all_results),
        "out_dir": str(out_dir),
    }


def main():
    ap = argparse.ArgumentParser(description="E2E Korean License Plate Detection & OCR")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="단일 이미지 경로")
    src.add_argument("--dir", type=Path, help="이미지 폴더(재귀 탐색)")
    ap.add_argument("--ext", dest="exts", nargs="*", default=[], help="폴더 모드에서 인식할 확장자 (예: --ext .jpg .png)")
    ap.add_argument("--weights", type=Path, required=True, help="YOLO 가중치 (예: runs/.../best.pt)")
    ap.add_argument("--out", type=Path, default=Path("out"), help="출력 디렉터리")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IOU threshold")
    ap.add_argument("--device", type=str, default="", help="장치 지정 (예: 'cpu' 또는 'cuda:0')")
    ap.add_argument("--no-crops", action="store_true", help="crop 저장 비활성화")
    ap.add_argument("--no-flats", action="store_true", help="warp 결과 저장 비활성화")
    ap.add_argument("--no-annotated", action="store_true", help="annotated 이미지 저장 비활성화")
    ap.add_argument("--save-bin-best", action="store_true", help="최고 득점 이진화 결과 저장")
    ap.add_argument("--tesseract-cmd", type=str, default="", help="Tesseract 실행 파일 경로 (필요시 지정)")
    ap.add_argument("--lang", type=str, default="kor+eng", help="Tesseract 언어 (기본: kor+eng)")

    args = ap.parse_args()

    try:
        res = run_e2e(
            weights=args.weights,
            image=args.image,
            dir=args.dir,
            out_dir=args.out,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save_crops=not args.no_crops,
            save_flats=not args.no_flats,
            save_annotated=not args.no_annotated,
            save_bin_best=args.save_bin_best,
            tesseract_cmd=args.tesseract_cmd,
            lang=args.lang,
            exts=args.exts,
        )
        print("✅ 완료:", json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print("❌ 오류:", e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()