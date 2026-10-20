import argparse
import csv
import os
import statistics
import time
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO



def summarize_tracks(tracks, total_frames, total_detections, short_limit=10):
 
    lengths = [t["n"] for t in tracks.values()]
    if not lengths:
        return {"unique_ids": 0, "mean_len": 0.0, "median_len": 0.0,
                "short_ratio": 0.0, "dets_per_frame": 0.0}
    short = sum(1 for n in lengths if n < short_limit)
    return {
        "unique_ids": len(lengths),
        "mean_len": statistics.mean(lengths),
        "median_len": statistics.median(lengths),
        "short_ratio": short / len(lengths),
        "dets_per_frame": total_detections / max(total_frames, 1),
    }


def color_for_id(track_id):
    rng = np.random.RandomState(track_id)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def trail_alpha(age, ttl):
 
    if age > ttl:
        return 0.0
    return 1.0 - 0.65 * (age / max(ttl, 1))


def draw_trails(frame, trails, trail_seen, frame_idx, ttl):
 
    for tid, pts_q in trails.items():
        age = frame_idx - trail_seen.get(tid, -10**9)
        a = trail_alpha(age, ttl)
        if a <= 0 or not pts_q:
            continue
        color = tuple(int(c * a) for c in color_for_id(tid))
        pts = np.array(pts_q, dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, color, 2 if age == 0 else 1)
        last = (int(pts[-1, 0, 0]), int(pts[-1, 0, 1]))
        cv2.circle(frame, last, 3 if age == 0 else 2, color, -1)   


def prune_trails(trails, trail_seen, frame_idx, ttl):
    stale = [t for t, seen in trail_seen.items() if frame_idx - seen > ttl]
    for t in stale:
        trails.pop(t, None)
        trail_seen.pop(t, None)


def draw_info_panel(frame, lines):
    line_h, x0, y0 = 22, 8, 8
    w, h = 270, 12 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x0 + 8, y0 + 20 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def build_panel_lines(tracker, frame_idx, active, unique_ids, frame_counts, max_rows=10):

    lines = [
        f"tracker: {os.path.basename(tracker).replace('.yaml', '')}",
        f"frame: {frame_idx}",
        f"active tracks: {active}",
        f"unique IDs so far: {unique_ids}",
        "--- in frame now ---",
    ]
    if frame_counts:
        for name, n in frame_counts.most_common(max_rows):
            lines.append(f"{name}: {n}")
    else:
        lines.append("(nothing detected)")
    return lines


def parse_args():
    p = argparse.ArgumentParser(description="Dron videosunda obyekt izləmə (baseline)")
    p.add_argument("--source", required=True, help="Video faylının yolu")
    p.add_argument("--model", default="yolov8s.pt",
                   help="YOLO modeli. Kiçik obyektlər üçün s/m, sürət üçün n")

    local_tracker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack_drone.yaml")
    p.add_argument("--tracker", default=local_tracker if os.path.exists(local_tracker) else "bytetrack.yaml",
                   help="bytetrack_drone.yaml, bytetrack.yaml və ya botsort.yaml")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="Yalnız bu sinif nömrələrini izlə. Verilməsə, modelin BÜTÜN sinifləri izlənir. "
                        "DİQQƏT: COCO modeli (yolov8s.pt) üçün 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck; "
                        "VisDrone ilə öyrədilmiş modeldə nömrələr fərqlidir (0=pedestrian, 1=people, "
                        "2=bicycle, 3=car, 4=van, 5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor)")
    p.add_argument("--conf", type=float, default=0.10,
                   help="Əminlik həddi. Aşağı olması daha çox obyekt tutur (həm də daha çox yalançı aşkarlama). "
                        "Çox yalançı qutu görsən 0.15-0.25 et")
    p.add_argument("--max-det", type=int, default=500,
                   help="Bir kadrda maksimum aşkarlama sayı (sıx dron kadrlarında standart 300 azdır)")
    p.add_argument("--trail-len", type=int, default=40,
                   help="İzdə saxlanan maksimum nöqtə sayı (kadr)")
    p.add_argument("--trail-ttl", type=int, default=45,
                   help="Obyekt aşkarlanmayandan sonra izi neçə kadr görünsün. "
                        "Bu müddətdən sonra iz tam yox olur")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Modelə verilən şəkil ölçüsü. Böyük olması kiçik obyektləri tutmağa kömək edir")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Yalnız ilk N kadrı emal et (sürətli sınaq üçün). 0 = hamısı")
    p.add_argument("--save", default=None, help="Nəticə videosunu bu fayla yaz")
    p.add_argument("--csv", default="tracks.csv", help="Hər ID üçün statistikanı bu CSV-yə yaz")
    p.add_argument("--labels", action="store_true",
                   help="Qutuların üstündə sinif adını da göstər (standart: yalnız ID)")
    p.add_argument("--no-show", action="store_true", help="Pəncərə açma")
    return p.parse_args()



def main():
    args = parse_args()

    model = YOLO(args.model)
    for c in (args.classes or []):
        if c not in model.names:
            raise SystemExit(f"Yanlış sinif nömrəsi: {c}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılmadı: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (width, height))

    tracks = defaultdict(lambda: {"first": None, "last": None, "n": 0, "cls": Counter()})
    trails = defaultdict(lambda: deque(maxlen=args.trail_len))
    trail_seen = {} 
    frame_idx = 0
    total_detections = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            classes=args.classes,
            conf=args.conf,
            imgsz=args.imgsz,
            max_det=args.max_det,
            verbose=False,
        )[0]

        boxes = results.boxes
        active = 0
        frame_counts = Counter()   

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()

            for (x1, y1, x2, y2), tid, cls_id in zip(xyxy, ids, classes):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                name = model.names[cls_id]
                active += 1
                total_detections += 1
                frame_counts[name] += 1

                t = tracks[tid]
                if t["first"] is None:
                    t["first"] = frame_idx
                t["last"] = frame_idx
                t["n"] += 1
                t["cls"][name] += 1

                color = color_for_id(tid)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trails[tid].append((cx, cy))
                trail_seen[tid] = frame_idx
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                label = f"{name} {tid}" if args.labels else str(tid)
                cv2.putText(frame, label, (x1, max(y1 - 3, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        draw_trails(frame, trails, trail_seen, frame_idx, args.trail_ttl)
        prune_trails(trails, trail_seen, frame_idx, args.trail_ttl)

        draw_info_panel(frame, build_panel_lines(
            args.tracker, frame_idx, active, len(tracks), frame_counts))

        if writer:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("Drone Tracker (q = cixis)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    summary = summarize_tracks(tracks, frame_idx, total_detections)

    print("\n===== YEKUN HESABAT =====")
    print(f"Izleyici          : {args.tracker}")
    print(f"Model / imgsz     : {args.model} / {args.imgsz}")
    print(f"Izlenen siniflər  : {args.classes if args.classes else 'hamisi'}")
    print(f"Emal olunan kadr  : {frame_idx}  (orta sürət: {frame_idx / max(elapsed, 1e-6):.1f} FPS)")
    print(f"Unikal ID sayı    : {summary['unique_ids']}")
    print(f"ID-nin orta ömrü  : {summary['mean_len']:.1f} kadr (median {summary['median_len']:.1f})")
    print(f"Qısa ömürlü ID    : {summary['short_ratio'] * 100:.1f}%  (< 10 kadr)")
    print(f"Kadr başına aşkarlama: {summary['dets_per_frame']:.1f}")

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "first_frame", "last_frame", "frames_seen"])
        for tid, t in sorted(tracks.items()):
            main_cls = t["cls"].most_common(1)[0][0]
            w.writerow([tid, main_cls, t["first"], t["last"], t["n"]])
    print(f"ID statistikası yazıldı: {args.csv}")


if __name__ == "__main__":
    main()
