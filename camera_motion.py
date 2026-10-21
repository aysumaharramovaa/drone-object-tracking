import argparse
import csv
import math

import cv2
import numpy as np

def to3x3(M):
    if M is None:
        return None
    if M.shape == (3, 3):
        return M
    H = np.eye(3)
    H[:2, :] = M
    return H


def scale_matrix(H_small, s):
    S = np.diag([s, s, 1.0])
    return np.linalg.inv(S) @ H_small @ S


def estimate_motion(prev_gray, cur_gray, mode="affine", max_corners=600,
                    ransac_thresh=3.0, min_points=12):
    pts_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners,
                                       qualityLevel=0.01, minDistance=10, blockSize=7)
    if pts_prev is None or len(pts_prev) < min_points:
        return None

    pts_cur, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, pts_prev, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    good = status.ravel() == 1                   
    p0 = pts_prev[good].reshape(-1, 2)
    p1 = pts_cur[good].reshape(-1, 2)
    if len(p0) < min_points:
        return None

    if mode == "homography":
        H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, ransac_thresh)
    else:
        M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thresh)
        H = to3x3(M)

    if H is None or inl is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < min_points:
        return None
    det = abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
    if not (0.5 < det < 2.0):
        return None

    return {"H": H, "p0": p0, "p1": p1, "inliers": inl}


def describe_motion(H, w, h):
    c = np.array([w / 2.0, h / 2.0, 1.0])
    c2 = H @ c
    c2 = c2[:2] / c2[2]
    dx, dy = c2[0] - w / 2.0, c2[1] - h / 2.0
    rot = math.degrees(math.atan2(H[1, 0], H[0, 0]))
    scale = math.sqrt(abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]))
    return dx, dy, rot, scale


def compensated_diff(prev, cur, H, gain=4):

    h, w = cur.shape[:2]
    warped = cv2.warpPerspective(prev, H, (w, h))
    valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (w, h))
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8)) > 0

    def blur_gray(img):
        return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    g_cur, g_warp, g_prev = blur_gray(cur), blur_gray(warped), blur_gray(prev)
    comp = cv2.absdiff(g_cur, g_warp)    
    raw = cv2.absdiff(g_cur, g_prev)   

    resid_comp = float(comp[valid].mean()) if valid.any() else 0.0
    resid_raw = float(raw[valid].mean()) if valid.any() else 0.0

    vis = np.clip(comp.astype(np.int32) * gain, 0, 255).astype(np.uint8)
    vis[~valid] = 0
    return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO), resid_comp, resid_raw

def put_lines(img, lines, x=8, y=8, color=(255, 255, 255)):
    line_h = 20
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + 300, y + 10 + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, t in enumerate(lines):
        c = color
        if t.startswith("LOW"):
            c = (0, 0, 255)
        cv2.putText(img, t, (x + 8, y + 20 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)


def draw_flow(img, p0, p1, inl, k):
    for a, b, ok in zip(p0, p1, inl):
        pa = (int(a[0] * k), int(a[1] * k))
        pb = (int(b[0] * k), int(b[1] * k))
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.line(img, pa, pb, color, 1)
        cv2.circle(img, pb, 2, color, -1)


def parse_args():
    p = argparse.ArgumentParser(description="Kamera hərəkətinin qiymətləndirilməsi və stabilləşdirmə")
    p.add_argument("--source", required=True, help="Video faylının yolu")
    p.add_argument("--motion", choices=["affine", "homography"], default="affine",
                   help="Hərəkət modeli (standart: affine, daha sabitdir)")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Hesablama üçün kadrın kiçildilməsi (0.5 = yarı ölçü, sürətlidir)")
    p.add_argument("--reanchor", type=int, default=60,
                   help="Stabilləşdirmədə istinad kadrı hər N kadrdan bir yenilənir. Dron irəli uçdukca "
                        "baxış sahəsi istinad kadrından çıxır, bu isə bunun qarşısını alır. 0 = heç vaxt")
    p.add_argument("--gain", type=float, default=4.0, help="Fərq şəklinin gücləndirilməsi")
    p.add_argument("--panel-width", type=int, default=960, help="Debug videosunda hər panelin eni")
    p.add_argument("--max-frames", type=int, default=0, help="Yalnız ilk N kadr (0 = hamısı)")
    p.add_argument("--save", default=None, help="Yan-yana debug videosu")
    p.add_argument("--save-stab", default=None, help="Stabilləşdirilmiş video")
    p.add_argument("--csv", default="camera_motion.csv", help="Kamera hərəkəti statistikası")
    p.add_argument("--no-show", action="store_true", help="Pəncərə açma")
    return p.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılmadı: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    s = args.scale

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Video boşdur")

    def small_gray(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else g

    pw = min(w, args.panel_width)
    ph = int(round(h * pw / w))
    k = pw / (w * s)              
    writer_dbg = writer_stab = None
    if args.save:
        writer_dbg = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw * 2, ph))
    if args.save_stab:
        writer_stab = cv2.VideoWriter(args.save_stab, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer_stab.write(prev)    

    prev_small = small_gray(prev)
    M = np.eye(3)                     
    ref_frame = 0
    rows = []
    resid_comp_sum = resid_raw_sum = 0.0
    low_conf = 0
    frame_idx = 0

    while True:
        ok, cur = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        cur_small = small_gray(cur)
        est = estimate_motion(prev_small, cur_small, mode=args.motion)

        if est is None:
            H, confident = np.eye(3), False
            p0 = p1 = np.zeros((0, 2))
            inl = np.zeros(0, bool)
            n_pts = n_in = 0
        else:
            H = scale_matrix(est["H"], s)
            p0, p1, inl = est["p0"], est["p1"], est["inliers"]
            n_pts, n_in = len(p0), int(inl.sum())
            confident = n_in >= 20 and n_in / n_pts >= 0.3  
        if not confident:
            low_conf += 1

        dx, dy, rot, scale = describe_motion(H, w, h)

        diff_img, r_comp, r_raw = compensated_diff(prev, cur, H, args.gain)
        resid_comp_sum += r_comp
        resid_raw_sum += r_raw


        M = M @ np.linalg.inv(H)
        if args.reanchor and frame_idx % args.reanchor == 0:
            M = np.eye(3)
            ref_frame = frame_idx

        if writer_stab or not args.no_show:
            stab = cv2.warpPerspective(cur, M, (w, h))
        if writer_stab:
            writer_stab.write(stab)

        rows.append([frame_idx, round(dx, 3), round(dy, 3), round(rot, 4), round(scale, 5),
                     n_in, n_pts, int(confident), round(r_raw, 3), round(r_comp, 3)])

        if writer_dbg or not args.no_show:
            left = cv2.resize(cur, (pw, ph))
            draw_flow(left, p0, p1, inl, k)
            put_lines(left, [
                f"frame {frame_idx}",
                f"bg shift dx={dx:+.1f} dy={dy:+.1f} px",
                f"rot={rot:+.2f} deg  scale={scale:.3f}",
                f"bg points {n_in}/{n_pts}",
                "OK" if confident else "LOW CONFIDENCE",
            ])
            right = cv2.resize(diff_img, (pw, ph))
            put_lines(right, [
                "motion-compensated difference",
                "bright = moving objects",
                f"residual {r_raw:.1f} -> {r_comp:.1f}",
            ])
            panel = np.hstack([left, right])
            if writer_dbg:
                writer_dbg.write(panel)
            if not args.no_show:
                cv2.imshow("Camera motion (q = cixis)", panel)
                cv2.imshow("Stabilized (ref frame %d)" % ref_frame, cv2.resize(stab, (pw, ph)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        prev, prev_small = cur, cur_small

    cap.release()
    for wr in (writer_dbg, writer_stab):
        if wr:
            wr.release()
    cv2.destroyAllWindows()

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "dx", "dy", "rot_deg", "scale", "inliers", "points",
                     "confident", "resid_raw", "resid_comp"])
        wr.writerows(rows)

    n = max(len(rows), 1)
    reduction = (1 - resid_comp_sum / resid_raw_sum) * 100 if resid_raw_sum > 0 else 0.0
    mean_shift = float(np.mean([math.hypot(r[1], r[2]) for r in rows])) if rows else 0.0
    print("\n===== KAMERA HƏRƏKƏTİ HESABATI =====")
    print(f"Emal olunan kadr            : {len(rows)}")
    print(f"Orta kamera sürüşməsi       : {mean_shift:.2f} piksel/kadr")
    print(f"Etibarsız (az nöqtə) kadr   : {low_conf} ({low_conf / n * 100:.1f}%)")
    print(f"Fonun qalıq fərqi           : {resid_raw_sum / n:.2f} -> {resid_comp_sum / n:.2f}")
    print(f"Kompensasiya fonun fərqini  : {reduction:.1f}% azaltdı")
    print(f"Statistika yazıldı          : {args.csv}")


if __name__ == "__main__":
    main()