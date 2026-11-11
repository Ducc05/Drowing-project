"""Bounded-size live preview; does not change inference or persisted results."""

def draw_cells(frame, overlay):
    """Faint cell grid plus a red tint on cells that currently hold a sustained drowning alarm (in place)."""
    import cv2

    height, width = frame.shape[:2]
    cols, rows = overlay["cols"], overlay["rows"]
    layer = frame.copy()
    for col, row in overlay["hot"]:
        cv2.rectangle(layer, (round(col * width / cols), round(row * height / rows)),
                      (round((col + 1) * width / cols) - 1, round((row + 1) * height / rows) - 1), (0, 0, 255), -1)
    cv2.addWeighted(layer, .3, frame, .7, 0, dst=frame)
    layer = frame.copy()
    for col in range(1, cols):
        x = round(col * width / cols)
        cv2.line(layer, (x, 0), (x, height - 1), (255, 255, 255), 1)
    for row in range(1, rows):
        y = round(row * height / rows)
        cv2.line(layer, (0, y), (width - 1, y), (255, 255, 255), 1)
    cv2.addWeighted(layer, .25, frame, .75, 0, dst=frame)
    return frame


def preview_jpeg(bgr, records, timestamp_ms, active_alert_tracks=(), cell_overlay=None):
    import cv2

    scale = min(1.0, 960 / max(bgr.shape[:2]))
    frame = cv2.resize(bgr, (max(1, round(bgr.shape[1] * scale)), max(1, round(bgr.shape[0] * scale))))
    if cell_overlay:
        draw_cells(frame, cell_overlay)
    colors = {"drowning": (0, 165, 255), "swimming": (80, 200, 80), "out_of_water": (170, 170, 170)}
    for row in records:
        x1, y1, x2, y2 = (round(value * scale) for value in row["bbox_xyxy"])
        label = row.get("label") or "unknown"
        color = (0, 0, 255) if row.get("alert") else colors.get(label, (220, 220, 220))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        score = row.get("detector_confidence")
        caption = f"{row['track_id']} {label}" + (f" {score:.2f}" if score is not None else "")
        cv2.putText(frame, caption, (max(0, x1), max(40, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, color, 1, cv2.LINE_AA)
    alert = bool(active_alert_tracks) or any(row.get("alert") for row in records)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 25), (0, 0, 180) if alert else (25, 25, 25), -1)
    caption = f"{timestamp_ms / 1000:.1f}s" + (" | DROWNING ALERT" if alert else "")
    cv2.putText(frame, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return encoded.tobytes() if ok else None
