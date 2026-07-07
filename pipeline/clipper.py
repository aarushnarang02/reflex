"""Evidence clip extraction: cut the seconds around a near miss.

Every flagged event gets a short video clip so a human can always review
what the machine saw. Clips are cut from the annotated output video so
boxes, speeds, and risk overlays are baked in.
"""

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

PAD_SECONDS = 2.0  # context kept before t_start and after t_end


def to_h264(path: Path) -> Path:
    """Re-encode in place to H.264, the codec browsers can actually play.
    OpenCV's writer produces MPEG-4 Part 2, which the <video> tag rejects."""
    tmp = path.with_suffix(".h264.mp4")
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
         "-i", str(path), "-c:v", "libx264", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        check=True)
    tmp.replace(path)
    return path


def extract_clip(source: Path, t_start: float, t_end: float, out_path: Path) -> Path:
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    first = max(0, int((t_start - PAD_SECONDS) * fps))
    last = min(total - 1, int((t_end + PAD_SECONDS) * fps))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for _ in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    return to_h264(out_path)
