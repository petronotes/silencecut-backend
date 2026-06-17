"""SilenceCut Backend тАФ FastAPI + FFmpeg"""
import os, uuid, asyncio, shutil, subprocess, json, time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SilenceCut API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path("/tmp/silencecut")
WORK_DIR.mkdir(parents=True, exist_ok=True)
tasks: dict = {}
TASK_TTL = 3600


def cleanup_old_tasks():
    now = time.time()
    dead = [t for t, v in list(tasks.items()) if now - v.get("created", now) > TASK_TTL]
    for tid in dead:
        shutil.rmtree(WORK_DIR / tid, ignore_errors=True)
        del tasks[tid]


@app.get("/")
def root():
    return {"service": "SilenceCut", "status": "ok"}


@app.get("/health")
def health():
    ffmpeg_path = shutil.which("ffmpeg")
    return {
        "status": "ok",
        "ffmpeg": ffmpeg_path is not None,
        "ffmpeg_path": ffmpeg_path or "not found",
    }


@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(-40.0),
    min_silence: float = Form(0.5),
    pad: float = Form(0.1),
    max_keep: float = Form(0.5),
):
    cleanup_old_tasks()
    task_id = str(uuid.uuid4())
    task_dir = WORK_DIR / task_id
    task_dir.mkdir(parents=True)
    ext = Path(file.filename).suffix.lower() or ".mp4"
    input_path = task_dir / ("input" + ext)
    with open(input_path, "wb") as fo:
        shutil.copyfileobj(file.file, fo)
    tasks[task_id] = {
        "status": "processing", "progress": 0, "log": "╨д╨░╨╣╨╗ ╨┐╨╛╨╗╤Г╤З╨╡╨╜тАж",
        "created": time.time(), "input_path": str(input_path),
        "task_dir": str(task_dir),
        "params": {
            "threshold": threshold, "min_silence": min_silence,
            "pad": pad, "max_keep": max_keep,
        },
    }
    background_tasks.add_task(run_processing, task_id)
    return {"task_id": task_id}


async def run_processing(task_id: str):
    t = tasks.get(task_id)
    if not t:
        return
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: process_with_ffmpeg(
                task_id,
                Path(t["input_path"]),
                Path(t["task_dir"]) / "output.mp4",
                t["params"],
            ),
        )
    except Exception as e:
        tasks[task_id].update({"status": "error", "detail": str(e)})


def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def probe_streams(path: Path) -> dict:
    """Return info about video/audio streams."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    data = json.loads(r.stdout)
    streams = data.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(data.get("format", {}).get("duration", 0))
    # detect rotation (Android vertical video)
    rotation = 0
    for s in streams:
        if s.get("codec_type") == "video":
            # Check side_data_list for rotation
            for sd in s.get("side_data_list", []):
                if "rotation" in sd:
                    rotation = abs(int(sd["rotation"]))
            # Also check tags
            tags = s.get("tags", {})
            if "rotate" in tags:
                rotation = abs(int(tags["rotate"]))
    return {"has_video": has_video, "has_audio": has_audio,
            "duration": duration, "rotation": rotation}


def detect_silence(input_path: Path, threshold: float, min_silence: float):
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-af", f"silencedetect=noise={threshold}dB:duration={min_silence}",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    total = get_duration(input_path)
    silence, cur_start = [], None
    for line in r.stderr.split("\n"):
        if "silence_start:" in line:
            try:
                cur_start = float(line.split("silence_start:")[-1].strip().split()[0])
            except Exception:
                pass
        elif "silence_end:" in line:
            try:
                end = float(line.split("silence_end:")[-1].strip().split()[0])
                if cur_start is not None:
                    silence.append((cur_start, end))
                    cur_start = None
            except Exception:
                pass
    if cur_start is not None:
        silence.append((cur_start, total))
    return silence, total


def build_loud(silence, total, pad, max_keep):
    if not silence:
        return [(0.0, total)]
    loud, prev = [], 0.0
    for s, e in silence:
        seg = (max(0.0, prev), min(total, s + pad))
        if seg[1] - seg[0] > 0.05:
            loud.append(seg)
        prev = max(prev, e - pad)
    if prev < total - 0.1:
        loud.append((prev, total))
    merged = []
    for seg in loud:
        if merged and seg[0] - merged[-1][1] <= max_keep:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))
    return [tuple(s) for s in merged if s[1] - s[0] > 0.05]


def build_filter(segs, has_video: bool, has_audio: bool, rotation: int) -> tuple:
    """
    Build filter_complex + map args.
    Fixes Android h264/avc1 issues:
      - force decode via scale filter (flushes decoder)
      - handle rotation via transpose
      - audio-only fallback if no video stream
    """
    n = len(segs)

    if has_video and has_audio:
        # Video filter: decode fully, handle rotation, trim, concat
        # scale=trunc(iw/2)*2:trunc(ih/2)*2 ensures even dimensions (required by libx264)
        # fps=fps=source forces constant framerate тАФ fixes frame=0 on Android VBR
        rot_filter = ""
        if rotation == 90:
            rot_filter = ",transpose=1"
        elif rotation == 180:
            rot_filter = ",transpose=1,transpose=1"
        elif rotation == 270:
            rot_filter = ",transpose=2"

        pv = [
            f"[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2{rot_filter},"
            f"fps=fps=30,trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}]"
            for i, (s, e) in enumerate(segs)
        ]
        pa = [
            f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]"
            for i, (s, e) in enumerate(segs)
        ]
        inputs_v = "".join(f"[v{i}]" for i in range(n))
        inputs_a = "".join(f"[a{i}]" for i in range(n))
        cat = (f"{inputs_v}concat=n={n}:v=1:a=0[outv];"
               f"{inputs_a}concat=n={n}:v=0:a=1[outa]")
        fc = ";".join(pv + pa + [cat])
        maps = ["-map", "[outv]", "-map", "[outa]"]

    elif has_video and not has_audio:
        pv = [
            f"[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,"
            f"fps=fps=30,trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}]"
            for i, (s, e) in enumerate(segs)
        ]
        inputs_v = "".join(f"[v{i}]" for i in range(n))
        cat = f"{inputs_v}concat=n={n}:v=1:a=0[outv]"
        fc = ";".join(pv + [cat])
        maps = ["-map", "[outv]"]

    else:
        # Audio only
        pa = [
            f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]"
            for i, (s, e) in enumerate(segs)
        ]
        inputs_a = "".join(f"[a{i}]" for i in range(n))
        cat = f"{inputs_a}concat=n={n}:v=0:a=1[outa]"
        fc = ";".join(pa + [cat])
        maps = ["-map", "[outa]"]

    return fc, maps


def process_with_ffmpeg(task_id: str, input_path: Path, output_path: Path, params: dict):
    t = tasks[task_id]

    t.update({"log": "╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╤О ╤Д╨╛╤А╨╝╨░╤В ╨▓╨╕╨┤╨╡╨╛тАж", "progress": 5})
    info = probe_streams(input_path)
    has_video = info["has_video"]
    has_audio = info["has_audio"]
    rotation = info["rotation"]

    if not has_audio:
        raise RuntimeError("╨Т ╤Д╨░╨╣╨╗╨╡ ╨╜╨╡╤В ╨░╤Г╨┤╨╕╨╛╨┤╨╛╤А╨╛╨╢╨║╨╕ тАФ ╨╜╨╡╤З╨╡╨│╨╛ ╨░╨╜╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╤В╤М.")

    t.update({"log": "╨Р╨╜╨░╨╗╨╕╨╖ ╨░╤Г╨┤╨╕╨╛тАж", "progress": 10})
    silence, total = detect_silence(input_path, params["threshold"], params["min_silence"])
    t.update({"log": f"╨Э╨░╨╣╨┤╨╡╨╜╨╛ ╨┐╨░╤Г╨╖: {len(silence)}", "progress": 30})

    segs = build_loud(silence, total, params["pad"], params["max_keep"])
    t.update({"log": f"╨Р╨║╤В╨╕╨▓╨╜╤Л╤Е ╤Б╨╡╨│╨╝╨╡╨╜╤В╨╛╨▓: {len(segs)}", "progress": 40})

    if not segs:
        raise RuntimeError("╨Э╨╡╤В ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╤Б╨╡╨│╨╝╨╡╨╜╤В╨╛╨▓. ╨б╨╜╨╕╨╖╤М ╨┐╨╛╤А╨╛╨│ ╨│╤А╨╛╨╝╨║╨╛╤Б╤В╨╕.")

    out_dur = sum(e - s for s, e in segs)
    t.update({"log": "╨Ь╨╛╨╜╤В╨░╨╢ тЖТ MP4тАж", "progress": 50})

    fc, maps = build_filter(segs, has_video, has_audio, rotation)

    # Codec args
    if has_video:
        codec_args = [
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",          # ensure compatibility
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
        ]
    else:
        codec_args = ["-c:a", "aac", "-b:a", "128k"]

    cmd = (
        ["ffmpeg", "-y",
         "-fflags", "+genpts",              # regenerate PTS тАФ fixes Android VBR issues
         "-i", str(input_path),
         "-filter_complex", fc]
        + maps
        + codec_args
        + [str(output_path)]
    )

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{proc.stderr[-1000:]}")

    removed = total - out_dur
    t.update({
        "status": "done", "progress": 100, "log": "╨У╨╛╤В╨╛╨▓╨╛!",
        "original_duration": total, "output_duration": out_dur,
        "removed_sec": removed,
        "removed_pct": removed / total * 100 if total > 0 else 0,
        "output_path": str(output_path),
    })


@app.get("/status/{task_id}")
def get_status(task_id: str):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return {k: t.get(k) for k in [
        "status", "progress", "log", "original_duration",
        "output_duration", "removed_sec", "removed_pct", "detail",
    ]}


@app.get("/download/{task_id}")
def download(task_id: str):
    t = tasks.get(task_id)
    if not t or t["status"] != "done":
        raise HTTPException(status_code=404, detail="Not ready")
    p = Path(t["output_path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail="File missing")
    return FileResponse(
        str(p), media_type="video/mp4", filename="output_cut.mp4",
        headers={"Access-Control-Allow-Origin": "*"},
    )
