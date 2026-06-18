"""SilenceCut Backend тАФ FastAPI + FFmpeg v4"""
import os, uuid, asyncio, shutil, subprocess, json, time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SilenceCut API", version="4.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
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
    return {"status": "ok", "ffmpeg": shutil.which("ffmpeg") is not None}


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
        "params": {"threshold": threshold, "min_silence": min_silence,
                   "pad": pad, "max_keep": max_keep},
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
                task_id, Path(t["input_path"]),
                Path(t["task_dir"]) / "output.mp4", t["params"],
            ),
        )
    except Exception as e:
        tasks[task_id].update({"status": "error", "detail": str(e)})


def run_cmd(cmd, timeout=900):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return proc


def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def detect_silence(path: Path, threshold: float, min_silence: float):
    cmd = ["ffmpeg", "-i", str(path),
           "-af", f"silencedetect=noise={threshold}dB:duration={min_silence}",
           "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    total = get_duration(path)
    silence, cur_start = [], None
    for line in r.stderr.split("\n"):
        if "silence_start:" in line:
            try: cur_start = float(line.split("silence_start:")[-1].strip().split()[0])
            except: pass
        elif "silence_end:" in line:
            try:
                end = float(line.split("silence_end:")[-1].strip().split()[0])
                if cur_start is not None:
                    silence.append((cur_start, end))
                    cur_start = None
            except: pass
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


def normalize_to_mkv(input_path: Path, work_dir: Path) -> Path:
    """
    ╨Я╨╡╤А╨╡╨║╨╛╨┤╨╕╤А╤Г╨╡╨╝ ╨▓╨╡╤Б╤М ╤Д╨░╨╣╨╗ ╨▓ MKV ╤Б keyframe ╨║╨░╨╢╨┤╤Г╤О ╤Б╨╡╨║╤Г╨╜╨┤╤Г.
    baseline profile тАФ ╨╜╨╡╤В B-frames, seek ╨▓╤Б╨╡╨│╨┤╨░ ╨┐╨╛╨┐╨░╨┤╨░╨╡╤В ╨▓ IDR.
    keyint=30 ╨┐╤А╨╕ 30fps = keyframe ╨║╨░╨╢╨┤╤Г╤О ╤Б╨╡╨║╤Г╨╜╨┤╤Г = seek ╤Б ╤В╨╛╤З╨╜╨╛╤Б╤В╤М╤О 1 ╤Б╨╡╨║.
    ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ aac ╨▓ pcm_s16le ╨┤╨╗╤П ╨░╤Г╨┤╨╕╨╛ тАФ ╨▒╨╡╨╖ ╨┐╨╛╤В╨╡╤А╤М ╨┐╤А╨╕ ╨╜╨░╤А╨╡╨╖╨║╨╡.
    """
    norm_path = work_dir / "norm.mkv"
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts+igndts",
        "-i", str(input_path),
        # ╨Т╨╕╨┤╨╡╨╛: baseline ╨▒╨╡╨╖ B-frames, keyframe ╨║╨░╨╢╨┤╤Г╤О ╤Б╨╡╨║╤Г╨╜╨┤╤Г
        "-c:v", "libx264",
        "-preset", "ultrafast",       # ╨╝╨░╨║╤Б╨╕╨╝╨░╨╗╤М╨╜╨░╤П ╤Б╨║╨╛╤А╨╛╤Б╤В╤М ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╕
        "-crf", "18",                  # ╨▓╤Л╤Б╨╛╨║╨╛╨╡ ╨║╨░╤З╨╡╤Б╤В╨▓╨╛, ╨┐╨╛╤В╨╡╤А╨╕ ╨╝╨╕╨╜╨╕╨╝╨░╨╗╤М╨╜╤Л
        "-profile:v", "baseline",     # ╨╜╨╡╤В B-frames тЖТ seek ╨▓╤Б╨╡╨│╨┤╨░ ╤В╨╛╤З╨╜╤Л╨╣
        "-level", "4.0",
        "-x264-params", "keyint=30:min-keyint=30:scenecut=0",  # ╤Б╤В╤А╨╛╨│╨╕╨╣ keyint
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-vsync", "cfr",
        # ╨Р╤Г╨┤╨╕╨╛: pcm ╨▒╨╡╨╖ ╨┐╨╛╤В╨╡╤А╤М тЖТ ╨╜╨╡╤В ╨░╤А╤В╨╡╤Д╨░╨║╤В╨╛╨▓ ╨┐╤А╨╕ ╨╜╨░╤А╨╡╨╖╨║╨╡
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        str(norm_path),
    ]
    run_cmd(cmd, timeout=900)
    return norm_path


def process_with_ffmpeg(task_id: str, input_path: Path, output_path: Path, params: dict):
    t = tasks[task_id]
    work_dir = input_path.parent

    # ╨и╨Р╨У 1 тАФ ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨░╤Ж╨╕╤П ╨▓ MKV baseline ╤Б ╤З╨░╤Б╤В╤Л╨╝╨╕ keyframe
    t.update({"log": "╨Э╨╛╤А╨╝╨░╨╗╨╕╨╖╨░╤Ж╨╕╤П ╨▓╨╕╨┤╨╡╨╛ (1/4)тАж", "progress": 5})
    norm_path = normalize_to_mkv(input_path, work_dir)

    # ╨и╨Р╨У 2 тАФ ╨░╨╜╨░╨╗╨╕╨╖ ╤В╨╕╤И╨╕╨╜╤Л ╨╜╨░ ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨╛╨▓╨░╨╜╨╜╨╛╨╝ ╤Д╨░╨╣╨╗╨╡
    t.update({"log": "╨Р╨╜╨░╨╗╨╕╨╖ ╨░╤Г╨┤╨╕╨╛ (2/4)тАж", "progress": 35})
    silence, total = detect_silence(norm_path, params["threshold"], params["min_silence"])
    segs = build_loud(silence, total, params["pad"], params["max_keep"])
    if not segs:
        raise RuntimeError("╨Э╨╡╤В ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╤Б╨╡╨│╨╝╨╡╨╜╤В╨╛╨▓. ╨Я╨╛╨┐╤А╨╛╨▒╤Г╨╣ ╤Б╨╜╨╕╨╖╨╕╤В╤М ╨┐╨╛╤А╨╛╨│ ╨│╤А╨╛╨╝╨║╨╛╤Б╤В╨╕.")

    out_dur = sum(e - s for s, e in segs)
    t.update({"log": f"╨б╨╡╨│╨╝╨╡╨╜╤В╨╛╨▓: {len(segs)}, ╨╜╨░╤А╨╡╨╖╨░╤О (3/4)тАж", "progress": 45})

    # ╨и╨Р╨У 3 тАФ ╨╜╨░╤А╨╡╨╖╨║╨░ ╤З╨╡╤А╨╡╨╖ -ss/-t -c copy (norm.mkv baseline тЖТ seek ╤В╨╛╤З╨╜╤Л╨╣)
    seg_paths = []
    n = len(segs)
    for i, (start, end) in enumerate(segs):
        seg_path = work_dir / f"seg_{i:04d}.mkv"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.4f}",
            "-i", str(norm_path),
            "-t", f"{end - start:.4f}",
            "-c", "copy",             # ╨┐╤А╨╛╤Б╤В╨╛ ╨║╨╛╨┐╨╕╤А╤Г╨╡╨╝ тАФ baseline ╨│╨░╤А╨░╨╜╤В╨╕╤А╤Г╨╡╤В ╤В╨╛╤З╨╜╤Л╨╣ seek
            str(seg_path),
        ]
        run_cmd(cmd, timeout=120)
        seg_paths.append(seg_path)
        t.update({
            "log": f"╨Э╨░╤А╨╡╨╖╨║╨░ {i+1}/{n}тАж",
            "progress": 45 + int((i + 1) / n * 35),
        })

    t.update({"log": "╨б╨║╨╗╨╡╨╣╨║╨░ тЖТ MP4 (4/4)тАж", "progress": 82})

    # ╨и╨Р╨У 4 тАФ ╤Б╨║╨╗╨╡╨╣╨║╨░ ╤З╨╡╤А╨╡╨╖ concat demuxer + ╤Д╨╕╨╜╨░╨╗╤М╨╜╨╛╨╡ ╨║╨╛╨┤╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨▓ MP4
    list_path = work_dir / "segments.txt"
    with open(list_path, "w") as f:
        for p in seg_paths:
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    run_cmd(cmd, timeout=600)

    # ╨з╨╕╤Б╤В╨╕╨╝ ╨▓╤А╨╡╨╝╨╡╨╜╨╜╤Л╨╡ ╤Д╨░╨╣╨╗╤Л
    norm_path.unlink(missing_ok=True)
    for p in seg_paths:
        p.unlink(missing_ok=True)
    list_path.unlink(missing_ok=True)

    removed = total - out_dur
    t.update({
        "status": "done", "progress": 100, "log": "╨У╨╛╤В╨╛╨▓╨╛!",
        "original_duration": total, "output_duration": out_dur,
        "removed_sec": removed,
        "removed_pct": removed / total * 100 if total > 0 else 0,
        "output_path": str(output_path),
    })


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    threshold: float = Form(-40.0),
    min_silence: float = Form(0.5),
):
    task_id = str(uuid.uuid4())
    task_dir = WORK_DIR / task_id
    task_dir.mkdir(parents=True)
    ext = Path(file.filename).suffix.lower() or ".mp4"
    input_path = task_dir / ("input" + ext)
    with open(input_path, "wb") as fo:
        shutil.copyfileobj(file.file, fo)
    try:
        total = get_duration(input_path)
        silence, _ = detect_silence(input_path, threshold, min_silence)
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-af", "asetnsamples=n=2400,astats=metadata=1:reset=1",
            "-f", "null", "-",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        rms_values = []
        for line in r.stderr.split("\n"):
            if "RMS level dB" in line:
                try:
                    val = float(line.split(":")[-1].strip())
                    val = max(-60.0, min(0.0, val))
                    rms_values.append(round((val + 60) / 60, 4))
                except:
                    pass
        if len(rms_values) > 800:
            step = len(rms_values) / 800
            rms_values = [rms_values[int(i * step)] for i in range(800)]
        return {
            "duration": total,
            "waveform": rms_values,
            "silence": [{"start": s, "end": e} for s, e in silence],
            "window_size": total / len(rms_values) if rms_values else 0,
        }
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


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
