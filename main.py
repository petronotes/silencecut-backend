"""SilenceCut Backend тАФ FastAPI + FFmpeg"""
import os, uuid, asyncio, shutil, subprocess, json, time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SilenceCut API", version="1.0.0")
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
    ffmpeg_path = shutil.which("ffmpeg")
    return {"status": "ok", "ffmpeg": ffmpeg_path is not None, "ffmpeg_path": ffmpeg_path or "not found"}


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
        "params": {"threshold": threshold, "min_silence": min_silence, "pad": pad, "max_keep": max_keep},
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


def run_cmd(cmd, timeout=600):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1200:])
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


def remux_to_rawvideo(input_path: Path, work_dir: Path) -> Path:
    """
    Step 1: ╨Я╨╛╨╗╨╜╨╛╨╡ ╨┐╨╡╤А╨╡╨║╨╛╨┤╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨▓ yuv420p + pcm_s16le (╨▒╨╡╨╖ ╤Б╨╢╨░╤В╨╕╤П).
    ╨н╤В╨╛ ╤А╨╡╤И╨░╨╡╤В ╨Т╨б╨Х ╨┐╤А╨╛╨▒╨╗╨╡╨╝╤Л ╤Б Android h264/avc1:
    - ╨▒╨╕╤В╤Л╨╣ PTS
    - VBR / VFR
    - ╨╜╨╡╤З╤С╤В╨╜╤Л╨╡ ╤А╨░╨╖╨╝╨╡╤А╤Л
    - rotation metadata
    - avc1 vs H.264 Annex B
    ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ .mkv тАФ ╨╛╨╜ ╨┐╨╛╨┤╨┤╨╡╤А╨╢╨╕╨▓╨░╨╡╤В ╨╗╤О╨▒╤Л╨╡ ╨║╨╛╨┤╨╡╨║╨╕ ╨▒╨╡╨╖ ╨╛╨│╤А╨░╨╜╨╕╤З╨╡╨╜╨╕╨╣ mp4.
    """
    raw_path = work_dir / "raw.mkv"
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts+igndts",   # ╨╕╨│╨╜╨╛╤А╨╕╤А╨╛╨▓╨░╤В╤М ╨┐╨╗╨╛╤Е╨╕╨╡ DTS, ╨│╨╡╨╜╨╡╤А╨╕╤А╨╛╨▓╨░╤В╤М PTS
        "-i", str(input_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ╤З╤С╤В╨╜╤Л╨╡ ╤А╨░╨╖╨╝╨╡╤А╤Л
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "15",  # ╨▒╤Л╤Б╤В╤А╨╛╨╡, ╨┐╨╛╤З╤В╨╕ ╨▒╨╡╨╖ ╨┐╨╛╤В╨╡╤А╤М
        "-pix_fmt", "yuv420p",
        "-r", "30",                    # CFR 30fps тАФ ╤Г╨▒╨╕╨▓╨░╨╡╤В VFR
        "-c:a", "pcm_s16le",          # ╨╜╨╡╤Б╨╢╨░╤В╨╛╨╡ ╨░╤Г╨┤╨╕╨╛ тАФ ╨╝╨░╨║╤Б╨╕╨╝╨░╨╗╤М╨╜╨░╤П ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╤М
        "-vsync", "cfr",              # ╤Д╨╛╤А╤Б╨╕╤А╨╛╨▓╨░╤В╤М ╨┐╨╛╤Б╤В╨╛╤П╨╜╨╜╤Л╨╣ FPS
        "-async", "1",               # ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╨╛╨▓╨░╤В╤М ╨░╤Г╨┤╨╕╨╛
        str(raw_path),
    ]
    run_cmd(cmd, timeout=600)
    return raw_path


def cut_segments(raw_path: Path, segs, output_path: Path):
    """
    Step 2: ╨а╨╡╨╖╨░╤В╤М ╨╕ ╤Б╨║╨╗╨╡╨╕╨▓╨░╤В╤М ╤Г╨╢╨╡ ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨╛╨▓╨░╨╜╨╜╤Л╨╣ ╤Д╨░╨╣╨╗.
    ╨в╨╡╨┐╨╡╤А╤М trim ╤А╨░╨▒╨╛╤В╨░╨╡╤В ╨║╨╛╤А╤А╨╡╨║╤В╨╜╨╛ тАФ ╨┤╨░╨╜╨╜╤Л╨╡ ╤Г╨╢╨╡ CFR + ╤З╨╕╤Б╤В╤Л╨╣ PTS.
    """
    n = len(segs)
    pv = [
        f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}]"
        for i, (s, e) in enumerate(segs)
    ]
    pa = [
        f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]"
        for i, (s, e) in enumerate(segs)
    ]
    inputs_v = "".join(f"[v{i}]" for i in range(n))
    inputs_a = "".join(f"[a{i}]" for i in range(n))
    cat = f"{inputs_v}concat=n={n}:v=1:a=0[outv];{inputs_a}concat=n={n}:v=0:a=1[outa]"
    fc = ";".join(pv + pa + [cat])

    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_path),
        "-filter_complex", fc,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    run_cmd(cmd, timeout=600)


def process_with_ffmpeg(task_id: str, input_path: Path, output_path: Path, params: dict):
    t = tasks[task_id]
    work_dir = input_path.parent

    # STEP 1 тАФ ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨░╤Ж╨╕╤П (╨╕╤Б╨┐╤А╨░╨▓╨╗╤П╨╡╤В ╨▓╤Б╨╡ Android-╨┐╤А╨╛╨▒╨╗╨╡╨╝╤Л)
    t.update({"log": "╨Э╨╛╤А╨╝╨░╨╗╨╕╨╖╨░╤Ж╨╕╤П ╨▓╨╕╨┤╨╡╨╛ (Android fix)тАж", "progress": 10})
    raw_path = remux_to_rawvideo(input_path, work_dir)

    # STEP 2 тАФ ╨░╨╜╨░╨╗╨╕╨╖ ╤В╨╕╤И╨╕╨╜╤Л ╨╜╨░ ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨╛╨▓╨░╨╜╨╜╨╛╨╝ ╤Д╨░╨╣╨╗╨╡
    t.update({"log": "╨Р╨╜╨░╨╗╨╕╨╖ ╨░╤Г╨┤╨╕╨╛тАж", "progress": 35})
    silence, total = detect_silence(raw_path, params["threshold"], params["min_silence"])
    t.update({"log": f"╨Э╨░╨╣╨┤╨╡╨╜╨╛ ╨┐╨░╤Г╨╖: {len(silence)}", "progress": 50})

    segs = build_loud(silence, total, params["pad"], params["max_keep"])
    t.update({"log": f"╨б╨╡╨│╨╝╨╡╨╜╤В╨╛╨▓ ╨║ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤О: {len(segs)}", "progress": 60})

    if not segs:
        raise RuntimeError("╨Э╨╡╤В ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╤Б╨╡╨│╨╝╨╡╨╜╤В╨╛╨▓. ╨Я╨╛╨┐╤А╨╛╨▒╤Г╨╣ ╤Б╨╜╨╕╨╖╨╕╤В╤М ╨┐╨╛╤А╨╛╨│ ╨│╤А╨╛╨╝╨║╨╛╤Б╤В╨╕.")

    out_dur = sum(e - s for s, e in segs)

    # STEP 3 тАФ ╨╝╨╛╨╜╤В╨░╨╢
    t.update({"log": "╨Ь╨╛╨╜╤В╨░╨╢ тЖТ MP4тАж", "progress": 65})
    cut_segments(raw_path, segs, output_path)

    # ╨г╨┤╨░╨╗╤П╨╡╨╝ ╨┐╤А╨╛╨╝╨╡╨╢╤Г╤В╨╛╤З╨╜╤Л╨╣ ╤Д╨░╨╣╨╗
    raw_path.unlink(missing_ok=True)

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
