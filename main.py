"""SilenceCut Backend — FastAPI + FFmpeg"""
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
    return {"status": "ok", "ffmpeg": ffmpeg_path is not None}


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
        "status": "processing", "progress": 0, "log": "Файл получен…",
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


def normalize_input(input_path: Path, work_dir: Path, task_id: str) -> Path:
    """
    Шаг 1: перекодировать в rawvideo + pcm_s16le в AVI.
    AVI не имеет ограничений на кодеки и PTS, trim на нём работает идеально.
    rawvideo = несжатые кадры, никаких проблем с декодером при trim.
    """
    raw_path = work_dir / "raw.avi"
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts+igndts",
        "-i", str(input_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "rawvideo",          # несжатое видео — trim всегда работает
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-c:a", "pcm_s16le",         # несжатое аудио
        "-vsync", "cfr",
        str(raw_path),
    ]
    run_cmd(cmd, timeout=900)
    return raw_path


def cut_to_segments(raw_path: Path, segs, work_dir: Path, task_id: str) -> list:
    """
    Шаг 2: вырезать каждый сегмент отдельным ffmpeg-вызовом через -ss/-to.
    Это самый надёжный способ — каждый вызов независим, нет filter_complex.
    """
    seg_paths = []
    for i, (start, end) in enumerate(segs):
        seg_path = work_dir / f"seg_{i:04d}.avi"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_path),
            "-ss", f"{start:.4f}",
            "-to", f"{end:.4f}",
            "-c", "copy",            # просто копируем — rawvideo всегда seekable
            str(seg_path),
        ]
        run_cmd(cmd, timeout=300)
        seg_paths.append(seg_path)
    return seg_paths


def concat_segments(seg_paths: list, work_dir: Path, output_path: Path):
    """
    Шаг 3: склеить через concat demuxer (список файлов) и закодировать в MP4.
    concat demuxer не требует filter_complex и работает с любыми форматами.
    """
    # Записываем список сегментов
    list_path = work_dir / "segments.txt"
    with open(list_path, "w") as f:
        for p in seg_paths:
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
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

    # ШАГ 1 — нормализация в rawvideo AVI
    t.update({"log": "Нормализация видео…", "progress": 5})
    raw_path = normalize_input(input_path, work_dir, task_id)

    # ШАГ 2 — анализ тишины
    t.update({"log": "Анализ аудио…", "progress": 30})
    silence, total = detect_silence(raw_path, params["threshold"], params["min_silence"])
    t.update({"log": f"Найдено пауз: {len(silence)}", "progress": 45})

    segs = build_loud(silence, total, params["pad"], params["max_keep"])
    if not segs:
        raise RuntimeError("Нет активных сегментов. Попробуй снизить порог громкости.")
    out_dur = sum(e - s for s, e in segs)
    t.update({"log": f"Сегментов: {len(segs)}, вырезаю…", "progress": 50})

    # ШАГ 3 — вырезаем каждый сегмент
    seg_paths = cut_to_segments(raw_path, segs, work_dir, task_id)
    t.update({"log": "Склейка → MP4…", "progress": 75})

    # ШАГ 4 — склейка и финальное кодирование
    concat_segments(seg_paths, work_dir, output_path)

    # Чистим временные файлы
    raw_path.unlink(missing_ok=True)
    for p in seg_paths:
        p.unlink(missing_ok=True)

    removed = total - out_dur
    t.update({
        "status": "done", "progress": 100, "log": "Готово!",
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
