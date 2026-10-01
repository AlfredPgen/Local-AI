r"""Save the transcripts of every video on YouTube channels as Markdown for data_prep.py.

    python youtube_transcripts.py @statquest
    python youtube_transcripts.py @statquest https://www.youtube.com/@iBiology --max-videos 200
    python youtube_transcripts.py --channels-file youtube_channels.txt
    python youtube_transcripts.py --channels-file youtube_channels.txt --whisper   # transcribe the rest

A channel can be given as @handle, handle, channel URL, channel ID (UC...),
playlist URL or single video URL. Videos are processed newest first; the
transcript of each comes from, in this order:

  1. English captions written by the creator;
  2. YouTube's automatic English captions, if they are punctuated (recent
     videos usually are);
  3. Whisper speech recognition (faster-whisper) of the audio track, only with
     --whisper. Without it such videos are marked "queued_whisper" and a later
     run with --whisper transcribes them. Whisper runs on the GPU only when the
     GPU is idle (never next to a training run), otherwise on 4 CPU threads.

Only on-topic videos are kept: the title and transcript must match the terms in
keywords.txt (data_prep.py's keyword test) at least 10 times, with at
least 3 different terms and at least 4 matches per 1,000 words, so a channel's
recurring intro alone does not qualify (--keywords none keeps everything). A video
that needs Whisper is transcribed only if its title or description mentions a
keyword at all. Shorts, live streams, videos under --min-minutes and
non-English videos are skipped.

No video files are downloaded; the audio track needed by Whisper is deleted
as soon as its transcript is written (or fails).

Output:
  <out>\<channel>\<date> <title> [<id>].md   title, a line naming the channel,
      then the transcript in paragraphs (no timestamps or [Music] tags)
  <out>\..\_<out name>_meta\<channel>_report.tsv    one row per video: status,
      transcript source, date, duration, words, licence, file
Reruns skip finished videos, so new uploads are added by running the command
again. If YouTube asks to "confirm you're not a bot", the run stops; wait a few
hours, or pass --cookies-from-browser firefox (this uses your YouTube login).

YouTube's terms restrict downloading content; the licence column shows which
videos are Creative Commons, and --creative-commons-only keeps only those.
"""

import argparse
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "ai_training_data", "lectures")
DEFAULT_KEYWORDS = os.path.join(HERE, "keywords.txt")
FINAL = {"ok", "not_english", "too_short", "unavailable", "not_cc", "empty", "live", "off_topic"}
TSV_COLUMNS = ["video_id", "status", "source", "title", "date", "duration_s", "words", "keyword_hits", "licence",
               "file"]
BOT_CHECK = ("confirm you", "not a bot", "sign in to confirm")
UNAVAILABLE = ("private video", "members-only", "join this channel", "video unavailable", "has been removed",
               "is not available", "please sign in",
               "age-restricted", "confirm your age", "premieres in", "copyright", "this live event")


class BotCheck(Exception):
    pass


# ---------------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------------
def ydl(cookies=None, **extra):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "skip_download": True,
            "js_runtimes": {"node": {}}, "sleep_interval_requests": 0.5, "retries": 3, "extractor_retries": 2}
    if cookies:
        opts["cookiesfrombrowser"] = (cookies,)
    opts.update(extra)
    return yt_dlp.YoutubeDL(opts)


AGE_LIMITED = ("confirm your age", "age-restricted", "inappropriate for some users")


def classify_error(exc):
    text = str(exc).lower()
    if any(a in text for a in AGE_LIMITED):  # "Sign in to confirm your age": one video, not a bot block
        return "unavailable"
    if any(b in text for b in BOT_CHECK):
        raise BotCheck(str(exc)[:200])
    return "unavailable" if any(u in text for u in UNAVAILABLE) else "error"


def source_url(spec, tab="videos"):
    s = spec.strip()
    if s.startswith("http"):
        if re.search(r"(watch\?v=|youtu\.be/|[?&]list=|/shorts/)", s):
            return s
        s = s.rstrip("/")
        return s if re.search(r"/(videos|streams|playlists|featured)$", s) else f"{s}/{tab}"
    if re.fullmatch(r"UC[\w-]{22}", s):
        return f"https://www.youtube.com/channel/{s}/{tab}"
    return f"https://www.youtube.com/@{s.lstrip('@').replace(' ', '')}/{tab}"


def list_videos(spec, cookies, max_videos):
    """(channel dict, [video entries]) for a channel, playlist or video."""
    import yt_dlp
    url = source_url(spec)
    try:
        with ydl(cookies, extract_flat="in_playlist") as y:
            info = y.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        classify_error(exc)
        if spec.startswith("http") or re.fullmatch(r"UC[\w-]{22}", spec):
            raise
        info = _find_channel_by_name(spec.lstrip("@"), cookies)  # "StatQuest with Josh Starmer" -> its channel
    entries = []
    stack = [info] if not info.get("entries") else list(info["entries"])
    while stack:
        e = stack.pop(0)
        if e.get("entries"):
            stack[:0] = list(e["entries"])
        elif e.get("id") and len(e["id"]) == 11:
            entries.append(e)
    seen, unique = set(), []
    for e in entries:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique.append(e)
    handle = (info.get("uploader_id") or info.get("channel_id") or info.get("id") or spec).lstrip("@")
    channel = {"name": info.get("channel") or info.get("uploader") or info.get("title") or spec,
               "key": safe_name(handle, 60)}
    return channel, unique[:max_videos] if max_videos else unique


def _find_channel_by_name(name, cookies):
    with ydl(cookies, extract_flat="in_playlist") as y:
        found = y.extract_info(f"ytsearch20:{name}", download=False)
    wanted = re.sub(r"\W", "", name).lower()
    for e in found.get("entries") or []:
        if re.sub(r"\W", "", e.get("channel") or e.get("uploader") or "").lower() == wanted and e.get("channel_id"):
            print(f"  '{name}' is channel {e.get('channel')} ({e['channel_id']})")
            with ydl(cookies, extract_flat="in_playlist") as y:
                return y.extract_info(f"https://www.youtube.com/channel/{e['channel_id']}/videos", download=False)
    raise LookupError(f"no channel named '{name}' found; give its exact @handle or URL")


def pick_track(tracks, prefer):
    """(language, json3 url) of the first English track in `prefer` order."""
    langs = [lang for lang in prefer if lang in tracks]
    langs += sorted(k for k in tracks if (k == "en" or k.startswith("en-")) and k not in langs)
    for lang in langs:
        fmt = next((f for f in tracks[lang] if f.get("ext") == "json3"), None)
        if fmt:
            return lang, fmt["url"]
    return None, None


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
SOUND_TAG_RE = re.compile(r"\[[^\]\n]{0,40}\]|\((?:[^)]{0,25}\b(?:music|applause|laugh\w*|inaudible|silence|"
                          r"cheer\w*|chuckl\w*))\)|♪[^♪\n]{0,200}♪|♪", re.I)
SENTENCE_RE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


def json3_text(raw):
    data = json.loads(raw)
    lines = ["".join(s.get("utf8", "") for s in ev["segs"]) for ev in data.get("events", []) if ev.get("segs")]
    return "\n".join(lines)


def paragraphs(text, max_chars=700):
    """Speaker changes (">>") start a paragraph; otherwise sentences are grouped
    into paragraphs of about max_chars."""
    text = SOUND_TAG_RE.sub(" ", html.unescape(text).replace("\u200b", ""))
    out = []
    for block in re.split(r"\s*>>\s*", text):
        block = " ".join(block.split())
        if not re.search(r"\w", block):
            continue
        current = ""
        for sentence in SENTENCE_RE.split(block):
            if current and len(current) + len(sentence) > max_chars:
                out.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            out.append(current)
    return "\n\n".join(out)


def punctuated(text):
    words = len(text.split())
    return words < 60 or len(re.findall(r"[.!?](?:\s|$)", text)) * 40 >= words


def safe_name(text, limit=90):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text or "untitled")
    return " ".join(name.split())[:limit].rstrip(" .") or "untitled"


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------
def gpu_busy():
    """True if another job uses the GPU (a training run holds GBs of memory);
    None without an NVIDIA GPU. Three samples, so a moment of desktop use does
    not count."""
    samples = []
    try:
        for k in range(3):
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
            samples.append([float(x) for x in out.strip().splitlines()[0].split(",")])
            if k < 2:
                time.sleep(1)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    util = sorted(s[0] for s in samples)[1]
    mem = sorted(s[1] for s in samples)[1]
    return mem > 2500 or util > 60


def _cuda_dll_dirs():
    """faster-whisper (CTranslate2) finds cuBLAS/cuDNN from the nvidia-* pip
    wheels only if their bin folders are on PATH before it is imported."""
    import site
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        for sub in (("nvidia", "cublas", "bin"), ("nvidia", "cudnn", "bin")):
            folder = os.path.join(base, *sub)
            if os.path.isdir(folder):
                os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(folder)


def gpu_memory_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class Whisper:
    """faster-whisper on the GPU when it is idle, else on the CPU. While on the
    GPU it watches the GPU memory: if another job (a training run) appears, it
    moves to the CPU at once, because two GPU jobs on this laptop crashed the
    driver before."""

    def __init__(self, device, model):
        self.requested_model = model
        if device == "auto":
            busy = gpu_busy()
            device = "cuda" if busy is False else "cpu"
            if busy:
                print("  The GPU is busy (training?), so Whisper runs on the CPU (4 threads, slower).")
        self.load(device)

    def load(self, device):
        if device == "cuda":
            _cuda_dll_dirs()
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        self.device = device
        self.name = self.requested_model or ("distil-large-v3" if device == "cuda" else "small.en")
        folder = os.path.join(os.path.expanduser("~"), ".cache", "faster-whisper", self.name.replace("/", "--"))
        if not os.path.isfile(os.path.join(folder, "model.bin")):
            print(f"  Downloading Whisper {self.name} (once) ...", flush=True)
            download_model(self.name, output_dir=folder)  # plain files: Windows needs admin rights for symlinks
        print(f"  Whisper {self.name} on {device}", flush=True)
        self.model = WhisperModel(folder, device=device, compute_type="float16" if device == "cuda" else "int8",
                                  cpu_threads=4)
        self.beam = 5 if device == "cuda" else 1
        self.baseline = gpu_memory_used() if device == "cuda" else None

    def check_gpu(self):
        if self.device != "cuda" or self.baseline is None:
            return
        used = gpu_memory_used()
        if used is not None and used > self.baseline + 1500:
            print(f"  Another GPU job appeared ({used - self.baseline:,.0f} MB more GPU memory in use): "
                  "Whisper moves to the CPU.", flush=True)
            self.model = None
            import gc
            gc.collect()
            self.requested_model = None
            self.load("cpu")

    def transcribe(self, audio_path, language):
        self.check_gpu()
        segments, info = self.model.transcribe(audio_path, language=language, beam_size=self.beam, vad_filter=True,
                                               condition_on_previous_text=False)
        if language is None and info.language != "en":
            return None, info.language
        parts, current, last_end = [], "", 0.0
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if current and (seg.start - last_end > 1.5 or (len(current) > 700 and current[-1] in ".!?")):
                parts.append(current)
                current = text
            else:
                current = f"{current} {text}".strip()
            last_end = seg.end
        if current:
            parts.append(current)
        return paragraphs("\n\n".join(parts).replace("\n\n", " >> ")), "en"


def download_audio(video_id, folder, cookies):
    with ydl(cookies, skip_download=False, format="bestaudio[ext=m4a]/bestaudio/best",
             outtmpl=os.path.join(folder, "%(id)s.%(ext)s")) as y:
        y.download([f"https://www.youtube.com/watch?v={video_id}"])
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(video_id)]
    return files[0] if files else None


# ---------------------------------------------------------------------------
# One video
# ---------------------------------------------------------------------------
def fetch(y, url, tries=4):
    """A caption file, with retries: network timeouts and server errors are
    retried after 5, 20 and 60 s; repeated 429 (too many requests) stops the
    run like the bot check, since continuing would only make it worse."""
    from yt_dlp.networking.exceptions import HTTPError, RequestError
    for attempt in range(tries):
        try:
            return y.urlopen(url).read()
        except HTTPError as exc:
            if exc.status not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                if exc.status == 429:
                    raise BotCheck("HTTP 429: too many requests") from exc
                raise
        except RequestError:
            if attempt == tries - 1:
                raise
        time.sleep((5, 20, 60)[min(attempt, 2)])


def process_video(entry, channel, args, whisper, tmp):
    import yt_dlp
    vid = entry["id"]
    row = {"video_id": vid, "status": "", "source": "", "title": entry.get("title") or "", "date": "",
           "duration_s": entry.get("duration") or "", "words": 0, "keyword_hits": "", "licence": "", "file": ""}
    if entry.get("live_status") in ("is_upcoming", "is_live"):
        return {**row, "status": "live"}
    if entry.get("duration") and entry["duration"] < args.min_minutes * 60:
        return {**row, "status": "too_short"}
    try:
        with ydl(args.cookies_from_browser) as y:
            info = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
            row.update(title=info.get("title") or row["title"], duration_s=info.get("duration") or row["duration_s"],
                       licence=info.get("license") or "YouTube standard", date=_date(info.get("upload_date")))
            if info.get("duration") and info["duration"] < args.min_minutes * 60:
                return {**row, "status": "too_short"}
            if args.creative_commons_only and "creative commons" not in row["licence"].lower():
                return {**row, "status": "not_cc"}
            lang = (info.get("language") or "").lower()
            auto = info.get("automatic_captions") or {}
            original = [k[:-5] for k in auto if k.endswith("-orig")]
            english = lang.startswith("en") or (not lang and (not original or any(o.startswith("en") for o in original)))
            text, source = None, ""
            m_lang, m_url = pick_track({k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"},
                                       ["en", "en-GB", "en-US"])
            if m_url:
                text, source = paragraphs(json3_text(fetch(y, m_url))), f"captions ({m_lang})"
            elif not english:
                return {**row, "status": "not_english"}
            else:
                a_lang, a_url = pick_track(auto, ["en-orig", "en"])
                if a_url:
                    raw = json3_text(fetch(y, a_url))
                    if punctuated(raw):
                        text, source = paragraphs(raw), "automatic captions"
    except yt_dlp.utils.DownloadError as exc:
        return {**row, "status": classify_error(exc), "source": str(exc)[:100]}
    if text is None:
        if args.kw is not None:  # no transcript yet: the title and description must mention the topic at all
            hits = args.kw_loose.evaluate(row["title"], info.get("description") or "")[2]
            if not hits:
                return {**row, "status": "off_topic", "source": "title and description", "keyword_hits": 0}
        if whisper is None:
            return {**row, "status": "queued_whisper"}
        try:
            audio = download_audio(vid, tmp, args.cookies_from_browser)
        except yt_dlp.utils.DownloadError as exc:
            return {**row, "status": classify_error(exc), "source": str(exc)[:100]}
        try:
            text, detected = whisper.transcribe(audio, "en" if lang.startswith("en") else None)
        finally:
            if audio and os.path.exists(audio):
                os.remove(audio)
        if text is None:
            return {**row, "status": "not_english", "source": f"whisper detected {detected}"}
        source = f"whisper {whisper.name}"
    words = len(text.split())
    if words < 100:
        return {**row, "status": "empty", "source": source, "words": words}
    if args.kw is not None:  # data_prep's keyword test, plus a density: intros and outros alone must not pass
        keep, _, hits, _ = args.kw.evaluate(row["title"], text)
        row["keyword_hits"] = hits
        if not keep or 1000 * hits / words < args.keyword_density:
            return {**row, "status": "off_topic", "source": source, "words": words}
    body = f"# {row['title']}\n\nTranscript of a video by {channel['name']}.\n\n{text}\n"  # dates stay in the tsv
    folder = os.path.join(args.out, channel["key"])
    os.makedirs(folder, exist_ok=True)
    name = f"{row['date'] or 'undated'} {safe_name(row['title'], 80)} [{vid}].md"
    path = os.path.join(folder, name)
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
    os.replace(path + ".part", path)
    return {**row, "status": "ok", "source": source, "words": words, "file": os.path.join(channel["key"], name)}


def _date(yyyymmdd):
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}" if yyyymmdd and len(yyyymmdd) == 8 else ""


def read_status(path):
    status = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            head = handle.readline().rstrip("\n").split("\t")
            for line in handle:
                row = dict(zip(head, line.rstrip("\n").split("\t")))
                status[row.get("video_id")] = row.get("status")
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("channels", nargs="*", help="@handle, channel/playlist/video URL, channel ID or channel name")
    p.add_argument("--channels-file", help="text file with one channel per line (# comments allowed)")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT})")
    p.add_argument("--max-videos", type=int, default=0, help="newest N videos per channel (0 = all)")
    p.add_argument("--min-minutes", type=float, default=3.0, help="skip shorter videos")
    p.add_argument("--whisper", action="store_true", help="transcribe videos without usable captions")
    p.add_argument("--whisper-device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="auto: the GPU when it is idle, otherwise the CPU")
    p.add_argument("--whisper-model", help="default distil-large-v3 on the GPU, small.en on the CPU")
    p.add_argument("--keywords", default=DEFAULT_KEYWORDS,
                   help="keep only videos whose title and transcript match these terms (data_prep's keyword test); "
                        "'none' keeps every video")
    p.add_argument("--keyword-min-hits", type=int, default=10, help="keyword matches (title counts 3 times)")
    p.add_argument("--keyword-min-distinct", type=int, default=3, help="different terms matched")
    p.add_argument("--keyword-density", type=float, default=4.0,
                   help="matches per 1,000 words (channel intros/outros alone stay below it)")
    p.add_argument("--creative-commons-only", action="store_true", help="keep only Creative Commons videos")
    p.add_argument("--delay", type=float, default=2.0, help="average seconds between videos (be gentle)")
    p.add_argument("--cookies-from-browser", help="e.g. firefox or edge, only if YouTube blocks the run")
    p.add_argument("--list-only", action="store_true", help="count videos and hours per channel, then stop")
    p.add_argument("--recheck-off-topic", action="store_true",
                   help="look again at videos rejected as off topic (after the keyword list has changed)")
    args = p.parse_args(argv)
    if args.channels_file:
        with open(args.channels_file, encoding="utf-8") as handle:
            args.channels += [ln.split("#", 1)[0].strip() for ln in handle if ln.split("#", 1)[0].strip()]
    if not args.channels:
        p.error("give at least one channel, or --channels-file")
    args.out = os.path.abspath(args.out)
    args.meta = os.path.join(os.path.dirname(args.out), f"_{os.path.basename(args.out)}_meta")
    args.kw = args.kw_loose = None
    if args.keywords and args.keywords.lower() != "none":
        sys.path.insert(0, HERE)
        import data_prep
        terms = data_prep.read_terms(args.keywords)
        args.kw = data_prep.KeywordFilter(terms, [], args.keyword_min_hits, args.keyword_min_distinct, 3)
        args.kw_loose = data_prep.KeywordFilter(terms, [], 1, 1, 3)
    return args


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_args(argv)
    import yt_dlp
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)
    except Exception:  # noqa: BLE001 - a courtesy only
        pass
    os.makedirs(args.meta, exist_ok=True)
    whisper = None
    tmp = tempfile.mkdtemp(prefix="yt_audio_")
    try:
        for spec in args.channels:
            print(f"\n[{spec}] listing videos ...", flush=True)
            try:
                channel, entries = list_videos(spec, args.cookies_from_browser, args.max_videos)
            except (LookupError, yt_dlp.utils.DownloadError) as exc:
                print(f"[{spec}] skipped: {str(exc).splitlines()[0][:200]}", flush=True)
                continue
            hours = sum(e.get("duration") or 0 for e in entries) / 3600
            print(f"[{spec}] {channel['name']}: {len(entries):,} videos, {hours:,.0f} hours", flush=True)
            if args.list_only:
                continue
            tsv = os.path.join(args.meta, f"{channel['key']}_report.tsv")
            old_name = os.path.join(args.meta, f"{channel['key']}.tsv")  # name used until 30 September 2026
            if os.path.isfile(old_name) and not os.path.exists(tsv):
                os.replace(old_name, tsv)
            status = read_status(tsv)
            final = FINAL if args.whisper else FINAL | {"queued_whisper"}
            if args.recheck_off_topic:
                final = final - {"off_topic"}
            todo = [e for e in entries if status.get(e["id"]) not in final]
            if args.whisper and whisper is None and any(status.get(e["id"]) in (None, "queued_whisper", "error")
                                                        for e in todo):
                whisper = Whisper(args.whisper_device, args.whisper_model)
            print(f"[{spec}] {len(entries) - len(todo):,} done earlier; processing {len(todo):,}", flush=True)
            counts = {}
            with open(tsv, "a", encoding="utf-8", newline="\n") as report:
                if not status:
                    report.write("\t".join(TSV_COLUMNS) + "\n")
                for k, entry in enumerate(todo, 1):
                    try:
                        row = process_video(entry, channel, args, whisper, tmp)
                    except (BotCheck, KeyboardInterrupt):
                        raise
                    except Exception as exc:  # noqa: BLE001 - one bad video must not stop the run; retried later
                        row = {c: "" for c in TSV_COLUMNS}
                        row.update(video_id=entry["id"], status="error", title=entry.get("title") or "",
                                   source=f"{type(exc).__name__}: {str(exc)[:80]}")
                    report.write("\t".join(" ".join(str(row[c]).split()) for c in TSV_COLUMNS) + "\n")
                    report.flush()
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
                    if k % 10 == 0 or k == len(todo):
                        print(f"  [{channel['key']}] {k:,}/{len(todo):,} {counts}", flush=True)
                    time.sleep(random.uniform(0.5, 1.5) * args.delay)
            print(f"[{spec}] done: {counts}; files in {os.path.join(args.out, channel['key'])}")
            if counts.get("queued_whisper") and not args.whisper:
                print(f"  {counts['queued_whisper']} videos have no usable captions: rerun with --whisper "
                      "(after training, so it can use the GPU).")
    except BotCheck as exc:
        print(f"\nYouTube asked to confirm this is not a bot ({exc}). Stopped; finished videos are saved. "
              "Wait a few hours and rerun, or add --cookies-from-browser firefox.")
        sys.exit(2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
