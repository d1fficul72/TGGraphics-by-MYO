"""
Telegram бот: эмодзи-мозаика + кружки из видео + GIF из видео + коллаж из фото

Зависимости:
    pip install python-telegram-bot Pillow opencv-python-headless numpy

Системные требования:
    ffmpeg

Переменные окружения:
    BOT_TOKEN
"""

import io
import os
import csv
import time
import shutil
import asyncio
import colorsys
import logging
import datetime
import tempfile
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputSticker, MessageEntity
from telegram.constants import StickerFormat, StickerType
from telegram.error import NetworkError, TimedOut, RetryAfter, BadRequest
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, ConversationHandler, filters,
)
from telegram.request import HTTPXRequest

# ── Логирование ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_logs.txt", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)


async def _tg_retry(coro_fn, *args, retries: int = 4, **kwargs):
    """Выполняет Telegram API вызов с повторами при сетевых ошибках."""
    delay = 2.0
    for attempt in range(retries):
        try:
            return await coro_fn(*args, **kwargs)
        except RetryAfter as e:
            wait = e.retry_after + 1
            log.warning(f"RetryAfter {wait}s (attempt {attempt+1})")
            await asyncio.sleep(wait)
        except (TimedOut, NetworkError) as e:
            if attempt == retries - 1:
                raise
            log.warning(f"Network error ({type(e).__name__}), retry {attempt+1} in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError("_tg_retry exhausted")


async def global_error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    err = ctx.error
    if isinstance(err, (TimedOut, NetworkError)):
        log.warning(f"Network error (ignored): {err}")
        return
    if isinstance(err, RetryAfter):
        log.warning(f"RetryAfter {err.retry_after}s (ignored at top level)")
        return
    log.exception(f"Unhandled exception: {err}", exc_info=err)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Произошла ошибка. Попробуй ещё раз или напиши /start",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("← Главное меню", callback_data="back_menu")
                ]]),
            )
        except Exception:
            pass


USERS_LOG   = str(Path(__file__).parent / "data" / "users_log.csv")    # полный лог действий
USERS_STATS = str(Path(__file__).parent / "data" / "users_stats.csv")  # сводная таблица по пользователям

REQUIRED_CHANNEL = "@myooffical"   # канал, на который нужна подписка

# Столбцы сводной таблицы
_STATS_COLS = [
    "id", "username", "имя",
    "первый_визит", "последний_визит",
    "эмодзи_мозаика", "кружок", "gif", "коллаж", "стикерпак",
    "всего_завершено",
]

# Какое ключевое слово в action → какая функция засчитывается
_FEATURE_MAP = {
    "создал пак":          "эмодзи_мозаика",
    "кружок отправлен":    "кружок",
    "GIF готов":           "gif",
    "коллаж":              "коллаж",
    "стикерпак":           "стикерпак",
}


def _detect_feature(action: str):
    for key, feat in _FEATURE_MAP.items():
        if key in action:
            return feat
    return None


def _update_stats(user, name: str, ts: str, feature: str):
    """Читает users_stats.csv, обновляет строку пользователя, записывает обратно."""
    stats_path = Path(USERS_STATS)
    rows: list[dict] = []

    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []

    uid = str(user.id)
    row = next((r for r in rows if r.get("id") == uid), None)

    if row is None:
        row = {col: "0" for col in _STATS_COLS}
        row["id"]            = uid
        row["первый_визит"]  = ts
        rows.append(row)

    row["username"]          = user.username or ""
    row["имя"]               = name
    row["последний_визит"]   = ts
    row[feature]             = str(int(row.get(feature) or "0") + 1)
    row["всего_завершено"]   = str(int(row.get("всего_завершено") or "0") + 1)

    try:
        with open(stats_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_STATS_COLS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        log.warning(f"users_stats error: {e}")


def log_action(user, action: str):
    name     = f"{user.first_name or ''} {user.last_name or ''}".strip()
    username = f"@{user.username}" if user.username else "без username"
    log.info(f"👤 {name} ({username}, id={user.id}) | {action}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Полный лог всех действий
    try:
        log_path    = Path(USERS_LOG)
        file_exists = log_path.exists()
        with open(log_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["дата_время", "id", "username", "имя", "действие"])
            writer.writerow([ts, user.id, user.username or "", name, action])
    except Exception as e:
        log.warning(f"users_log error: {e}")

    # 2. Сводная статистика — только при успешном завершении функции
    feature = _detect_feature(action)
    if feature:
        _update_stats(user, name, ts, feature)


# ── Константы ────────────────────────────────────────────────────────────
BOT_TOKEN        = os.environ["BOT_TOKEN"]
FFMPEG           = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE          = os.environ.get("FFPROBE_PATH", "ffprobe")
WELCOME_GIF      = str(Path(__file__).parent / "assets" / "welcome.gif")
WELCOME_GIF2     = str(Path(__file__).parent / "assets" / "welcome2.gif")
MENU_GIF         = str(Path(__file__).parent / "assets" / "gifmenu.gif")

# Эмодзи-мозаика
STICKER_SIZE     = 100
MAX_STICKERS     = 200
MAX_COLS         = 12
MAX_VIDEO_FRAMES = 90   # 3 сек × 30 fps — плавная анимация
MAX_VIDEO_SEC    = 3.0
WIDE_COLS        = 12   # «широкий» вариант — вся ширина экрана

# Кружок
CIRCLE_MAX_DURATION = 60
CIRCLE_NOTE_SIZE    = 384

# GIF
GIF_MAX_DURATION = 30
GIF_QUALITY_PRESETS = {
    "low":    (320, 12, "— Лёгкая",   15),   # целевой макс. размер в МБ
    "medium": (480, 18, "— Стандарт", 20),
    "high":   (640, 24, "— Высокое",  30),
}
GIF_TELEGRAM_LIMIT = 50  # Telegram не принимает документы > 50 МБ

# Коллаж
CANVAS_W = 1080
COLLAGE_TEMPLATES = {
    "2h":   {"label": "2 фото рядом",           "count": 2, "ratio": 16/9,  "cells": [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)]},
    "2v":   {"label": "2 фото стопкой",         "count": 2, "ratio": 9/16,  "cells": [(0.0, 0.0, 1.0, 0.5), (0.0, 0.5, 1.0, 0.5)]},
    "3h":   {"label": "3 фото рядом",           "count": 3, "ratio": 16/7,  "cells": [(0.0, 0.0, 1/3, 1.0), (1/3, 0.0, 1/3, 1.0), (2/3, 0.0, 1/3, 1.0)]},
    "1l2r": {"label": "Большое + 2 справа",     "count": 3, "ratio": 4/3,   "cells": [(0.0, 0.0, 0.6, 1.0), (0.6, 0.0, 0.4, 0.5), (0.6, 0.5, 0.4, 0.5)]},
    "2l1r": {"label": "2 слева + большое",      "count": 3, "ratio": 4/3,   "cells": [(0.0, 0.0, 0.4, 0.5), (0.0, 0.5, 0.4, 0.5), (0.4, 0.0, 0.6, 1.0)]},
    "1t2b": {"label": "Большое + 2 снизу",      "count": 3, "ratio": 1/1,   "cells": [(0.0, 0.0, 1.0, 0.6), (0.0, 0.6, 0.5, 0.4), (0.5, 0.6, 0.5, 0.4)]},
    "2t1b": {"label": "2 сверху + большое",     "count": 3, "ratio": 1/1,   "cells": [(0.0, 0.0, 0.5, 0.4), (0.5, 0.0, 0.5, 0.4), (0.0, 0.4, 1.0, 0.6)]},
    "2x2":  {"label": "Сетка 2×2",             "count": 4, "ratio": 1/1,   "cells": [(x/2, y/2, 0.5, 0.5) for y in range(2) for x in range(2)]},
    "3x2":  {"label": "Сетка 3×2",             "count": 6, "ratio": 3/2,   "cells": [(x/3, y/2, 1/3, 0.5) for y in range(2) for x in range(3)]},
    "3x3":  {"label": "Сетка 3×3",             "count": 9, "ratio": 1/1,   "cells": [(x/3, y/3, 1/3, 1/3) for y in range(3) for x in range(3)]},
}
COLLAGE_FIT_OPTIONS = {
    "fit_crop": "✂️ Обрезать — заполнить клетку целиком",
    "fit_fit":  "◻ Вписать — сохранить пропорции",
}
COLLAGE_GAP_OPTIONS = {
    "gap0": ("Без отступов", 0),
    "gap1": ("Маленькие",   12),
    "gap2": ("Большие",     30),
}
COLLAGE_BG_OPTIONS = {
    "bg_white": "Белый фон",
    "bg_black": "Чёрный фон",
    "bg_blur":  "Размытое фото",
}

_user_locks: dict[int, asyncio.Lock] = {}

def get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]


GRID_OPTIONS = [
    # Широкие (вся ширина экрана, 12 колонок)
    (12,  1), (12,  2),
    # Маленькие (≤ 25 эмодзи)
    (2,   2), (2,   3), (3,   2),
    (3,   3), (3,   4), (4,   3),
    (4,   4), (4,   5), (5,   4),
    (5,   5),
    # Средние
    (8,   5), (10,  5), (12,  5),
    (5,   8), (8,   8),
    (10,  8), (12,  8),
    (5,  10), (10, 10), (12, 10),
    # Крупные
    (5,  13), (8,  13), (10, 13), (12, 13),
]

# Состояния разговора
(MAIN_MENU, EMOJI_WAIT_FILE, EMOJI_WAIT_BG_REMOVE, EMOJI_WAIT_ASPECT,
 EMOJI_WAIT_GRID, EMOJI_WAIT_CUSTOM_GRID, EMOJI_WAIT_PACK_NAME,
 CIRCLE_WAIT_VIDEO, GIF_WAIT_VIDEO, GIF_WAIT_QUALITY,
 COL_WAIT_PHOTOS, COL_WAIT_TEMPLATE, COL_WAIT_FIT, COL_WAIT_GAP, COL_WAIT_BG,
 STK_WAIT_MODE, STK_WAIT_SHADOW, STK_WAIT_PHOTO, STK_WAIT_BG, STK_WAIT_NAME,
 STK_VERT_SHADOW, STK_VERT_PHOTO, STK_VERT_BG, STK_VERT_NAME,
 STK_WAIT_ANIM_FILE,
 TXT_WAIT_TYPE, TXT_WAIT_FONT, TXT_WAIT_HEIGHT, TXT_WAIT_ANIM,
 TXT_WAIT_COLOR, TXT_WAIT_TEXT, TXT_WAIT_NAME) = range(32)

STK_SIZE = 512  # размер стикера (не путать с STICKER_SIZE=100 для эмодзи)

# ── Константы: Текст → эмодзи ────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
_FONT_DIR  = Path(os.environ.get("FONT_DIR", "/usr/share/fonts"))

def _find_font(win_name: str, linux_names: list) -> str:
    win_path = Path("C:/Windows/Fonts") / win_name
    if win_path.exists():
        return str(win_path)
    for name in linux_names:
        for p in _FONT_DIR.rglob(name):
            return str(p)
    return ""

FONT_PATHS = {
    "impact":      _find_font("impact.ttf",    ["Impact.ttf", "impact.ttf", "LiberationSans-Bold.ttf"]),
    "arial_black": _find_font("ariblk.ttf",    ["ArialBlack.ttf", "ariblk.ttf", "LiberationSans-Bold.ttf"]),
    "courier":     _find_font("cour.ttf",      ["CourierNew.ttf", "cour.ttf", "LiberationMono-Regular.ttf"]),
    "serif":       _find_font("timesbd.ttf",   ["TimesNewRoman-Bold.ttf", "timesbd.ttf", "LiberationSerif-Bold.ttf"]),
    "shadow_3d":   _find_font("impact.ttf",    ["Impact.ttf", "impact.ttf", "LiberationSans-Bold.ttf"]),
    "snap":        _find_font("SNAP____.TTF",  ["Snap.ttf", "snap.ttf", "LiberationSans-Bold.ttf"]),
    "stencil":     _find_font("STENCIL.TTF",   ["Stencil.ttf", "stencil.ttf", "LiberationSans-Bold.ttf"]),
    "outline":     _find_font("ariblk.ttf",    ["ArialBlack.ttf", "ariblk.ttf", "LiberationSans-Bold.ttf"]),
}
FONT_LABELS = {
    "impact":      "◼ Impact — жирный широкий",
    "arial_black": "▪ Arial Black — плотный",
    "courier":     "— Courier — моноширинный",
    "serif":       "◻ Serif — классический",
    "shadow_3d":   "▶ 3D Shadow — объёмный",
    "snap":        "○ Snap — блок/скруглённый",
    "stencil":     "▪ Stencil — трафарет",
    "outline":     "□ Outline — контур/полые буквы",
}
TXT_COLORS = {
    "white":  ((255, 255, 255, 255), "□ Белый"),
    "black":  ((15,  15,  15,  255), "■ Чёрный"),
    "silver": ((180, 180, 185, 255), "◻ Серебристый"),
    "gold":   ((200, 165, 80,  255), "◈ Золотой"),
    "blue":   ((90,  130, 190, 255), "○ Синий"),
    "red":    ((190, 75,  75,  255), "◆ Красный"),
    "green":  ((75,  155, 95,  255), "◇ Зелёный"),
    "beige":  ((210, 190, 155, 255), "▷ Бежевый"),
}
TXT_MAX_CHARS = {1: 16, 2: 10, 3: 7}
TXT_ANIM_OPTIONS = {
    "none":    "○ Без анимации (статичный)",
    "pulse":   "◉ Пульс — плавное мигание",
    "shimmer": "◈ Блик — световая волна",
    "wave":    "◻ Волна — буквы качаются",
    "fade":    "— Фейд — плавное появление",
    "rainbow": "▶ Радуга — цвета переливаются",
    "scan":    "▪ Скан — луч сверху вниз",
    "glitch":  "◆ Глитч — смещение полос",
}
PACK_WATERMARK = " @myooffical"   # добавляется к каждому паку

def _make_title(title: str) -> str:
    """Добавляет watermark к названию пака."""
    if len(title) + len(PACK_WATERMARK) <= 64:
        return title + PACK_WATERMARK
    return title[: 64 - len(PACK_WATERMARK)] + PACK_WATERMARK


# ════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ: кружок
# ════════════════════════════════════════════════════════════════════════

def convert_to_circle(input_path: str, output_path: str) -> bool:
    cmd = [
        FFMPEG, "-y",
        "-i", input_path,
        "-vf", f"crop=min(iw\\,ih):min(iw\\,ih),scale={CIRCLE_NOTE_SIZE}:{CIRCLE_NOTE_SIZE}",
        "-t", str(CIRCLE_MAX_DURATION),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg error: %s", result.stderr)
        return False
    return True


# ════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ: эмодзи-мозаика
# ════════════════════════════════════════════════════════════════════════

def sort_grids_by_aspect(img_w: int, img_h: int) -> list[tuple[int, int, float]]:
    photo_ratio = img_w / img_h
    results = []
    for cols, rows in GRID_OPTIONS:
        if cols * rows > MAX_STICKERS:
            continue
        grid_ratio = cols / rows
        score = abs(photo_ratio - grid_ratio) / photo_ratio
        results.append((cols, rows, score))
    results.sort(key=lambda x: x[2])
    return results


def slice_image(img, cols, rows):
    img = img.convert("RGBA")
    if cols == WIDE_COLS:
        # Масштабируем на 1 колонку шире, затем центрально кропаем — без сжатия
        scale_cols = WIDE_COLS + 1
        scaled = img.resize((scale_cols * STICKER_SIZE, rows * STICKER_SIZE), Image.LANCZOS)
        offset = ((scale_cols - cols) * STICKER_SIZE) // 2
        img = scaled.crop((offset, 0, offset + cols * STICKER_SIZE, rows * STICKER_SIZE))
    else:
        img = img.resize((cols * STICKER_SIZE, rows * STICKER_SIZE), Image.LANCZOS)
    cells = []
    for row in range(rows):
        for col in range(cols):
            box = (
                col * STICKER_SIZE, row * STICKER_SIZE,
                (col + 1) * STICKER_SIZE, (row + 1) * STICKER_SIZE,
            )
            cells.append(img.crop(box))
    return cells


def ensure_valid(cell):
    cell = cell.convert("RGBA").resize((STICKER_SIZE, STICKER_SIZE), Image.LANCZOS)
    arr = np.array(cell)
    if arr[:, :, 3].max() == 0:
        cell.putpixel((50, 50), (255, 255, 255, 2))
    return cell


def save_cells_png(cells, out_dir):
    paths = []
    for i, cell in enumerate(cells):
        p = out_dir / f"cell_{i:04d}.png"
        ensure_valid(cell).save(p, "PNG", optimize=True, compress_level=9)
        paths.append(p)
    return paths


def read_frames_ffmpeg(path: str) -> tuple[list, float]:
    """
    Извлекает кадры через ffmpeg с сохранением альфа-канала (WebM, WebP, MOV и др.).
    Возвращает (list of BGRA numpy arrays, target_fps).
    Корректно сохраняет прозрачность и скорость анимации.
    """
    # 1. Получаем fps оригинала
    try:
        r_fps = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, timeout=15,
        )
        num, den = r_fps.stdout.strip().split("/")
        orig_fps = float(num) / max(float(den), 1)
    except Exception:
        orig_fps = 24.0

    # 2. Получаем длительность
    try:
        r_dur = subprocess.run(
            [FFPROBE, "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, timeout=15,
        )
        orig_dur = float(r_dur.stdout.strip())
    except Exception:
        orig_dur = MAX_VIDEO_SEC

    # 3. Вычисляем целевой fps, чтобы уложиться в MAX_VIDEO_FRAMES кадров
    clip_dur   = min(orig_dur, MAX_VIDEO_SEC)
    target_fps = min(orig_fps, 30.0, MAX_VIDEO_FRAMES / max(clip_dur, 0.1))
    target_fps = max(round(target_fps, 3), 1.0)

    # 4. Извлекаем кадры в PNG (ffmpeg сохраняет альфа-канал)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_pat = str(Path(tmpdir) / "f_%04d.png")
        subprocess.run(
            [FFMPEG, "-y", "-i", str(path),
             "-t", str(clip_dur),
             "-vf", f"fps={target_fps}",
             out_pat],
            capture_output=True, timeout=60, check=True,
        )
        frame_files = sorted(Path(tmpdir).glob("f_*.png"))
        frames = []
        for fp in frame_files[:MAX_VIDEO_FRAMES]:
            arr = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            if arr.ndim == 2:                          # grayscale → BGRA
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
            elif arr.shape[2] == 3:                    # BGR → BGRA
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
            # 4-канальный PNG из ffmpeg: opencv читает как BGRA — порядок правильный
            frames.append(arr)

    return frames, target_fps


# Оставляем для совместимости (не используются в основном пайплайне)
def read_frames_opencv(path):
    return read_frames_ffmpeg(path)

def read_frames_pillow(path):
    return read_frames_ffmpeg(path)


def split_frames_to_cells(frames, cols, rows):
    src_h = rows * STICKER_SIZE
    cells = [[] for _ in range(cols * rows)]
    cell_h = src_h // rows
    for frame in frames:
        if cols == WIDE_COLS:
            # Масштабируем на 1 колонку шире, затем центрально кропаем
            scale_cols = WIDE_COLS + 1
            scale_w = scale_cols * STICKER_SIZE
            big = cv2.resize(frame, (scale_w, src_h))
            offset = ((scale_cols - cols) * STICKER_SIZE) // 2
            r = big[:, offset: offset + cols * STICKER_SIZE]
        else:
            src_w = cols * STICKER_SIZE
            r = cv2.resize(frame, (src_w, src_h))
        cell_w = STICKER_SIZE
        for row in range(rows):
            for col in range(cols):
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w
                cell = cv2.resize(r[y0:y1, x0:x1], (STICKER_SIZE, STICKER_SIZE))
                cells[row * cols + col].append(cell)
    return cells


def cell_frames_to_webm(cell_frames, out_path, fps):
    tmp = out_path.parent / f"_t_{out_path.stem}"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(cell_frames):
        # Конвертируем в RGBA и сохраняем через PIL — корректный порядок каналов для ffmpeg
        if f.shape[2] == 3:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2BGRA)
        rgba = cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA)
        Image.fromarray(rgba, "RGBA").save(str(tmp / f"f_{i:04d}.png"))
    out_fps = min(fps, 30)
    subprocess.run([
        FFMPEG, "-y",
        "-framerate", str(out_fps),
        "-i", str(tmp / "f_%04d.png"),
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0",
        "-b:v", "0", "-crf", "35",
        "-t", str(MAX_VIDEO_SEC),
        "-s", f"{STICKER_SIZE}x{STICKER_SIZE}",
        str(out_path),
    ], check=True, capture_output=True, timeout=60)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_path


def video_to_sticker_webm(src: str, out_path: Path) -> Path:
    """Конвертирует видео/GIF/MP4 в 512×512 WebM VP9 для анимированного стикера.
    Обрезает до MAX_VIDEO_SEC, сохраняет альфа-канал если есть."""
    subprocess.run([
        FFMPEG, "-y", "-i", str(src),
        "-t", str(MAX_VIDEO_SEC),
        "-vf", f"fps=30,scale=512:512:force_original_aspect_ratio=decrease,"
               f"pad=512:512:(ow-iw)/2:(oh-ih)/2:color=0x00000000",
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0",
        "-b:v", "0", "-crf", "30",
        str(out_path),
    ], check=True, capture_output=True, timeout=120)
    return out_path


async def build_emoji_pack(ctx, user_id, paths, fmt, title, progress_chat_id=None, progress_msg_id=None):
    bot = ctx.bot
    me = await bot.get_me()
    pack_name = f"m{str(int(time.time()))[-6:]}_{user_id}_by_{me.username}"
    total = len(paths)

    def read_sticker(p):
        with open(p, "rb") as fh:
            return fh.read()

    await _tg_retry(bot.create_new_sticker_set,
        user_id=user_id,
        name=pack_name,
        title=_make_title(title),
        stickers=[InputSticker(sticker=io.BytesIO(read_sticker(paths[0])), emoji_list=["🟦"], format=fmt)],
        sticker_type=StickerType.CUSTOM_EMOJI,
    )

    for i, p in enumerate(paths[1:], start=2):
        await _tg_retry(bot.add_sticker_to_set,
            user_id=user_id,
            name=pack_name,
            sticker=InputSticker(sticker=io.BytesIO(read_sticker(p)), emoji_list=["🟦"], format=fmt),
        )
        if progress_chat_id and progress_msg_id and i % 10 == 0:
            pct = int((i / total) * 100)
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
            try:
                await bot.edit_message_text(
                    f"... Загружаю эмодзи\n{bar} {pct}% ({i}/{total})",
                    chat_id=progress_chat_id,
                    message_id=progress_msg_id,
                )
            except Exception:
                pass

    return pack_name


async def send_emoji_preview(bot, chat_id: int, pack_name: str, cols: int, rows: int):
    """Отправляет сетку кастомных эмодзи из пака — готово к копированию."""
    try:
        sticker_set = await bot.get_sticker_set(pack_name)
        stickers    = sticker_set.stickers

        if len(stickers) < cols * rows:
            return  # пак ещё не полный — пропускаем

        # ⬜ (U+2B1C) — BMP-символ = 1 UTF-16 unit, отображается до загрузки эмодзи
        placeholder = "⬜"
        entities    = []
        offset      = 0

        for r in range(rows):
            for c in range(cols):
                sticker = stickers[r * cols + c]
                emoji_id = sticker.custom_emoji_id or sticker.file_unique_id
                entities.append(MessageEntity(
                    type=MessageEntity.CUSTOM_EMOJI,
                    offset=offset,
                    length=1,
                    custom_emoji_id=emoji_id,
                ))
                offset += 1          # 1 символ ⬜
            offset += 1              # символ \n

        text = "\n".join(placeholder * cols for _ in range(rows))

        await bot.send_message(
            chat_id=chat_id,
            text=text,
            entities=entities,
        )
        await bot.send_message(
            chat_id=chat_id,
            text="☝️ Готовая сетка — копируй и вставляй в пост.\n"
                 "_Если часть эмодзи отображается квадратиками — подожди несколько секунд, "
                 "Telegram догружает пак._",
            parse_mode="Markdown",
        )
    except Exception as e:
        log.warning(f"send_emoji_preview: {e}")


# ════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ: GIF
# ════════════════════════════════════════════════════════════════════════

def get_video_duration(path: str) -> float | None:
    try:
        result = subprocess.run(
            [
                FFPROBE, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def convert_to_gif(input_path: str, output_path: str, width: int, fps: int) -> tuple[bool, str]:
    palette_path = str(Path(output_path).parent / "palette.png")
    # scale: точный размер + lanczos; force_original_aspect_ratio сохраняет пропорции
    scale_filter = (
        f"fps={fps},"
        f"scale={width}:-1:flags=lanczos+accurate_rnd"
    )

    # Проход 1: палитра по diff-кадрам (меньше "мусорных" цветов → лучше сжатие)
    pass1 = subprocess.run(
        [
            FFMPEG, "-y", "-i", input_path,
            "-vf", f"{scale_filter},palettegen=max_colors=256:stats_mode=diff:reserve_transparent=1",
            palette_path,
        ],
        capture_output=True, text=True, timeout=120,
    )
    if pass1.returncode != 0:
        return False, pass1.stderr

    # Проход 2: кодирование
    #   bayer_scale=3   — меньше шума по сравнению с 5, LZW лучше сжимает
    #   diff_mode=rectangle — обновляет только изменившийся прямоугольник кадра
    #   +transdiff      — неизменившиеся пиксели → прозрачные: ключевой флаг
    #                     именно это делает онлайн-конвертеры "легче"
    #   -loop 0         — бесконечный цикл
    paletteuse = (
        "paletteuse=dither=bayer:bayer_scale=3"
        ":diff_mode=rectangle"
    )
    pass2 = subprocess.run(
        [
            FFMPEG, "-y",
            "-i", input_path,
            "-i", palette_path,
            "-lavfi", f"{scale_filter} [x]; [x][1:v] {paletteuse}",
            "-gifflags", "+transdiff",
            "-loop", "0",
            "-t", str(GIF_MAX_DURATION),
            output_path,
        ],
        capture_output=True, text=True, timeout=180,
    )
    if pass2.returncode != 0:
        return False, pass2.stderr

    return True, ""


def convert_to_gif_adaptive(
    input_path: str, output_path: str,
    width: int, fps: int, max_mb: int,
    status_callback=None,          # async callable(text) для обновления статуса
) -> tuple[bool, str, int, int]:
    """
    Конвертирует GIF с автоматическим уменьшением качества,
    если файл превышает max_mb. Делает до 4 попыток.
    Возвращает (ok, err, итоговый_width, итоговый_fps).
    """
    cur_w, cur_fps = width, fps
    for attempt in range(4):
        ok, err = convert_to_gif(input_path, output_path, cur_w, cur_fps)
        if not ok:
            return False, err, cur_w, cur_fps

        size_mb = Path(output_path).stat().st_size / 1_048_576
        if size_mb <= max_mb:
            break   # вписались в лимит

        if attempt < 3:
            # Уменьшаем разрешение на 20% и fps на 15% для следующей попытки
            cur_w   = max(160, int(cur_w   * 0.80))
            cur_fps = max(6,   int(cur_fps * 0.85))
            log.info(f"GIF {size_mb:.1f} МБ > {max_mb} МБ, пробую {cur_w}p/{cur_fps}fps")

    return True, "", cur_w, cur_fps


def format_file_size(path: str) -> str:
    size = Path(path).stat().st_size
    if size >= 1_048_576:
        return f"{size / 1_048_576:.1f} МБ"
    return f"{size / 1024:.0f} КБ"


# ════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ: удаление фона
# ════════════════════════════════════════════════════════════════════════

def remove_solid_bg(img: Image.Image, mode: str, tolerance: int = 35) -> Image.Image:
    """Убирает однотонный белый или чёрный фон по порогу."""
    img  = img.convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    if mode == "white":
        mask = (r >= 255 - tolerance) & (g >= 255 - tolerance) & (b >= 255 - tolerance)
    else:  # black
        mask = (r <= tolerance) & (g <= tolerance) & (b <= tolerance)
    data[mask, 3] = 0
    return Image.fromarray(data, "RGBA")



def bg_remove_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("□  Убрать белый фон", callback_data="bgr_white")],
        [InlineKeyboardButton("■  Убрать чёрный фон", callback_data="bgr_black")],
        [InlineKeyboardButton("→  Пропустить",        callback_data="bgr_skip")],
    ])


# ════════════════════════════════════════════════════════════════════════
#  МЕНЮ
# ════════════════════════════════════════════════════════════════════════

WELCOME_TEXT = (
    "Вот что я умею:\n\n"
    "▪ *Стикерпак* — создаю стикерпак из твоих фото. Обычный (до 50 стикеров) "
    "или вертикальный (1 фото → 2 стикера 512×512).\n\n"
    "◼ *Эмодзи-мозаика* (доступно TG Premium) — разрезаю фото или анимацию на кастомные эмодзи.\n\n"
    "Ａ *Текст → эмодзи* (доступно TG Premium) — превращаю текст в кастомные эмодзи с выбором шрифта, "
    "цвета, высоты и анимации.\n\n"
    "◉ *Кружок из видео* — конвертирую любое видео в круглое видео-сообщение "
    "(до 60 секунд).\n\n"
    "▶ *GIF из видео* — конвертирую видео в высококачественный GIF "
    "с выбором качества.\n\n"
    "◻ *Коллаж из фото* — собираю 2–9 фото в один коллаж на выбор из 10 шаблонов.\n\n"
    "Выбери, что хочешь сделать:"
)

def main_menu_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▪  Стикерпак",                      callback_data="go_stickers")],
        [InlineKeyboardButton("◼  Эмодзи-мозаика  (TG Premium)",   callback_data="go_emoji")],
        [InlineKeyboardButton("◉  Кружок из видео",                callback_data="go_circle")],
        [InlineKeyboardButton("▶  GIF из видео",                   callback_data="go_gif")],
        [InlineKeyboardButton("Ａ  Текст → эмодзи  (TG Premium)",   callback_data="go_txt")],
        [InlineKeyboardButton("◻  Коллаж из фото",                 callback_data="go_collage")],
        [InlineKeyboardButton("—  Инструкция",                     callback_data="help")],
        [InlineKeyboardButton("—  Инструкция (Photoshop или AE)",  callback_data="help_ps")],
        [InlineKeyboardButton("💛  Donate",                         callback_data="donate")],
    ])

def back_kb():
    return InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]])


async def safe_edit(query, text: str, reply_markup=None, parse_mode="Markdown"):
    """
    Пытается edit_message_text. Если сообщение содержит медиа (анимация/фото) —
    удаляет его и отправляет новое текстовое сообщение.
    """
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except Exception:
        try:
            await query.message.delete()
        except Exception:
            pass
        await query.message.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)

def emoji_mode_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("◼  Фото (PNG / JPG / WebP)",       callback_data="mode_image")],
        [InlineKeyboardButton("▶  Анимация (WebM / MOV / MP4)",    callback_data="mode_video")],
        [InlineKeyboardButton("🎞  GIF → анимированные эмодзи",   callback_data="mode_gif")],
        [InlineKeyboardButton("← Главное меню",                   callback_data="back_menu")],
    ])

async def _send_main_menu(message):
    """Отправляет главное меню с гиф (если есть) новым сообщением."""
    kb = main_menu_kb()
    gif_path = Path(MENU_GIF)
    if gif_path.exists():
        with open(gif_path, "rb") as f:
            await message.reply_animation(
                animation=f,
                caption=WELCOME_TEXT,
                reply_markup=kb,
                parse_mode="Markdown",
            )
    else:
        await message.reply_text(WELCOME_TEXT, reply_markup=kb, parse_mode="Markdown")


async def show_main_menu(update: Update):
    kb = main_menu_kb()
    if update.callback_query:
        try:
            # Пробуем удалить старое сообщение и отправить новое с гиф
            await update.callback_query.message.delete()
        except Exception:
            pass
        await _send_main_menu(update.callback_query.message)
    else:
        await _send_main_menu(update.message)


# ════════════════════════════════════════════════════════════════════════
#  ОБЩИЕ ХЭНДЛЕРЫ
# ════════════════════════════════════════════════════════════════════════

async def check_subscription(bot, user_id: int) -> bool:
    """Возвращает True если пользователь подписан на REQUIRED_CHANNEL."""
    try:
        member = await bot.get_chat_member(REQUIRED_CHANNEL, user_id)
        return member.status in ("member", "administrator", "creator")
    except Exception:
        # Если бот не может проверить (нет прав) — пропускаем
        return True


def _sub_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📢 Подписаться", url=f"https://t.me/{REQUIRED_CHANNEL.lstrip('@')}")],
        [InlineKeyboardButton("✅ Я подписался — проверить", callback_data="check_sub")],
    ])


async def _send_sub_screen(target, is_callback: bool = False):
    """Отправляет экран с просьбой подписаться (с гиф если есть)."""
    text = (
        "👋 Привет!\n\n"
        "Чтобы пользоваться ботом, подпишись на канал:\n"
        f"{REQUIRED_CHANNEL}\n\n"
        "После подписки нажми кнопку ниже 👇"
    )
    kb = _sub_keyboard()
    gif_path = Path(WELCOME_GIF2)

    if is_callback:
        # После нажатия кнопки — просто обновляем текст (гиф уже есть)
        try:
            await target.edit_message_caption(caption=text, reply_markup=kb)
        except Exception:
            try:
                await target.edit_message_text(text=text, reply_markup=kb)
            except Exception:
                pass
    else:
        if gif_path.exists():
            with open(gif_path, "rb") as f:
                await target.reply_animation(animation=f, caption=text, reply_markup=kb)
        else:
            await target.reply_text(text, reply_markup=kb)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data.clear()
    log_action(update.effective_user, "открыл бота /start")

    if not await check_subscription(ctx.bot, update.effective_user.id):
        await _send_sub_screen(update.message, is_callback=False)
        return MAIN_MENU  # остаёмся в MAIN_MENU чтобы кнопка check_sub работала

    welcome_caption = (
        "Привет, в этом боте сможешь разнообразить посты для своего канала.\n\n"
        "by MYO | d1fficul7"
    )

    start_kb = InlineKeyboardMarkup([[InlineKeyboardButton("▶  Начать", callback_data="go_start")]])

    gif_path = Path(WELCOME_GIF)
    if gif_path.exists():
        with open(gif_path, "rb") as f:
            await update.message.reply_animation(animation=f, caption=welcome_caption, reply_markup=start_kb)
    else:
        await update.message.reply_text(welcome_caption, reply_markup=start_kb)

    return MAIN_MENU


async def btn_go_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    try:
        await update.callback_query.message.delete()
    except Exception:
        pass
    await _send_main_menu(update.callback_query.message)
    return MAIN_MENU


async def btn_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await safe_edit(q,
        "◼ *Эмодзи-мозаика:*\n"
        "1. Нажми «Эмодзи-мозаика» и выбери тип файла\n"
        "2. Отправь файл как документ (скрепка)\n"
        "3. Выбери размер сетки\n"
        "4. Введи название эмодзи-пака\n"
        "5. Получи ссылку, добавь пак и вставь шаблон в чат\n\n"
        "! PNG/WebP — с прозрачным фоном\n"
        "! JPG — фон будет белым\n\n"
        "○ *Кружок из видео:*\n"
        "1. Нажми «Кружок из видео»\n"
        "2. Отправь видео (до 60 сек)\n"
        "3. Получи круглое видео-сообщение\n\n"
        "▶ *GIF из видео:*\n"
        "1. Нажми «GIF из видео»\n"
        "2. Отправь видео (до 30 сек) — MP4, MOV, AVI, MKV, WebM и др.\n"
        "3. Выбери качество: лёгкая / стандарт / высокое\n"
        "4. Получи GIF-файл\n\n"
        "◻ *Коллаж из фото:*\n"
        "1. Нажми «Коллаж из фото»\n"
        "2. Отправляй фото по одной (2–9 штук)\n"
        "3. Нажми «Готово» и выбери шаблон\n"
        "4. Выбери режим вписывания, отступы и фон\n"
        "5. Получи готовый коллаж 1080px\n\n"
        "▪ *Стикерпак:*\n"
        "1. Нажми «Стикерпак» и выбери режим\n"
        "2. Обычный: отправляй фото по одному, нажми «Готово»\n"
        "3. Вертикальный: отправь одно вертикальное фото — разрежу на 2 стикера\n"
        "4. Введи название → получи ссылку на стикерпак\n\n"
        "Если что-то пошло не так — напиши /start",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Назад", callback_data="back_menu")]]),
    )
    return MAIN_MENU


async def btn_check_sub(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    user_id = q.from_user.id

    if await check_subscription(ctx.bot, user_id):
        # Подписан — удаляем экран подписки и показываем меню
        try:
            await q.message.delete()
        except Exception:
            pass
        await _send_main_menu(q.message)
    else:
        await q.answer("❌ Подписка не найдена. Подпишись и попробуй снова.", show_alert=True)

    return MAIN_MENU


async def btn_help_ps(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await safe_edit(q,
        "🎨 *Photoshop и After Effects*\n\n"
        "С помощью Photoshop и After Effects ты можешь не только убрать задний фон, "
        "но и креативно реализовывать свои идеи — создавать уникальные изображения, "
        "вырезать объекты, делать анимации с эффектами, и всё это с прозрачным фоном.\n\n"
        "Бот сохранит прозрачность в итоговых эмодзи и стикерах — так твои материалы "
        "будут смотреться чисто и красиво в Telegram.\n\n"
        "Как найти инструкции — ищи в интернете или пиши мне: @d1fficul7",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Назад", callback_data="back_menu")]]),
    )
    return MAIN_MENU


async def btn_donate(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_caption(
        caption=(
            "Если бот был вам полезен и вы хотите меня поддержать:\n\n"
            "> `2202205365861688`"
        ),
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Назад", callback_data="back_menu")]]),
        parse_mode="Markdown",
    )
    return MAIN_MENU


async def btn_back(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    ctx.user_data.clear()
    await show_main_menu(update)
    return MAIN_MENU


async def fallback_start_hint(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Что-то пошло не так. Напиши /start чтобы начать заново."
    )
    return MAIN_MENU


# ════════════════════════════════════════════════════════════════════════
#  ХЭНДЛЕРЫ: кружок
# ════════════════════════════════════════════════════════════════════════

async def btn_go_circle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    log_action(q.from_user, "выбрал режим: кружок")
    await safe_edit(q,
        "○ *Кружок из видео*\n\n"
        "Отправь видео — я обрежу его до квадрата и верну круглым видео-сообщением.\n"
        "Ограничение Telegram: максимум 60 секунд.",
        reply_markup=back_kb(),
    )
    return CIRCLE_WAIT_VIDEO


async def handle_circle_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    message = update.message
    video = message.video or message.document

    if message.document and not (message.document.mime_type or "").startswith("video/"):
        await message.reply_text(
            "❌ Это не видеофайл. Отправь видео!\n\n"
            "Или вернись в главное меню через /start",
            reply_markup=back_kb(),
        )
        return CIRCLE_WAIT_VIDEO

    if not video:
        await message.reply_text(
            "Отправь видео. Если потерялся — напиши /start",
            reply_markup=back_kb(),
        )
        return CIRCLE_WAIT_VIDEO

    status = await message.reply_text("... Обрабатываю видео")
    log_action(update.effective_user, "отправил видео для кружка")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.mp4"
        output_file = Path(tmpdir) / "circle.mp4"

        tg_file = await ctx.bot.get_file(video.file_id)
        await tg_file.download_to_drive(str(input_file))

        if not convert_to_circle(str(input_file), str(output_file)):
            await status.edit_text(
                "❌ Ошибка при конвертации. Попробуй другое видео.\n\n"
                "Напиши /start чтобы начать заново.",
                reply_markup=back_kb(),
            )
            return CIRCLE_WAIT_VIDEO

        await status.edit_text("... Отправляю кружок")
        with open(output_file, "rb") as f:
            await message.reply_video_note(f)

        await status.delete()
        log_action(update.effective_user, "✅ кружок отправлен")

    await message.reply_text(
        "Готово. Отправь ещё видео или вернись в главное меню.\n\n"
        "В случае неполадок запустите бота заново — /start",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("← Главное меню", callback_data="back_menu")]
        ]),
    )
    return CIRCLE_WAIT_VIDEO


# ════════════════════════════════════════════════════════════════════════
#  ХЭНДЛЕРЫ: эмодзи-мозаика
# ════════════════════════════════════════════════════════════════════════

async def btn_go_emoji(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    log_action(q.from_user, "выбрал режим: эмодзи-мозаика")
    await safe_edit(q,
        "◼ *Эмодзи-мозаика*\n\n"
        "Если у твоего файла прозрачный фон — он сохранится прозрачным в Telegram.\n"
        "_(прозрачный фон можно сделать в Photoshop)_\n\n"
        "Выбери тип файла:",
        reply_markup=emoji_mode_kb(),
    )
    return EMOJI_WAIT_FILE


async def btn_emoji_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    mode = q.data
    ctx.user_data["mode"] = mode
    log_action(q.from_user, f"выбрал подрежим: {mode}")
    if mode == "mode_image":
        text = "◼ Отправь PNG, JPG или статичный WebP как документ (скрепка):\n! JPG — фон будет белым"
    elif mode == "mode_gif":
        text = "🎞 Отправь GIF как документ (скрепка 📎):\n\nБот нарежет каждый кадр на ячейки и создаст анимированные эмодзи."
    else:
        text = "▶ Отправь WebM, MOV или анимированный WebP как документ (скрепка):"
    await q.edit_message_text(text, reply_markup=back_kb())
    return EMOJI_WAIT_FILE


async def handle_emoji_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message

    if msg.document:
        fobj  = msg.document
        mime  = fobj.mime_type or ""
        fname = fobj.file_name or "file"
    elif msg.sticker:
        fobj  = msg.sticker
        mime  = "image/webp"
        fname = "sticker.webp"
    elif msg.animation:
        fobj  = msg.animation
        mime  = fobj.mime_type or "image/webp"
        fname = fobj.file_name or "anim.webp"
    elif msg.video:
        fobj  = msg.video
        mime  = fobj.mime_type or "video/webm"
        fname = fobj.file_name or "video.webm"
    elif msg.photo:
        fobj  = msg.photo[-1]
        mime  = "image/jpeg"
        fname = "photo.jpg"
    else:
        await msg.reply_text(
            "Отправь файл как документ (скрепка 📎).\n\nЕсли потерялся — напиши /start",
            reply_markup=back_kb(),
        )
        return EMOJI_WAIT_FILE

    fl = fname.lower()
    is_jpg   = mime in ("image/jpeg", "image/jpg") or fl.endswith((".jpg", ".jpeg"))
    is_png   = mime == "image/png" or fl.endswith(".png")
    is_webp  = mime == "image/webp" or fl.endswith(".webp")
    is_gif   = mime == "image/gif" or fl.endswith(".gif")
    # Telegram конвертирует GIF → mp4; принимаем video/mp4 в режиме GIF
    _sel_mode = ctx.user_data.get("mode", "")
    if mime == "video/mp4" and _sel_mode == "mode_gif":
        is_gif = True
    is_video = mime in ("video/webm", "video/quicktime", "video/mp4") or fl.endswith((".webm", ".mov", ".mp4"))

    if not (is_jpg or is_png or is_webp or is_video or is_gif):
        await msg.reply_text(
            f"❌ Формат не поддерживается ({mime}).\nНужны: PNG, JPG, GIF, WebM, MOV или WebP.\n\n"
            "Напиши /start чтобы начать заново.",
            reply_markup=back_kb(),
        )
        return EMOJI_WAIT_FILE

    if "mode" not in ctx.user_data:
        if is_gif:
            ctx.user_data["mode"] = "mode_gif"
        elif is_video:
            ctx.user_data["mode"] = "mode_video"
        else:
            ctx.user_data["mode"] = "mode_image"

    ctx.user_data["file_id"]   = fobj.file_id
    ctx.user_data["file_name"] = fname
    ctx.user_data["is_jpg"]    = is_jpg
    ctx.user_data["is_webp"]   = is_webp
    ctx.user_data["is_gif"]    = is_gif
    ctx.user_data["is_video"]  = is_video
    log_action(update.effective_user, f"загрузил файл: {fname}")

    img_w, img_h = None, None
    aspect_note = ""

    if not is_video:
        try:
            tmp_dir = Path(tempfile.mkdtemp())
            ctx.user_data["tmp_dir"] = str(tmp_dir)
            src = tmp_dir / fname
            tg_file = await ctx.bot.get_file(fobj.file_id)
            await tg_file.download_to_drive(src)
            ctx.user_data["cached_src"] = str(src)

            with Image.open(src) as probe:
                img_w, img_h = probe.size
            ctx.user_data["img_w"] = img_w
            ctx.user_data["img_h"] = img_h
            ratio = img_w / img_h
            orient = "горизонтальное" if ratio > 1.1 else ("вертикальное" if ratio < 0.9 else "квадратное")
            aspect_note = f"Фото {img_w}×{img_h} ({orient}). Сетки отсортированы по соотношению сторон.\n\n"
        except Exception as e:
            log.warning(f"Не удалось определить размер: {e}")

    # Для видео/GIF предупреждаем об обрезке если длиннее 3 сек
    if is_video or is_gif:
        try:
            tg_file = await ctx.bot.get_file(fobj.file_id)
            tmp_dir = Path(tempfile.mkdtemp())
            ctx.user_data["tmp_dir"] = str(tmp_dir)
            src = tmp_dir / fname
            await tg_file.download_to_drive(src)
            ctx.user_data["cached_src"] = str(src)
            r = subprocess.run(
                [FFPROBE, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
                capture_output=True, text=True, timeout=15,
            )
            dur = float(r.stdout.strip())
            if dur > MAX_VIDEO_SEC:
                await msg.reply_text(
                    f"⚠️ Видео длиной {dur:.1f} с — Telegram разрешает эмодзи до {MAX_VIDEO_SEC:.0f} с.\n"
                    f"Бот возьмёт первые {MAX_VIDEO_SEC:.0f} секунды.",
                )
        except Exception:
            pass

    # Спрашиваем форм-фактор эмодзи перед сеткой
    return await _show_emoji_aspect(msg, ctx)


# Соотношения сторон ячейки эмодзи для разных платформ
EMOJI_ASPECTS = {
    "universal": (1.0,  "◉ Универсальный — 1:1 (рекомендуется)"),
    "android":   (1.0,  "▪ Android — 1:1"),
    "ios":       (1.05, "○ iOS — чуть выше 1:1"),
    "desktop":   (0.9,  "◻ Компьютер — чуть шире"),
}

async def _show_emoji_aspect(message, ctx: ContextTypes.DEFAULT_TYPE):
    """Шаг 1 после загрузки файла: выбор форм-фактора эмодзи."""
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(label, callback_data=f"aspect_{key}")]
        for key, (_, label) in EMOJI_ASPECTS.items()
    ] + [[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]])
    await message.reply_text(
        "Выбери под какую платформу делать эмодзи:\n\n"
        "Это влияет на то, как будут выглядеть ячейки у получателя.",
        reply_markup=kb,
    )
    return EMOJI_WAIT_ASPECT


async def btn_emoji_aspect(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    key = q.data.replace("aspect_", "")
    ctx.user_data["emoji_aspect"] = key
    return await _show_emoji_grid(q.message, ctx)


async def _show_emoji_grid(msg_or_query, ctx: ContextTypes.DEFAULT_TYPE):
    """Показывает клавиатуру выбора сетки. Принимает message или CallbackQuery.message."""
    img_w = ctx.user_data.get("img_w")
    img_h = ctx.user_data.get("img_h")

    if img_w and img_h:
        sorted_grids = sort_grids_by_aspect(img_w, img_h)
        ratio  = img_w / img_h
        orient = "горизонтальное" if ratio > 1.1 else ("вертикальное" if ratio < 0.9 else "квадратное")
        aspect_note = f"📐 Фото {img_w}×{img_h} ({orient}). Сетки отсортированы по соотношению сторон.\n\n"
    else:
        sorted_grids = [(c, r, 0.0) for c, r in GRID_OPTIONS if c * r <= MAX_STICKERS]
        aspect_note  = ""

    grid_btns = [[InlineKeyboardButton("✏️  Своя сетка (ввести вручную)", callback_data="grid_custom")]]
    for cols, rows, score in sorted_grids:
        total    = cols * rows
        fit_mark = "◼ " if score < 0.05 else ("▪ " if score < 0.15 else "  ")
        wide_mark = "▶ " if cols == WIDE_COLS else "  "
        if cols == WIDE_COLS and rows <= 2:
            suffix = " — текст/надпись"
        elif cols == WIDE_COLS:
            suffix = " — вся ширина"
        else:
            suffix = ""
        label = f"{fit_mark}{wide_mark}{cols}×{rows} ({total} эмодзи){suffix}"
        grid_btns.append([InlineKeyboardButton(label, callback_data=f"grid_{cols}_{rows}")])
    grid_btns.append([InlineKeyboardButton("← Главное меню", callback_data="back_menu")])

    text = (
        f"Выбери размер сетки:\n{aspect_note}"
        "◼ — отлично подходит\n"
        "▪ — хорошо подходит\n"
        "▶ — вся ширина экрана"
    )
    await msg_or_query.reply_text(text, reply_markup=InlineKeyboardMarkup(grid_btns))
    return EMOJI_WAIT_GRID


async def btn_emoji_bg_remove(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    mode = q.data.replace("bgr_", "")  # white / black / ai / skip
    cached_src = ctx.user_data.get("cached_src")

    if mode != "skip" and cached_src:
        status = await q.edit_message_text("... Убираю фон")
        try:
            img = Image.open(cached_src)
            result = remove_solid_bg(img, mode)

            # Сохраняем обратно как PNG с прозрачностью
            new_path = str(Path(cached_src).with_suffix(".png"))
            result.save(new_path, "PNG")
            ctx.user_data["cached_src"] = new_path
            ctx.user_data["is_jpg"]     = False  # теперь PNG с прозрачностью

            # Обновляем размеры
            ctx.user_data["img_w"], ctx.user_data["img_h"] = result.size
            await status.edit_text("Фон убран.")
        except RuntimeError as e:
            await status.edit_text(f"❌ {e}")
            return EMOJI_WAIT_BG_REMOVE
        except Exception as e:
            log.exception("Ошибка удаления фона")
            await status.edit_text(f"❌ Ошибка: {e}\n\nПродолжаем без изменений.")
    else:
        await q.edit_message_text("→ Фон оставлен как есть.")

    return await _show_emoji_grid(q.message, ctx)


async def btn_emoji_grid_custom(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Пользователь нажал «Своя сетка» — просим ввести размер текстом."""
    q = update.callback_query
    await q.answer()
    await q.message.reply_text(
        "Введи размер сетки в формате *КОЛ×СТРОК* (или через пробел/x):\n\n"
        "Например: `4×5` или `6 3` или `8x4`\n\n"
        f"Ограничение: максимум {MAX_COLS} колонок и {MAX_STICKERS} эмодзи всего.",
        reply_markup=back_kb(),
        parse_mode="Markdown",
    )
    return EMOJI_WAIT_CUSTOM_GRID


async def handle_emoji_custom_grid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает ввод своей сетки типа '4×5' или '6 3'."""
    import re
    text = update.message.text.strip()
    m = re.match(r"(\d+)\s*[×x\s,]\s*(\d+)", text, re.IGNORECASE)
    if not m:
        await update.message.reply_text(
            "❌ Не понял формат. Введи как `4×5` или `6 3`.",
            reply_markup=back_kb(), parse_mode="Markdown",
        )
        return EMOJI_WAIT_CUSTOM_GRID

    cols, rows = int(m.group(1)), int(m.group(2))
    total = cols * rows

    if cols < 1 or rows < 1:
        await update.message.reply_text("❌ Размер должен быть не менее 1×1.", reply_markup=back_kb())
        return EMOJI_WAIT_CUSTOM_GRID
    if cols > MAX_COLS:
        await update.message.reply_text(f"❌ Максимум {MAX_COLS} колонок.", reply_markup=back_kb())
        return EMOJI_WAIT_CUSTOM_GRID
    if total > MAX_STICKERS:
        await update.message.reply_text(f"❌ Максимум {MAX_STICKERS} эмодзи ({cols}×{rows}={total}).", reply_markup=back_kb())
        return EMOJI_WAIT_CUSTOM_GRID

    ctx.user_data["cols"] = cols
    ctx.user_data["rows"] = rows
    log_action(update.effective_user, f"выбрал свою сетку {cols}×{rows}")
    await update.message.reply_text(
        f"Сетка {cols}×{rows} ({total} эмодзи). Введи название пака:",
        reply_markup=back_kb(),
    )
    return EMOJI_WAIT_PACK_NAME


async def btn_emoji_grid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    parts = q.data.split("_")
    cols, rows = int(parts[1]), int(parts[2])
    ctx.user_data["cols"] = cols
    ctx.user_data["rows"] = rows
    total = cols * rows
    log_action(q.from_user, f"выбрал сетку {cols}×{rows}")
    note = "— вся ширина экрана" if cols == WIDE_COLS else ""
    await q.edit_message_text(
        f"Сетка {cols}×{rows} ({total} эмодзи). {note}\n\nВведи название для эмодзи-пака:",
        reply_markup=back_kb(),
    )
    return EMOJI_WAIT_PACK_NAME


async def handle_emoji_pack_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lock    = get_user_lock(user_id)

    if lock.locked():
        await update.message.reply_text("... Подожди — предыдущий пак ещё создаётся.")
        return EMOJI_WAIT_PACK_NAME

    async with lock:
        title    = update.message.text.strip()[:64]
        cols     = ctx.user_data["cols"]
        rows     = ctx.user_data["rows"]
        total    = cols * rows
        file_id  = ctx.user_data["file_id"]
        fname    = ctx.user_data["file_name"]
        mode     = ctx.user_data.get("mode", "mode_image")
        is_jpg   = ctx.user_data.get("is_jpg",   False)
        is_webp  = ctx.user_data.get("is_webp",  False)
        is_gif   = ctx.user_data.get("is_gif",   False)
        is_video = ctx.user_data.get("is_video", False)
        cached_src = ctx.user_data.get("cached_src")

        progress_msg = await update.message.reply_text(
            f"... Создаю эмодзи-пак «{title}» ({cols}×{rows})\n"
            "░░░░░░░░░░ 0%\n"
            "Это может занять от 30 секунд до нескольких минут."
        )

        existing_tmp = ctx.user_data.get("tmp_dir")
        tmp_dir = Path(existing_tmp) if existing_tmp else Path(tempfile.mkdtemp())

        try:
            if cached_src:
                src = Path(cached_src)
            else:
                tg_file = await ctx.bot.get_file(file_id)
                src = tmp_dir / fname
                await tg_file.download_to_drive(src)

            cells_dir = tmp_dir / "cells"
            cells_dir.mkdir(exist_ok=True)
            sticker_paths = []

            if mode in ("mode_video", "mode_gif") or is_video or is_gif:
                fmt = StickerFormat.VIDEO
                frames, fps = read_frames_ffmpeg(src)
                cell_list = split_frames_to_cells(frames, cols, rows)
                for i, cf in enumerate(cell_list):
                    out = cells_dir / f"cell_{i:04d}.webm"
                    cell_frames_to_webm(cf, out, fps)
                sticker_paths = sorted(cells_dir.glob("*.webm"))
            else:
                fmt = StickerFormat.STATIC
                raw = Image.open(src)
                if is_jpg:
                    bg = Image.new("RGBA", raw.size, (255, 255, 255, 255))
                    bg.paste(raw.convert("RGB"))
                    img = bg
                else:
                    img = raw.convert("RGBA")
                cells = slice_image(img, cols, rows)
                sticker_paths = save_cells_png(cells, cells_dir)

            pack_name = await build_emoji_pack(
                ctx, user_id, sticker_paths, fmt, title,
                progress_chat_id=progress_msg.chat_id,
                progress_msg_id=progress_msg.message_id,
            )
            log_action(update.effective_user, f"✅ создал пак «{title}» {cols}×{rows} → {pack_name}")

            pack_url = f"https://t.me/addemoji/{pack_name}"

            await progress_msg.edit_text(
                f"✅ Эмодзи-пак «{title}» создан ({cols}×{rows}, {total} эмодзи).\n\n"
                f"→ {pack_url}\n\n"
                f"Добавь пак по ссылке — и ниже появится готовая сетка.\n"
                f"Можешь скопировать её и вставить в пост как есть, "
                f"или расставить эмодзи вручную.\n\n"
                f"В случае неполадок — /start",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("↺  Создать ещё", callback_data="back_menu")]
                ]),
            )
            await asyncio.sleep(5)   # ждём пока Telegram закэширует весь пак
            await send_emoji_preview(
                ctx.bot, progress_msg.chat_id, pack_name, cols, rows
            )

        except Exception as e:
            log.exception("Ошибка при создании пака")
            await progress_msg.edit_text(
                f"❌ Что-то пошло не так: {e}\n\nНапиши /start чтобы начать заново.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    ctx.user_data.clear()
    return MAIN_MENU


# ════════════════════════════════════════════════════════════════════════
#  ХЭНДЛЕРЫ: GIF
# ════════════════════════════════════════════════════════════════════════

def gif_quality_kb():
    rows = []
    for key, (w, fps, label, _max_mb) in GIF_QUALITY_PRESETS.items():
        rows.append([InlineKeyboardButton(
            f"{label}  {w}p · {fps} кадр/с",
            callback_data=f"gifq_{key}",
        )])
    rows.append([InlineKeyboardButton("← Главное меню", callback_data="back_menu")])
    return InlineKeyboardMarkup(rows)


async def btn_go_gif(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    log_action(q.from_user, "выбрал режим: GIF")
    await safe_edit(q,
        "▶ *GIF из видео*\n\n"
        "Отправь видео — я конвертирую его в высококачественный GIF.\n\n"
        "📁 Форматы: MP4, MOV, AVI, MKV, WebM, FLV, WMV, 3GP и другие\n"
        "⏱ Максимум: 30 секунд\n\n"
        "Отправляй как документ (📎 скрепка) или обычным видео.",
        reply_markup=back_kb(),
    )
    return GIF_WAIT_VIDEO


async def handle_gif_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message

    if msg.video:
        fobj  = msg.video
        fname = fobj.file_name or "video.mp4"
    elif msg.document:
        fobj  = msg.document
        mime  = fobj.mime_type or ""
        fname = fobj.file_name or "file"
        if not mime.startswith("video/") and not Path(fname).suffix.lower() in {
            ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".3gp", ".ts", ".m4v"
        }:
            await msg.reply_text(
                "❌ Это не видеофайл. Отправь видео!\n\nИли вернись через /start",
                reply_markup=back_kb(),
            )
            return GIF_WAIT_VIDEO
    elif msg.animation:
        fobj  = msg.animation
        fname = fobj.file_name or "anim.mp4"
    else:
        await msg.reply_text(
            "Отправь видеофайл. Если потерялся — напиши /start",
            reply_markup=back_kb(),
        )
        return GIF_WAIT_VIDEO

    tg_duration = getattr(fobj, "duration", None)
    if tg_duration and tg_duration > GIF_MAX_DURATION:
        await msg.reply_text(
            f"❌ Видео слишком длинное: {tg_duration} сек.\n"
            f"Максимум — {GIF_MAX_DURATION} секунд. Обрежь видео и попробуй снова.",
            reply_markup=back_kb(),
        )
        return GIF_WAIT_VIDEO

    ctx.user_data["gif_file_id"]  = fobj.file_id
    ctx.user_data["gif_file_name"] = fname
    log_action(update.effective_user, f"загрузил видео для GIF: {fname}")

    dur_txt = f" ({tg_duration} сек)" if tg_duration else ""
    await msg.reply_text(
        f"Видео получено{dur_txt}. Выбери качество GIF:",
        reply_markup=gif_quality_kb(),
    )
    return GIF_WAIT_QUALITY


async def btn_gif_quality(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    key = q.data.replace("gifq_", "")
    if key not in GIF_QUALITY_PRESETS:
        return GIF_WAIT_QUALITY

    width, fps, label, max_mb = GIF_QUALITY_PRESETS[key]
    file_id  = ctx.user_data.get("gif_file_id")
    fname    = ctx.user_data.get("gif_file_name", "video.mp4")

    log_action(q.from_user, f"выбрал качество GIF: {label}")

    status = await q.edit_message_text(
        f"... Конвертирую в GIF\n"
        f"Качество: {label} · {width}p · {fps} кадр/с\n\n"
        "Это займёт от 10 до 60 секунд."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ext         = Path(fname).suffix or ".mp4"
        input_file  = Path(tmpdir) / f"input{ext}"
        output_file = Path(tmpdir) / "output.gif"

        try:
            tg_file = await ctx.bot.get_file(file_id)
            await tg_file.download_to_drive(str(input_file))
        except Exception as e:
            await status.edit_text(
                f"❌ Не удалось скачать файл: {e}\n\nНапиши /start и попробуй снова."
            )
            return GIF_WAIT_VIDEO

        duration = get_video_duration(str(input_file))
        if duration and duration > GIF_MAX_DURATION:
            await status.edit_text(
                f"❌ Видео слишком длинное: {duration:.0f} сек.\n"
                f"Максимум — {GIF_MAX_DURATION} секунд.\n\nНапиши /start"
            )
            return GIF_WAIT_VIDEO

        ok, err, final_w, final_fps = convert_to_gif_adaptive(
            str(input_file), str(output_file), width, fps, max_mb
        )
        if not ok:
            log.error(f"ffmpeg gif error: {err}")
            await status.edit_text(
                "❌ Ошибка при конвертации. Попробуй другой файл.\n\nНапиши /start"
            )
            return GIF_WAIT_VIDEO

        gif_size    = format_file_size(str(output_file))
        size_mb     = Path(output_file).stat().st_size / 1_048_576
        reduced_note = f" (авто-уменьшено до {final_w}p/{final_fps}fps)" if (final_w != width or final_fps != fps) else ""
        log_action(q.from_user, f"✅ GIF готов: {gif_size}{reduced_note}")

        if size_mb > GIF_TELEGRAM_LIMIT:
            await status.edit_text(
                f"❌ GIF слишком большой ({gif_size}) — Telegram не принимает файлы > {GIF_TELEGRAM_LIMIT} МБ.\n\n"
                "Попробуй более короткое видео или выбери качество «Лёгкая».\n\nНапиши /start"
            )
            return GIF_WAIT_VIDEO

        await status.edit_text(f"... Отправляю GIF ({gif_size}){reduced_note}")

        # Сначала анимация — превью в чате
        with open(output_file, "rb") as f:
            await update.callback_query.message.reply_animation(
                animation=f,
                caption=f"GIF готов · {final_w}p · {final_fps} кадр/с · {gif_size}"
                        + (f"\n{reduced_note.strip()}" if reduced_note else ""),
            )

        # Потом тот же файл документом — чтобы можно было скачать как .gif
        with open(output_file, "rb") as f:
            await update.callback_query.message.reply_document(
                document=f,
                filename="result.gif",
                caption=(
                    "⬆️ Выше — превью\n"
                    "📎 Здесь — файл .gif для скачивания\n\n"
                    "Отправь ещё видео или вернись в меню.\n\n"
                    "В случае неполадок — /start"
                ),
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("← Главное меню", callback_data="back_menu")]
                ]),
            )

        await status.delete()

    ctx.user_data.pop("gif_file_id", None)
    ctx.user_data.pop("gif_file_name", None)
    return GIF_WAIT_VIDEO


# ════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ: коллаж
# ════════════════════════════════════════════════════════════════════════

def collage_crop_to_fill(img: Image.Image, tw: int, th: int) -> Image.Image:
    sw, sh = img.size
    if sw / sh > tw / th:
        nw = int(sh * tw / th)
        img = img.crop(((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh))
    else:
        nh = int(sw * th / tw)
        img = img.crop((0, (sh - nh) // 2, sw, (sh - nh) // 2 + nh))
    return img.resize((tw, th), Image.LANCZOS)


def collage_fit_into_cell(img: Image.Image, tw: int, th: int, bg: tuple) -> Image.Image:
    sw, sh = img.size
    scale = min(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    cell = Image.new("RGB", (tw, th), bg)
    cell.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return cell


def make_collage(images: list, template_key: str, fit_mode: str, gap: int, bg_key: str) -> Image.Image:
    tmpl     = COLLAGE_TEMPLATES[template_key]
    use_crop = fit_mode == "fit_crop"
    bg_color = (255, 255, 255) if bg_key == "bg_white" else (0, 0, 0)
    cw       = CANVAS_W
    ch       = int(cw / tmpl["ratio"])

    if bg_key == "bg_blur":
        base = images[0].convert("RGB").resize((cw, ch), Image.LANCZOS)
        canvas = base.filter(ImageFilter.GaussianBlur(radius=30))
    else:
        canvas = Image.new("RGB", (cw, ch), bg_color)

    hg = gap // 2
    for i, (cx, cy, cfw, cfh) in enumerate(tmpl["cells"]):
        if i >= len(images):
            break
        px, py = int(cx * cw) + hg, int(cy * ch) + hg
        pw, ph = int(cfw * cw) - gap, int(cfh * ch) - gap
        if pw <= 0 or ph <= 0:
            continue
        src = images[i].convert("RGB")
        tile = collage_crop_to_fill(src, pw, ph) if use_crop else collage_fit_into_cell(src, pw, ph, bg_color if bg_key != "bg_blur" else (0, 0, 0))
        canvas.paste(tile, (px, py))
    return canvas


# ── Клавиатуры коллажа ──────────────────────────────────────────────────

def col_template_kb(count: int) -> InlineKeyboardMarkup:
    rows = []
    for key, t in COLLAGE_TEMPLATES.items():
        if t["count"] > count:
            continue
        mark = "◼ " if t["count"] == count else ""
        rows.append([InlineKeyboardButton(f"{mark}{t['label']}  [{t['count']} фото]", callback_data=f"coltpl_{key}")])
    rows.append([InlineKeyboardButton("← Главное меню", callback_data="back_menu")])
    return InlineKeyboardMarkup(rows)


def col_fit_kb() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(label, callback_data=key)] for key, label in COLLAGE_FIT_OPTIONS.items()]
    rows.append([InlineKeyboardButton("← Главное меню", callback_data="back_menu")])
    return InlineKeyboardMarkup(rows)


def col_gap_kb() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(label, callback_data=key)] for key, (label, _) in COLLAGE_GAP_OPTIONS.items()]
    rows.append([InlineKeyboardButton("← Главное меню", callback_data="back_menu")])
    return InlineKeyboardMarkup(rows)


def col_bg_kb() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(label, callback_data=key)] for key, label in COLLAGE_BG_OPTIONS.items()]
    rows.append([InlineKeyboardButton("← Главное меню", callback_data="back_menu")])
    return InlineKeyboardMarkup(rows)


# ── Хэндлеры коллажа ────────────────────────────────────────────────────

async def btn_go_collage(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["col_photos"] = []
    log_action(q.from_user, "выбрал режим: коллаж")
    await safe_edit(q,
        "◻ *Коллаж из фото*\n\n"
        "Отправляй фото по одной (от 2 до 9).\n"
        "После последнего нажми *«Готово»*.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
    )
    return COL_WAIT_PHOTOS


async def handle_col_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg    = update.message
    photos = ctx.user_data.setdefault("col_photos", [])

    if msg.photo:
        file_id = msg.photo[-1].file_id
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        file_id = msg.document.file_id
    else:
        await msg.reply_text("Отправь фотографию.")
        return COL_WAIT_PHOTOS

    photos.append(file_id)
    count = len(photos)

    if count >= 9:
        await msg.reply_text(
            f"Добавлено {count} фото (максимум). Выбери шаблон:",
            reply_markup=col_template_kb(count),
        )
        return COL_WAIT_TEMPLATE

    done_kb = InlineKeyboardMarkup([[InlineKeyboardButton(f"→  Готово ({count} фото)", callback_data="col_done")]])
    await msg.reply_text(f"Фото {count} добавлено. Отправь ещё или нажми «Готово».", reply_markup=done_kb)
    return COL_WAIT_PHOTOS


async def btn_col_done(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    photos = ctx.user_data.get("col_photos", [])
    if len(photos) < 2:
        await update.callback_query.edit_message_text("Нужно минимум 2 фото. Отправь ещё.")
        return COL_WAIT_PHOTOS
    await update.callback_query.edit_message_text(
        f"Фото собраны: {len(photos)} шт. Выбери шаблон коллажа:",
        reply_markup=col_template_kb(len(photos)),
    )
    return COL_WAIT_TEMPLATE


async def btn_col_template(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    key = q.data.replace("coltpl_", "")
    ctx.user_data["col_template"] = key
    await q.edit_message_text(
        f"Шаблон: {COLLAGE_TEMPLATES[key]['label']}\n\nКак вписывать фото в клетки?",
        reply_markup=col_fit_kb(),
    )
    return COL_WAIT_FIT


async def btn_col_fit(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["col_fit"] = q.data
    await q.edit_message_text(
        f"Режим: {COLLAGE_FIT_OPTIONS[q.data].split(' — ')[0]}\n\nВыбери отступы:",
        reply_markup=col_gap_kb(),
    )
    return COL_WAIT_GAP


async def btn_col_gap(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["col_gap"] = q.data
    label, _ = COLLAGE_GAP_OPTIONS[q.data]
    await q.edit_message_text(
        f"Отступы: {label}\n\nВыбери цвет фона:",
        reply_markup=col_bg_kb(),
    )
    return COL_WAIT_BG


async def btn_col_bg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    bg_key       = q.data
    template_key = ctx.user_data["col_template"]
    fit_mode     = ctx.user_data.get("col_fit", "fit_crop")
    _, gap_px    = COLLAGE_GAP_OPTIONS[ctx.user_data["col_gap"]]
    photo_ids    = ctx.user_data.get("col_photos", [])
    needed       = COLLAGE_TEMPLATES[template_key]["count"]

    status = await q.edit_message_text("... Создаю коллаж")

    images = []
    try:
        for fid in photo_ids[:needed]:
            tg_file = await ctx.bot.get_file(fid)
            buf = io.BytesIO()
            await tg_file.download_to_memory(buf)
            buf.seek(0)
            images.append(Image.open(buf).copy())
    except Exception as e:
        await status.edit_text(f"❌ Ошибка при загрузке фото: {e}\n\nНапиши /start")
        return COL_WAIT_PHOTOS

    try:
        collage = make_collage(images, template_key, fit_mode, gap_px, bg_key)
    except Exception as e:
        log.exception("Ошибка при создании коллажа")
        await status.edit_text(f"❌ Ошибка: {e}\n\nНапиши /start")
        return COL_WAIT_PHOTOS

    out = io.BytesIO()
    collage.save(out, format="JPEG", quality=95)
    out.seek(0)
    await status.delete()
    await q.message.reply_photo(
        photo=out,
        caption="Коллаж готов. Отправь новые фото или вернись в меню.\n\nВ случае неполадок запустите бота заново — /start",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
    )
    log_action(q.from_user, f"✅ коллаж {template_key} готов")

    ctx.user_data.pop("col_photos", None)
    ctx.user_data.pop("col_template", None)
    ctx.user_data.pop("col_fit", None)
    ctx.user_data.pop("col_gap", None)
    return COL_WAIT_PHOTOS


# ════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ: стикерпак
# ════════════════════════════════════════════════════════════════════════

def add_drop_shadow(img: Image.Image, offset: int = 6, blur_r: int = 8, opacity: int = 170) -> Image.Image:
    """Добавляет мягкую тень под стикер (прозрачный фон)."""
    _, _, _, a = img.split()
    shadow_fill = Image.new("RGBA", img.size, (0, 0, 0, opacity))
    shadow_fill.putalpha(a)
    blurred = shadow_fill.filter(ImageFilter.GaussianBlur(radius=blur_r))
    result  = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result.paste(blurred, (offset, offset), blurred)
    result  = Image.alpha_composite(result, img)
    return result


# ════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ: Текст → эмодзи
# ════════════════════════════════════════════════════════════════════════

def _load_font(font_key: str, size: int):
    path = FONT_PATHS.get(font_key, FONT_PATHS["impact"])
    for p in (path, r"C:\Windows\Fonts\ariblk.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if p:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_text_image(
    text: str, font_key: str, rows: int,
    color_rgba: tuple, add_txt_shadow: bool,
) -> Image.Image:
    """Рендерит текст на прозрачном холсте WIDE_COLS×rows ячеек (RGBA)."""
    img_w = WIDE_COLS * STICKER_SIZE
    img_h = rows      * STICKER_SIZE

    font_size = int(img_h * 0.78)
    font      = _load_font(font_key, font_size)

    canvas = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw   = ImageDraw.Draw(canvas)
    bbox   = draw.textbbox((0, 0), text, font=font)
    tw     = bbox[2] - bbox[0]

    # Уменьшаем шрифт если текст шире холста
    if tw > img_w * 0.94:
        font_size = int(font_size * (img_w * 0.94) / tw)
        font      = _load_font(font_key, font_size)
        bbox      = draw.textbbox((0, 0), text, font=font)
        tw        = bbox[2] - bbox[0]

    th = bbox[3] - bbox[1]
    x  = (img_w - tw) // 2 - bbox[0]
    y  = (img_h - th) // 2 - bbox[1]
    sh = max(3, img_h // 25)   # смещение тени

    stroke_w = max(2, font_size // 22)

    if font_key == "shadow_3d":
        draw.text((x + sh, y + sh), text, font=font, fill=(0, 0, 0, 225))
        draw.text((x, y),           text, font=font, fill=color_rgba)
    elif font_key == "outline":
        # Полые буквы: только обводка, внутри прозрачно
        if add_txt_shadow:
            draw.text((x + sh, y + sh), text, font=font,
                      fill=(0,0,0,0), stroke_width=stroke_w+2, stroke_fill=(0,0,0,110))
        draw.text((x, y), text, font=font,
                  fill=(0, 0, 0, 0), stroke_width=stroke_w, stroke_fill=color_rgba)
    else:
        if add_txt_shadow:
            draw.text((x + sh, y + sh), text, font=font, fill=(0, 0, 0, 130))
        draw.text((x, y), text, font=font, fill=color_rgba)

    return canvas


def render_oval_base(text: str, rows: int,
                     color: tuple = (255, 255, 255, 255),
                     font_key: str = "impact") -> Image.Image:
    """Текст + неоновая pill-рамка, прозрачный фон."""
    img_w = WIDE_COLS * STICKER_SIZE
    img_h = rows      * STICKER_SIZE

    font_size = int(img_h * 0.68)
    font      = _load_font(font_key, font_size)

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox  = probe.textbbox((0, 0), text, font=font)
    tw    = bbox[2] - bbox[0]
    if tw > img_w * 0.78:
        font_size = int(font_size * img_w * 0.78 / tw)
        font      = _load_font(font_key, font_size)
        bbox      = probe.textbbox((0, 0), text, font=font)
        tw        = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (img_w - tw) // 2 - bbox[0]
    y = (img_h - th) // 2 - bbox[1]

    pad_x  = int(img_h * 0.22)
    pad_y  = int(img_h * 0.07)
    rx1    = max(6, x + bbox[0] - pad_x)
    ry1    = max(6, y + bbox[1] - pad_y)
    rx2    = min(img_w - 6, x + bbox[2] + pad_x)
    ry2    = min(img_h - 6, y + bbox[3] + pad_y)
    radius = (ry2 - ry1) // 2

    r, g, b = color[0], color[1], color[2]

    # Прозрачный фон
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))

    # Неоновое свечение
    for i in range(6, 0, -1):
        glow = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        ImageDraw.Draw(glow).rounded_rectangle(
            [rx1 - i * 3, ry1 - i * 3, rx2 + i * 3, ry2 + i * 3],
            radius=radius + i * 3,
            outline=(r, g, b, max(20, 80 // i)),
            width=2,
        )
        img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(radius=i * 2)))

    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([rx1, ry1, rx2, ry2], radius=radius,
                            outline=(r, g, b, 255), width=3)
    draw.text((x, y), text, font=font, fill=(r, g, b, 255))
    return img


def make_oval_frames(base_img: Image.Image, fps: int = 20) -> tuple[list, float]:
    """Oval: пульсирующее свечение рамки."""
    n_frm = int(fps * MAX_VIDEO_SEC)
    base  = np.array(base_img.convert("RGBA"), dtype=np.float32)
    frames = []
    for i in range(n_frm):
        t      = i / n_frm * 2 * np.pi * 2   # 2 цикла
        bright = 0.65 + 0.35 * np.sin(t)      # 0.65–1.0
        frame  = base.copy()
        frame[:, :, :3] = np.clip(base[:, :, :3] * bright, 0, 255)
        frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    return frames, float(fps)


def render_rect_base(text: str, rows: int,
                     color: tuple = (255, 255, 255, 255),
                     font_key: str = "impact") -> Image.Image:
    """Текст + неоновая прямоугольная рамка, прозрачный фон."""
    img_w = WIDE_COLS * STICKER_SIZE
    img_h = rows      * STICKER_SIZE

    font_size = int(img_h * 0.68)
    font      = _load_font(font_key, font_size)

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox  = probe.textbbox((0, 0), text, font=font)
    tw    = bbox[2] - bbox[0]
    if tw > img_w * 0.78:
        font_size = int(font_size * img_w * 0.78 / tw)
        font      = _load_font(font_key, font_size)
        bbox      = probe.textbbox((0, 0), text, font=font)
        tw        = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (img_w - tw) // 2 - bbox[0]
    y = (img_h - th) // 2 - bbox[1]

    pad_x = int(img_h * 0.22)
    pad_y = int(img_h * 0.07)
    rx1   = max(6, x + bbox[0] - pad_x)
    ry1   = max(6, y + bbox[1] - pad_y)
    rx2   = min(img_w - 6, x + bbox[2] + pad_x)
    ry2   = min(img_h - 6, y + bbox[3] + pad_y)
    radius = 12

    r, g, b = color[0], color[1], color[2]

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))

    for i in range(6, 0, -1):
        glow = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        ImageDraw.Draw(glow).rounded_rectangle(
            [rx1 - i * 3, ry1 - i * 3, rx2 + i * 3, ry2 + i * 3],
            radius=radius,
            outline=(r, g, b, max(20, 80 // i)),
            width=2,
        )
        img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(radius=i * 2)))

    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([rx1, ry1, rx2, ry2], radius=radius,
                            outline=(r, g, b, 255), width=3)
    draw.text((x, y), text, font=font, fill=(r, g, b, 255))
    return img


def render_glow_base(text: str, rows: int,
                     color: tuple = (255, 255, 255, 255),
                     font_key: str = "impact") -> Image.Image:
    """Текст с неоновым свечением, прозрачный фон — без рамки."""
    img_w = WIDE_COLS * STICKER_SIZE
    img_h = rows      * STICKER_SIZE

    font_size = int(img_h * 0.72)
    font      = _load_font(font_key, font_size)

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox  = probe.textbbox((0, 0), text, font=font)
    tw    = bbox[2] - bbox[0]
    if tw > img_w * 0.90:
        font_size = int(font_size * img_w * 0.90 / tw)
        font      = _load_font(font_key, font_size)
        bbox      = probe.textbbox((0, 0), text, font=font)
        tw        = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (img_w - tw) // 2 - bbox[0]
    y = (img_h - th) // 2 - bbox[1]

    r, g, b = color[0], color[1], color[2]

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))

    for radius in (18, 12, 7, 3):
        glow = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        alpha = {18: 60, 12: 90, 7: 130, 3: 180}[radius]
        ImageDraw.Draw(glow).text((x, y), text, font=font, fill=(r, g, b, alpha))
        img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(radius=radius)))

    ImageDraw.Draw(img).text((x, y), text, font=font, fill=(r, g, b, 255))
    return img


def make_shimmer_frames(base_img: Image.Image, fps: int = 20) -> tuple[list, float]:
    """Анимация: световой блик скользит по тексту."""
    w, h  = base_img.size
    n_frm = int(fps * MAX_VIDEO_SEC)
    base  = np.array(base_img.convert("RGBA"), dtype=np.float32)
    alpha = base[:, :, 3]

    frames = []
    for i in range(n_frm):
        xc    = int(i / n_frm * (w + 240)) - 120
        dist  = np.abs(np.arange(w, dtype=np.float32) - xc)
        bright = np.clip(1.0 - dist / 90.0, 0.0, 1.0) * 210.0
        frame  = base.copy()
        for c in range(3):
            frame[:, :, c] = np.clip(
                base[:, :, c] + bright[np.newaxis, :] * (alpha / 255.0), 0, 255)
        frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    return frames, float(fps)


def make_pulse_frames(base_img: Image.Image, fps: int = 20) -> tuple[list, float]:
    """Анимация: текст плавно мигает."""
    n_frm = int(fps * MAX_VIDEO_SEC)
    base  = np.array(base_img.convert("RGBA"), dtype=np.float32)
    frames = []
    for i in range(n_frm):
        t     = i / n_frm * 2 * np.pi * 1.5
        scale = 0.45 + 0.55 * (np.sin(t) * 0.5 + 0.5)
        frame = base.copy()
        frame[:, :, 3] = np.clip(base[:, :, 3] * scale, 0, 255)
        frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    return frames, float(fps)


def make_glitch_frames(base_img: Image.Image, fps: int = 20) -> tuple[list, float]:
    """Анимация: глитч — горизонтальные полосы со смещением."""
    import random
    rng   = random.Random(42)
    w, h  = base_img.size
    n_frm = int(fps * MAX_VIDEO_SEC)
    base  = np.array(base_img.convert("RGBA"))
    frames = []
    for i in range(n_frm):
        frame = base.copy()
        if i % 7 in (0, 1):
            y0    = rng.randint(0, max(1, h - 15))
            y1    = min(y0 + rng.randint(4, 18), h)
            shift = rng.choice([-10, -6, 6, 10])
            if shift > 0:
                frame[y0:y1, shift:, :]  = base[y0:y1, :-shift, :]
                frame[y0:y1, :shift, 3]  = 0
            else:
                frame[y0:y1, :shift, :]  = base[y0:y1, -shift:, :]
                frame[y0:y1, shift:,  3] = 0
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA))
    return frames, float(fps)


def make_wave_frames(base_img: Image.Image, fps: int = 20) -> tuple[list, float]:
    """Плавная волна: столбцы качаются вверх-вниз."""
    w, h   = base_img.size
    n_frm  = int(fps * MAX_VIDEO_SEC)
    base   = np.array(base_img.convert("RGBA"))
    amp    = max(2, int(h * 0.07))
    xs     = np.arange(w, dtype=np.float32)
    phase  = xs / w * 2 * np.pi * 2   # 2 волны по ширине

    frames = []
    for i in range(n_frm):
        t      = i / n_frm * 2 * np.pi
        shifts = (amp * np.sin(phase + t)).astype(int)
        frame  = np.zeros_like(base)
        for x in range(w):
            s = shifts[x]
            if s >= 0:
                frame[s:, x, :]   = base[:h - s, x, :]
            else:
                frame[:h + s, x, :] = base[-s:, x, :]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA))
    return frames, float(fps)


def make_fade_frames(base_img: Image.Image, fps: int = 20) -> tuple[list, float]:
    """Плавное появление и исчезание (fade in → hold → fade out)."""
    n_frm = int(fps * MAX_VIDEO_SEC)
    base  = np.array(base_img.convert("RGBA"), dtype=np.float32)
    frames = []
    for i in range(n_frm):
        t = i / n_frm
        if   t < 0.30: scale = t / 0.30
        elif t < 0.70: scale = 1.0
        else:          scale = (1.0 - t) / 0.30
        scale = max(0.04, min(1.0, scale))
        frame = base.copy()
        frame[:, :, 3] = np.clip(base[:, :, 3] * scale, 0, 255)
        frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    return frames, float(fps)


def make_rainbow_frames(base_img: Image.Image, fps: int = 20) -> tuple[list, float]:
    """Радужные цвета переливаются по тексту."""
    w, h  = base_img.size
    n_frm = int(fps * MAX_VIDEO_SEC)
    base  = np.array(base_img.convert("RGBA"), dtype=np.float32)
    alpha = base[:, :, 3]

    frames = []
    for i in range(n_frm):
        t    = i / n_frm
        hues = (np.arange(w, dtype=np.float32) / w * 1.5 + t) % 1.0
        rgb  = np.array([colorsys.hsv_to_rgb(h, 0.95, 1.0) for h in hues],
                        dtype=np.float32)   # (w, 3)
        frame = np.zeros((h, w, 4), dtype=np.float32)
        for c in range(3):
            frame[:, :, c] = rgb[:, c][np.newaxis, :] * (alpha / 255.0) * 255
        frame[:, :, 3] = alpha
        frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    return frames, float(fps)


def make_scan_frames(base_img: Image.Image, fps: int = 20) -> tuple[list, float]:
    """Луч-сканер движется сверху вниз."""
    w, h  = base_img.size
    n_frm = int(fps * MAX_VIDEO_SEC)
    base  = np.array(base_img.convert("RGBA"), dtype=np.float32)
    alpha = base[:, :, 3]
    scan_w = max(3, h // 8)

    frames = []
    for i in range(n_frm):
        t      = (i / n_frm * 2) % 1.0     # 2 прохода за 3 секунды
        scan_y = int(t * (h + scan_w * 2)) - scan_w
        ys     = np.arange(h, dtype=np.float32)

        # Затемнение ниже луча
        dim = np.where(ys > scan_y,
                       np.clip(1.0 - (ys - scan_y) / h * 0.65, 0.35, 1.0),
                       1.0)
        # Сам луч — яркая полоса
        bright = np.clip(1.0 - np.abs(ys - scan_y) / scan_w, 0, 1) * 170

        frame = base.copy()
        for c in range(3):
            frame[:, :, c] = np.clip(
                base[:, :, c] * dim[:, np.newaxis]
                + bright[:, np.newaxis] * (alpha / 255.0), 0, 255)
        frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    return frames, float(fps)


def prepare_sticker(img: Image.Image) -> Image.Image:
    """Вписывает изображение в 512×512 с прозрачным фоном."""
    img = img.convert("RGBA")
    w, h = img.size
    scale = min(STK_SIZE / w, STK_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (STK_SIZE, STK_SIZE), (0, 0, 0, 0))
    canvas.paste(resized, ((STK_SIZE - nw) // 2, (STK_SIZE - nh) // 2))
    return canvas


def split_vertical_sticker(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Разрезает вертикальное фото на 2 стикера 512×512.
    Масштабирует по ширине 512, располагает по центру на холсте 512×1024,
    остаток — прозрачный. Возвращает (верх, низ).
    """
    img = img.convert("RGBA")
    w, h = img.size
    scale = STK_SIZE / w
    nw, nh = STK_SIZE, int(h * scale)

    scaled = img.resize((nw, nh), Image.LANCZOS)
    canvas_h = STK_SIZE * 2
    canvas = Image.new("RGBA", (STK_SIZE, canvas_h), (0, 0, 0, 0))
    paste_y = (canvas_h - nh) // 2
    canvas.paste(scaled, (0, paste_y))

    top    = canvas.crop((0, 0,        STK_SIZE, STK_SIZE))
    bottom = canvas.crop((0, STK_SIZE, STK_SIZE, canvas_h))
    return top, bottom


def has_transparency(img: Image.Image) -> bool:
    """Проверяет, есть ли в изображении прозрачные пиксели."""
    if img.mode == "RGBA":
        return np.array(img)[:, :, 3].min() < 255
    return False


def sticker_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def create_sticker_pack(ctx, user_id: int, sticker_images: list[Image.Image], title: str) -> str:
    bot  = ctx.bot
    me   = await bot.get_me()
    name = f"s{str(int(time.time()))[-6:]}_{user_id}_by_{me.username}"

    first = InputSticker(
        sticker=io.BytesIO(sticker_to_bytes(sticker_images[0])),
        emoji_list=["⭐"],
        format=StickerFormat.STATIC,
    )
    await _tg_retry(bot.create_new_sticker_set,
        user_id=user_id,
        name=name,
        title=_make_title(title),
        stickers=[first],
        sticker_type=StickerType.REGULAR,
    )
    for img in sticker_images[1:]:
        await _tg_retry(bot.add_sticker_to_set,
            user_id=user_id,
            name=name,
            sticker=InputSticker(
                sticker=io.BytesIO(sticker_to_bytes(img)),
                emoji_list=["⭐"],
                format=StickerFormat.STATIC,
            ),
        )
    return name


async def create_anim_sticker_pack(ctx, user_id: int, webm_paths: list, title: str) -> str:
    """Создаёт анимированный стикерпак из готовых WebM-файлов."""
    bot  = ctx.bot
    me   = await bot.get_me()
    name = f"a{str(int(time.time()))[-6:]}_{user_id}_by_{me.username}"

    first = InputSticker(
        sticker=open(str(webm_paths[0]), "rb"),
        emoji_list=["⭐"],
        format=StickerFormat.VIDEO,
    )
    await _tg_retry(bot.create_new_sticker_set,
        user_id=user_id,
        name=name,
        title=_make_title(title),
        stickers=[first],
        sticker_type=StickerType.REGULAR,
    )
    for p in webm_paths[1:]:
        await _tg_retry(bot.add_sticker_to_set,
            user_id=user_id,
            name=name,
            sticker=InputSticker(
                sticker=open(str(p), "rb"),
                emoji_list=["⭐"],
                format=StickerFormat.VIDEO,
            ),
        )
    return name


# ── Клавиатуры стикерпака ───────────────────────────────────────────────

def stk_mode_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("◼  Обычный стикерпак",                callback_data="stk_regular")],
        [InlineKeyboardButton("◻  Вертикальный (1 фото → 2 стикера)", callback_data="stk_vertical")],
        [InlineKeyboardButton("▶  Анимированный (видео/GIF/MP4)",    callback_data="stk_anim")],
        [InlineKeyboardButton("← Главное меню",                      callback_data="back_menu")],
    ])

def stk_shadow_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("◼  Да — добавить тень",  callback_data="stk_shadow_yes")],
        [InlineKeyboardButton("—  Нет, без тени",       callback_data="stk_shadow_no")],
        [InlineKeyboardButton("← Главное меню",         callback_data="back_menu")],
    ])

def stk_bg_kb(prefix: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("□  Убрать белый фон",  callback_data=f"{prefix}bgr_white")],
        [InlineKeyboardButton("■  Убрать чёрный фон", callback_data=f"{prefix}bgr_black")],
        [InlineKeyboardButton("→  Оставить как есть", callback_data=f"{prefix}bgr_skip")],
    ])


# ── Хэндлеры: стикерпак (обычный) ───────────────────────────────────────

async def btn_go_stickers(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["stk_photos"] = []
    log_action(q.from_user, "выбрал режим: стикерпак")
    await safe_edit(q,
        "▪ *Стикерпак*\n\n"
        "Если у твоих фото прозрачный фон — он сохранится прозрачным в Telegram.\n"
        "_(прозрачный фон можно сделать в Photoshop)_\n\n"
        "Выбери режим:",
        reply_markup=stk_mode_kb(),
    )
    return STK_WAIT_MODE


async def btn_stk_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["stk_photos"] = []
    ctx.user_data["stk_webms"]  = []
    ctx.user_data["stk_mode"]   = q.data  # stk_regular / stk_vertical / stk_anim

    if q.data == "stk_anim":
        await q.edit_message_text(
            "▶ *Анимированный стикерпак*\n\n"
            "Отправляй видео по одному — каждое станет отдельным анимированным стикером.\n\n"
            "Поддерживаются: *WebM, MOV, MP4, GIF*\n"
            "⚠️ Telegram ограничивает стикеры до *3 секунд* — если видео длиннее, "
            "бот возьмёт первые 3 секунды.\n\n"
            "Прозрачный фон (WebM с альфа-каналом) — сохранится.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
            parse_mode="Markdown",
        )
        return STK_WAIT_ANIM_FILE

    await q.edit_message_text(
        "Добавить тень к стикерам?\n\n"
        "Тень делает стикеры объёмнее — лучше смотрятся на любом фоне. Рекомендуем ✓",
        reply_markup=stk_shadow_kb(),
    )
    return STK_WAIT_SHADOW


async def handle_stk_anim_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    file_id, fname = None, "video"

    if msg.video:
        file_id = msg.video.file_id
        fname   = msg.video.file_name or "video.mp4"
    elif msg.animation:
        file_id = msg.animation.file_id
        fname   = msg.animation.file_name or "anim.mp4"
    elif msg.document:
        mime = (msg.document.mime_type or "").lower()
        fl   = (msg.document.file_name or "").lower()
        if (mime.startswith("video/") or mime == "image/gif"
                or fl.endswith((".webm", ".mov", ".mp4", ".gif"))):
            file_id = msg.document.file_id
            fname   = msg.document.file_name or "video"
    if not file_id:
        await msg.reply_text(
            "Отправь видео или GIF.\nПоддерживаются: WebM, MOV, MP4, GIF."
        )
        return STK_WAIT_ANIM_FILE

    status = await msg.reply_text("... Конвертирую в стикер")
    try:
        tg_file = await ctx.bot.get_file(file_id)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / fname
            await tg_file.download_to_drive(str(src))

            # Получаем длительность для предупреждения
            try:
                r = subprocess.run(
                    [FFPROBE, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
                    capture_output=True, text=True, timeout=15,
                )
                orig_dur = float(r.stdout.strip())
            except Exception:
                orig_dur = 0.0

            out = Path(tmp) / "sticker.webm"
            video_to_sticker_webm(str(src), out)
            with open(out, "rb") as f:
                sticker_bytes = f.read()

        ctx.user_data.setdefault("stk_webms", []).append(sticker_bytes)
        n = len(ctx.user_data["stk_webms"])
        trimmed_note = f"\n_Обрезано до 3 с._" if orig_dur > MAX_VIDEO_SEC else ""
        done_kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"→  Готово ({n} стикеров)", callback_data="stk_anim_done")
        ]])
        await status.edit_text(
            f"Стикер {n} добавлен.{trimmed_note}\nОтправь ещё или нажми «Готово».",
            reply_markup=done_kb,
            parse_mode="Markdown",
        )
    except Exception as e:
        log.exception("stk_anim convert error")
        await status.edit_text(f"Ошибка конвертации: {e}")
    return STK_WAIT_ANIM_FILE


async def btn_stk_anim_done(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    webms = ctx.user_data.get("stk_webms", [])
    if not webms:
        await q.edit_message_text("Нет ни одного стикера. Отправь видео.")
        return STK_WAIT_ANIM_FILE
    await q.edit_message_text(
        f"Стикеров готово: {len(webms)}.\n\nВведи название для стикерпака (до 64 символов):"
    )
    return STK_WAIT_NAME


async def btn_stk_shadow(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["stk_shadow"] = (q.data == "stk_shadow_yes")
    mode = ctx.user_data.get("stk_mode", "stk_regular")
    shadow_txt = "Тень: включена ✓" if ctx.user_data["stk_shadow"] else "Тень: выключена"

    if mode == "stk_regular":
        await q.edit_message_text(
            f"◼ *Обычный стикерпак*  |  {shadow_txt}\n\n"
            "Отправляй фото по одному — каждое станет отдельным стикером.\n"
            "Можно добавить от 1 до 50 стикеров.\n\n"
            "PNG/WebP с прозрачным фоном — фон сохранится.\n"
            "JPG — предложу убрать белый или чёрный фон.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
            parse_mode="Markdown",
        )
        return STK_WAIT_PHOTO
    else:
        await q.edit_message_text(
            f"◻ *Вертикальный стикерпак*  |  {shadow_txt}\n\n"
            "Отправь одно вертикальное фото — я разрежу его на 2 стикера 512×512.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
            parse_mode="Markdown",
        )
        return STK_VERT_PHOTO


async def handle_stk_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg.photo:
        file_id = msg.photo[-1].file_id
        fname   = "photo.jpg"
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        file_id = msg.document.file_id
        fname   = msg.document.file_name or "file"
    else:
        await msg.reply_text("Отправь фото или изображение.")
        return STK_WAIT_PHOTO

    # Скачиваем
    tg_file = await ctx.bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(buf)
    buf.seek(0)
    img = Image.open(buf).copy()

    ctx.user_data["stk_current"] = img
    count = len(ctx.user_data.get("stk_photos", []))

    # Если PNG/WebP с прозрачностью — сразу добавляем
    fname_lower = fname.lower()
    is_transparent_format = fname_lower.endswith((".png", ".webp")) or \
                            (msg.document and (msg.document.mime_type or "") in ("image/png", "image/webp"))

    if is_transparent_format and has_transparency(img):
        sticker = prepare_sticker(img)
        if ctx.user_data.get("stk_shadow"):
            sticker = add_drop_shadow(sticker)
        ctx.user_data.setdefault("stk_photos", []).append(sticker)
        n = len(ctx.user_data["stk_photos"])
        done_kb = InlineKeyboardMarkup([[InlineKeyboardButton(f"→  Готово ({n} стикеров)", callback_data="stk_done")]])
        await msg.reply_text(
            f"Стикер {n} добавлен — прозрачный фон сохранён.\nОтправь ещё или нажми «Готово».",
            reply_markup=done_kb,
        )
        return STK_WAIT_PHOTO

    # Иначе предлагаем убрать фон
    await msg.reply_text(
        f"Фото получено. Убрать фон перед добавлением в стикерпак?",
        reply_markup=stk_bg_kb("stk_"),
    )
    return STK_WAIT_BG


async def btn_stk_bg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    mode = q.data.replace("stk_bgr_", "")
    img  = ctx.user_data.get("stk_current")

    if img is None:
        await q.edit_message_text("Что-то пошло не так. Напиши /start")
        return STK_WAIT_PHOTO

    if mode in ("white", "black"):
        await q.edit_message_text("... Убираю фон")
        try:
            img = remove_solid_bg(img, mode)
            await q.edit_message_text("Фон убран.")
        except Exception as e:
            await q.edit_message_text(f"Ошибка: {e}\nДобавляю без изменений.")
    else:
        await q.edit_message_text("→ Фон оставлен как есть.")

    sticker = prepare_sticker(img)
    if ctx.user_data.get("stk_shadow"):
        sticker = add_drop_shadow(sticker)
    ctx.user_data.setdefault("stk_photos", []).append(sticker)
    n = len(ctx.user_data["stk_photos"])

    done_kb = InlineKeyboardMarkup([[InlineKeyboardButton(f"→  Готово ({n} стикеров)", callback_data="stk_done")]])
    await q.message.reply_text(
        f"Стикер {n} добавлен. Отправь ещё или нажми «Готово».",
        reply_markup=done_kb,
    )
    return STK_WAIT_PHOTO


async def btn_stk_done(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    photos = ctx.user_data.get("stk_photos", [])
    if not photos:
        await q.edit_message_text("Нет ни одного стикера. Отправь фото.")
        return STK_WAIT_PHOTO
    n = len(photos)
    await q.edit_message_text(
        f"Стикеров готово: {n}.\n\nВведи название для стикерпака (до 64 символов):"
    )
    return STK_WAIT_NAME


async def handle_stk_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    title    = update.message.text.strip()[:64]
    user_id  = update.effective_user.id
    stk_mode = ctx.user_data.get("stk_mode", "stk_regular")
    webms    = ctx.user_data.get("stk_webms", [])
    photos   = ctx.user_data.get("stk_photos", [])

    # Анимированный стикерпак
    if stk_mode == "stk_anim":
        if not webms:
            await update.message.reply_text("Нет стикеров. Начни заново — /start")
            return STK_WAIT_ANIM_FILE
        status = await update.message.reply_text(f"... Создаю анимированный стикерпак «{title}» ({len(webms)} стикеров)")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for i, data in enumerate(webms):
                    p = Path(tmp) / f"s_{i:04d}.webm"
                    p.write_bytes(data)
                    paths.append(p)
                pack_name = await create_anim_sticker_pack(ctx, user_id, paths, title)
            log_action(update.effective_user, f"✅ anim-стикерпак «{title}» → {pack_name}")
            await status.edit_text(
                f"Анимированный стикерпак «{title}» создан.\n\n"
                f"→ t.me/addstickers/{pack_name}\n\n"
                "В случае неполадок запустите бота заново — /start",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
            )
        except Exception as e:
            log.exception("Ошибка создания anim-стикерпака")
            await status.edit_text(
                f"Ошибка: {e}\n\nНапиши /start чтобы начать заново.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
            )
        ctx.user_data.pop("stk_webms", None)
        return MAIN_MENU

    # Обычный стикерпак
    if not photos:
        await update.message.reply_text("Нет стикеров. Начни заново — /start")
        return STK_WAIT_PHOTO

    status = await update.message.reply_text(f"... Создаю стикерпак «{title}» ({len(photos)} стикеров)")
    try:
        pack_name = await create_sticker_pack(ctx, user_id, photos, title)
        log_action(update.effective_user, f"✅ стикерпак «{title}» → {pack_name}")
        await status.edit_text(
            f"Стикерпак «{title}» создан.\n\n"
            f"→ t.me/addstickers/{pack_name}\n\n"
            "В случае неполадок запустите бота заново — /start",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
        )
    except Exception as e:
        log.exception("Ошибка создания стикерпака")
        await status.edit_text(
            f"Ошибка: {e}\n\nНапиши /start чтобы начать заново.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
        )
    ctx.user_data.pop("stk_photos", None)
    ctx.user_data.pop("stk_current", None)
    return MAIN_MENU


# ── Хэндлеры: вертикальный стикерпак ────────────────────────────────────

async def handle_stk_vert_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg.photo:
        file_id = msg.photo[-1].file_id
        fname   = "photo.jpg"
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        file_id = msg.document.file_id
        fname   = msg.document.file_name or "file"
    else:
        await msg.reply_text("Отправь фото.")
        return STK_VERT_PHOTO

    tg_file = await ctx.bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(buf)
    buf.seek(0)
    img = Image.open(buf).copy()
    w, h = img.size

    if w >= h:
        await msg.reply_text(
            "Лучше отправь вертикальное фото (выше чем шире) для красивой нарезки.\n"
            "Хочешь продолжить с этим фото?",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("→  Да, продолжить", callback_data="stk_vert_continue")],
                [InlineKeyboardButton("← Отмена",          callback_data="back_menu")],
            ]),
        )
        ctx.user_data["stk_vert_img"] = img
        return STK_VERT_PHOTO

    ctx.user_data["stk_vert_img"] = img
    fname_lower = fname.lower()
    is_transparent_format = fname_lower.endswith((".png", ".webp")) or \
                            (msg.document and (msg.document.mime_type or "") in ("image/png", "image/webp"))

    if is_transparent_format and has_transparency(img):
        return await _do_vert_split(msg, ctx, img)

    await msg.reply_text(
        "Фото получено. Убрать фон перед нарезкой?",
        reply_markup=stk_bg_kb("stk_vert_"),
    )
    return STK_VERT_BG


async def btn_stk_vert_continue(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    img = ctx.user_data.get("stk_vert_img")
    if img is None:
        await q.edit_message_text("Что-то пошло не так. Напиши /start")
        return STK_VERT_PHOTO
    await q.edit_message_text("Убрать фон перед нарезкой?", reply_markup=stk_bg_kb("stk_vert_"))
    return STK_VERT_BG


async def btn_stk_vert_bg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    mode = q.data.replace("stk_vert_bgr_", "")
    img  = ctx.user_data.get("stk_vert_img")

    if mode in ("white", "black"):
        await q.edit_message_text("... Убираю фон")
        try:
            img = remove_solid_bg(img, mode)
        except Exception as e:
            await q.edit_message_text(f"Ошибка: {e}\nПродолжаю без изменений.")
    else:
        await q.edit_message_text("→ Фон оставлен как есть.")

    return await _do_vert_split(q.message, ctx, img)


async def _do_vert_split(msg, ctx: ContextTypes.DEFAULT_TYPE, img: Image.Image):
    top, bottom = split_vertical_sticker(img)
    if ctx.user_data.get("stk_shadow"):
        top    = add_drop_shadow(top)
        bottom = add_drop_shadow(bottom)
    ctx.user_data["stk_vert_stickers"] = [top, bottom]
    await msg.reply_text(
        "Фото нарезано на 2 стикера.\n\nВведи название для стикерпака:",
    )
    return STK_VERT_NAME


async def handle_stk_vert_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    title   = update.message.text.strip()[:64]
    user_id = update.effective_user.id
    stickers = ctx.user_data.get("stk_vert_stickers", [])

    if not stickers:
        await update.message.reply_text("Нет стикеров. Начни заново — /start")
        return STK_VERT_PHOTO

    status = await update.message.reply_text(f"... Создаю вертикальный стикерпак «{title}»")
    try:
        pack_name = await create_sticker_pack(ctx, user_id, stickers, title)
        log_action(update.effective_user, f"✅ верт. стикерпак «{title}» → {pack_name}")
        await status.edit_text(
            f"Стикерпак «{title}» создан (2 стикера).\n\n"
            f"→ t.me/addstickers/{pack_name}\n\n"
            "В случае неполадок запустите бота заново — /start",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
        )
    except Exception as e:
        log.exception("Ошибка создания вертикального стикерпака")
        await status.edit_text(
            f"Ошибка: {e}\n\nНапиши /start чтобы начать заново.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Главное меню", callback_data="back_menu")]]),
        )
    ctx.user_data.pop("stk_vert_img", None)
    ctx.user_data.pop("stk_vert_stickers", None)
    return MAIN_MENU


# ════════════════════════════════════════════════════════════════════════
#  ХЭНДЛЕРЫ: Текст → эмодзи
# ════════════════════════════════════════════════════════════════════════

def txt_type_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("○  Обычный (статичный)",          callback_data="txt_type_static")],
        [InlineKeyboardButton("▶  Анимированный",                callback_data="txt_type_anim")],
        [InlineKeyboardButton("⬜  Овал — неоновая рамка",         callback_data="txt_type_oval")],
        [InlineKeyboardButton("▭  Прямоугольник — неоновая рамка", callback_data="txt_type_rect")],
        [InlineKeyboardButton("✦  Свечение — неоновый текст",    callback_data="txt_type_glow")],
        [InlineKeyboardButton("← Главное меню",                  callback_data="back_menu")],
    ])

def txt_font_kb():
    rows = [[InlineKeyboardButton(label, callback_data=f"txt_font_{key}")]
            for key, label in FONT_LABELS.items()]
    rows.append([InlineKeyboardButton("← Главное меню", callback_data="back_menu")])
    return InlineKeyboardMarkup(rows)

def txt_height_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"—  1 строка  (макс. {TXT_MAX_CHARS[1]} симв.)", callback_data="txt_h_1")],
        [InlineKeyboardButton(f"◻  2 строки  (макс. {TXT_MAX_CHARS[2]} симв.)", callback_data="txt_h_2")],
        [InlineKeyboardButton(f"◼  3 строки  (макс. {TXT_MAX_CHARS[3]} симв.)", callback_data="txt_h_3")],
        [InlineKeyboardButton("← Главное меню", callback_data="back_menu")],
    ])

def txt_anim_kb():
    rows = [[InlineKeyboardButton(label, callback_data=f"txt_anim_{key}")]
            for key, label in TXT_ANIM_OPTIONS.items()]
    rows.append([InlineKeyboardButton("← Главное меню", callback_data="back_menu")])
    return InlineKeyboardMarkup(rows)

def txt_color_kb():
    rows = [[InlineKeyboardButton(label, callback_data=f"txt_color_{key}")]
            for key, (_, label) in TXT_COLORS.items()]
    rows.append([InlineKeyboardButton("← Главное меню", callback_data="back_menu")])
    return InlineKeyboardMarkup(rows)


async def btn_go_txt(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data.clear()
    log_action(q.from_user, "выбрал режим: текст→эмодзи")
    await safe_edit(q,
        "Ａ *Текст → эмодзи*\n\n"
        "Превращу любой текст в набор кастомных эмодзи.\n\n"
        "Выбери тип:",
        reply_markup=txt_type_kb(),
    )
    return TXT_WAIT_TYPE


async def btn_txt_type(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    t = q.data
    ctx.user_data["txt_animated"] = t != "txt_type_static"
    ctx.user_data["txt_oval"]     = t == "txt_type_oval"
    ctx.user_data["txt_rect"]     = t == "txt_type_rect"
    ctx.user_data["txt_glow"]     = t == "txt_type_glow"

    neon_hints = {
        "txt_type_oval": "⬜ *Овал* — рамка pill + неоновое свечение, прозрачный фон",
        "txt_type_rect": "▭ *Прямоугольник* — рамка + неоновое свечение, прозрачный фон",
        "txt_type_glow": "✦ *Свечение* — неоновый текст без рамки, прозрачный фон",
    }
    hint = neon_hints.get(t, "")
    prefix = f"{hint}\n\n" if hint else ""

    await q.edit_message_text(
        f"{prefix}Выбери шрифт:",
        reply_markup=txt_font_kb(),
        parse_mode="Markdown",
    )
    return TXT_WAIT_FONT


async def btn_txt_font(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    key = q.data.replace("txt_font_", "")
    ctx.user_data["txt_font"] = key
    await q.edit_message_text(
        f"Шрифт: {FONT_LABELS[key]}\n\nВыбери высоту текста:",
        reply_markup=txt_height_kb(),
    )
    return TXT_WAIT_HEIGHT


async def btn_txt_height(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    rows = int(q.data.replace("txt_h_", ""))
    ctx.user_data["txt_rows"] = rows
    max_chars = TXT_MAX_CHARS[rows]

    if ctx.user_data.get("txt_animated"):
        await q.edit_message_text(
            f"Высота: {rows} {'строка' if rows == 1 else 'строки'}.\n\nВыбери анимацию:",
            reply_markup=txt_anim_kb(),
        )
        return TXT_WAIT_ANIM

    await q.edit_message_text(
        f"Высота: {rows} {'строка' if rows == 1 else 'строки'}.\n\nВыбери цвет текста:",
        reply_markup=txt_color_kb(),
    )
    return TXT_WAIT_COLOR


async def btn_txt_anim(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    key = q.data.replace("txt_anim_", "")
    ctx.user_data["txt_anim"] = key
    label = TXT_ANIM_OPTIONS.get(key, key)
    await q.edit_message_text(
        f"Анимация: {label}\n\nВыбери цвет текста:",
        reply_markup=txt_color_kb(),
    )
    return TXT_WAIT_COLOR


async def btn_txt_color(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    key = q.data.replace("txt_color_", "")
    ctx.user_data["txt_color"] = key
    rows      = ctx.user_data.get("txt_rows", 1)
    max_chars = TXT_MAX_CHARS[rows]
    _, label  = TXT_COLORS[key]
    await q.edit_message_text(
        f"Цвет: {label}\n\n"
        f"Введи текст — не более *{max_chars} символов*:\n\n"
        "Совет: ЗАГЛАВНЫЕ буквы смотрятся выразительнее.",
        reply_markup=back_kb(),
        parse_mode="Markdown",
    )
    return TXT_WAIT_TEXT


async def handle_txt_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text      = update.message.text.strip()
    rows      = ctx.user_data.get("txt_rows", 1)
    max_chars = TXT_MAX_CHARS[rows]

    if len(text) > max_chars:
        await update.message.reply_text(
            f"❌ Текст слишком длинный ({len(text)} симв.).\n"
            f"Для высоты {rows} — максимум {max_chars} символов.\n\nПопробуй снова:",
            reply_markup=back_kb(),
        )
        return TXT_WAIT_TEXT

    ctx.user_data["txt_text"] = text
    await update.message.reply_text(
        f"Текст: «{text}»\n\nВведи название для эмодзи-пака:",
        reply_markup=back_kb(),
    )
    return TXT_WAIT_NAME


async def handle_txt_pack_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id   = update.effective_user.id
    title     = update.message.text.strip()[:64]
    text      = ctx.user_data.get("txt_text", "TEXT")
    font_key  = ctx.user_data.get("txt_font", "impact")
    rows      = ctx.user_data.get("txt_rows", 1)
    color_key = ctx.user_data.get("txt_color", "white")
    is_anim   = ctx.user_data.get("txt_animated", False)
    anim_key  = ctx.user_data.get("txt_anim", "shimmer")

    color_rgba, _ = TXT_COLORS[color_key]
    add_sh        = rows >= 2  # тень для многострочного

    progress = await update.message.reply_text(
        f"... Создаю пак «{title}» из текста «{text}»\n"
        "Рендерю шрифт и нарезаю на ячейки..."
    )

    try:
        is_oval = ctx.user_data.get("txt_oval", False)
        is_rect = ctx.user_data.get("txt_rect", False)
        is_glow = ctx.user_data.get("txt_glow", False)

        if is_oval:
            base_img = render_oval_base(text, rows, color_rgba, font_key)
        elif is_rect:
            base_img = render_rect_base(text, rows, color_rgba, font_key)
        elif is_glow:
            base_img = render_glow_base(text, rows, color_rgba, font_key)
        else:
            base_img = render_text_image(text, font_key, rows, color_rgba, add_sh)
        cols = WIDE_COLS  # 12 колонок

        with tempfile.TemporaryDirectory() as tmpdir:
            cells_dir = Path(tmpdir) / "cells"
            cells_dir.mkdir()

            neon_anim_map = {
                "none":    None,
                "pulse":   make_oval_frames,   # пульс = default для неона
                "shimmer": make_shimmer_frames,
                "wave":    make_wave_frames,
                "fade":    make_fade_frames,
                "rainbow": make_rainbow_frames,
                "scan":    make_scan_frames,
                "glitch":  make_glitch_frames,
            }
            neon_no_anim = (is_oval or is_rect or is_glow) and anim_key == "none"

            if (is_oval or is_rect or is_glow) and not neon_no_anim:
                anim_fn = neon_anim_map.get(anim_key, make_oval_frames)
                frames, fps = anim_fn(base_img)
                cell_list   = split_frames_to_cells(frames, cols, rows)
                for i, cf in enumerate(cell_list):
                    cell_frames_to_webm(cf, cells_dir / f"cell_{i:04d}.webm", fps)
                sticker_paths = sorted(cells_dir.glob("*.webm"))
                fmt = StickerFormat.VIDEO
            elif is_anim and anim_key != "none":
                anim_funcs = {
                    "shimmer": make_shimmer_frames,
                    "pulse":   make_pulse_frames,
                    "wave":    make_wave_frames,
                    "fade":    make_fade_frames,
                    "rainbow": make_rainbow_frames,
                    "scan":    make_scan_frames,
                    "glitch":  make_glitch_frames,
                }
                frames, fps = anim_funcs.get(anim_key, make_shimmer_frames)(base_img)
                cell_list   = split_frames_to_cells(frames, cols, rows)
                for i, cf in enumerate(cell_list):
                    cell_frames_to_webm(cf, cells_dir / f"cell_{i:04d}.webm", fps)
                sticker_paths = sorted(cells_dir.glob("*.webm"))
                fmt = StickerFormat.VIDEO
            else:
                # Статичный вариант (в т.ч. неон без анимации)
                cells         = slice_image(base_img, cols, rows)
                sticker_paths = save_cells_png(cells, cells_dir)
                fmt           = StickerFormat.STATIC

            pack_name = await build_emoji_pack(
                ctx, user_id, sticker_paths, fmt, _make_title(title),
                progress_chat_id=progress.chat_id,
                progress_msg_id=progress.message_id,
            )
            log_action(update.effective_user, f"✅ создал пак «{title}» текст→эмодзи")

            pack_url = f"https://t.me/addemoji/{pack_name}"

            await progress.edit_text(
                f"✅ Эмодзи-пак «{title}» создан.\n\n"
                f"→ {pack_url}\n\n"
                f"Добавь пак по ссылке — и ниже появится готовая сетка.\n"
                f"Можешь скопировать её и вставить в пост как есть.\n\n"
                "В случае неполадок — /start",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("↺  Создать ещё", callback_data="back_menu")]
                ]),
            )
            await asyncio.sleep(5)   # ждём пока Telegram закэширует весь пак
            await send_emoji_preview(
                ctx.bot, progress.chat_id, pack_name, cols, rows
            )

    except Exception as e:
        log.exception("Ошибка text→emoji")
        await progress.edit_text(
            f"❌ Ошибка: {e}\n\nНапиши /start",
            reply_markup=back_kb(),
        )

    ctx.user_data.clear()
    return MAIN_MENU


# ════════════════════════════════════════════════════════════════════════
#  ЗАПУСК
# ════════════════════════════════════════════════════════════════════════

def main():
    request = HTTPXRequest(
        connection_pool_size=16,
        read_timeout=120,
        write_timeout=120,
        connect_timeout=30,
        pool_timeout=30,
    )
    upd_request = HTTPXRequest(
        connection_pool_size=4,
        read_timeout=45,
        write_timeout=45,
        connect_timeout=30,
        pool_timeout=30,
    )
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .get_updates_request(upd_request)
        .build()
    )
    app.add_error_handler(global_error_handler)

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(btn_go_start,    pattern="^go_start$"),
                CallbackQueryHandler(btn_help,        pattern="^help$"),
                CallbackQueryHandler(btn_help_ps,     pattern="^help_ps$"),
                CallbackQueryHandler(btn_donate,      pattern="^donate$"),
                CallbackQueryHandler(btn_go_emoji,    pattern="^go_emoji$"),
                CallbackQueryHandler(btn_go_circle,   pattern="^go_circle$"),
                CallbackQueryHandler(btn_go_gif,      pattern="^go_gif$"),
                CallbackQueryHandler(btn_go_collage,  pattern="^go_collage$"),
                CallbackQueryHandler(btn_go_stickers, pattern="^go_stickers$"),
                CallbackQueryHandler(btn_go_txt,      pattern="^go_txt$"),
                # Кнопка возврата работает и из MAIN_MENU (после завершения задач)
                CallbackQueryHandler(btn_back,        pattern="^back_menu$"),
                CallbackQueryHandler(btn_check_sub,   pattern="^check_sub$"),
            ],
            CIRCLE_WAIT_VIDEO: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_circle_video),
            ],
            EMOJI_WAIT_FILE: [
                CallbackQueryHandler(btn_back,       pattern="^back_menu$"),
                CallbackQueryHandler(btn_emoji_mode, pattern="^mode_(image|video|gif)$"),
                MessageHandler(
                    filters.Document.ALL | filters.Sticker.ALL |
                    filters.ANIMATION | filters.VIDEO | filters.PHOTO,
                    handle_emoji_file,
                ),
            ],
            EMOJI_WAIT_ASPECT: [
                CallbackQueryHandler(btn_back,         pattern="^back_menu$"),
                CallbackQueryHandler(btn_emoji_aspect, pattern=r"^aspect_(universal|android|ios|desktop)$"),
            ],
            EMOJI_WAIT_GRID: [
                CallbackQueryHandler(btn_back,             pattern="^back_menu$"),
                CallbackQueryHandler(btn_emoji_grid_custom, pattern="^grid_custom$"),
                CallbackQueryHandler(btn_emoji_grid,       pattern=r"^grid_\d+_\d+$"),
            ],
            EMOJI_WAIT_CUSTOM_GRID: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_emoji_custom_grid),
            ],
            EMOJI_WAIT_PACK_NAME: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_emoji_pack_name),
            ],
            GIF_WAIT_VIDEO: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                MessageHandler(filters.VIDEO | filters.Document.ALL | filters.ANIMATION, handle_gif_video),
            ],
            GIF_WAIT_QUALITY: [
                CallbackQueryHandler(btn_back,        pattern="^back_menu$"),
                CallbackQueryHandler(btn_gif_quality, pattern=r"^gifq_(low|medium|high)$"),
            ],
            COL_WAIT_PHOTOS: [
                CallbackQueryHandler(btn_back,     pattern="^back_menu$"),
                CallbackQueryHandler(btn_col_done, pattern="^col_done$"),
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_col_photo),
            ],
            COL_WAIT_TEMPLATE: [
                CallbackQueryHandler(btn_back,         pattern="^back_menu$"),
                CallbackQueryHandler(btn_col_template, pattern=r"^coltpl_\w+$"),
            ],
            COL_WAIT_FIT: [
                CallbackQueryHandler(btn_back,    pattern="^back_menu$"),
                CallbackQueryHandler(btn_col_fit, pattern=r"^fit_(crop|fit)$"),
            ],
            COL_WAIT_GAP: [
                CallbackQueryHandler(btn_back,    pattern="^back_menu$"),
                CallbackQueryHandler(btn_col_gap, pattern=r"^gap\d$"),
            ],
            COL_WAIT_BG: [
                CallbackQueryHandler(btn_back,   pattern="^back_menu$"),
                CallbackQueryHandler(btn_col_bg, pattern=r"^bg_\w+$"),
            ],
            STK_WAIT_MODE: [
                CallbackQueryHandler(btn_back,     pattern="^back_menu$"),
                CallbackQueryHandler(btn_stk_mode, pattern=r"^stk_(regular|vertical|anim)$"),
            ],
            STK_WAIT_SHADOW: [
                CallbackQueryHandler(btn_back,       pattern="^back_menu$"),
                CallbackQueryHandler(btn_stk_shadow, pattern=r"^stk_shadow_(yes|no)$"),
            ],
            STK_WAIT_PHOTO: [
                CallbackQueryHandler(btn_back,     pattern="^back_menu$"),
                CallbackQueryHandler(btn_stk_done, pattern="^stk_done$"),
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_stk_photo),
            ],
            STK_WAIT_BG: [
                CallbackQueryHandler(btn_back,   pattern="^back_menu$"),
                CallbackQueryHandler(btn_stk_bg, pattern=r"^stk_bgr_(white|black|skip)$"),
            ],
            STK_WAIT_ANIM_FILE: [
                CallbackQueryHandler(btn_back,          pattern="^back_menu$"),
                CallbackQueryHandler(btn_stk_anim_done, pattern="^stk_anim_done$"),
                MessageHandler(
                    filters.VIDEO | filters.ANIMATION |
                    filters.Document.VIDEO |
                    filters.Document.MimeType("image/gif") |
                    filters.Document.MimeType("video/mp4") |
                    filters.Document.MimeType("video/webm") |
                    filters.Document.MimeType("video/quicktime"),
                    handle_stk_anim_file,
                ),
            ],
            STK_WAIT_NAME: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_stk_name),
            ],
            STK_VERT_SHADOW: [
                CallbackQueryHandler(btn_back,       pattern="^back_menu$"),
                CallbackQueryHandler(btn_stk_shadow, pattern=r"^stk_shadow_(yes|no)$"),
            ],
            STK_VERT_PHOTO: [
                CallbackQueryHandler(btn_back,              pattern="^back_menu$"),
                CallbackQueryHandler(btn_stk_vert_continue, pattern="^stk_vert_continue$"),
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_stk_vert_photo),
            ],
            STK_VERT_BG: [
                CallbackQueryHandler(btn_back,        pattern="^back_menu$"),
                CallbackQueryHandler(btn_stk_vert_bg, pattern=r"^stk_vert_bgr_(white|black|skip)$"),
            ],
            STK_VERT_NAME: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_stk_vert_name),
            ],
            TXT_WAIT_TYPE: [
                CallbackQueryHandler(btn_back,     pattern="^back_menu$"),
                CallbackQueryHandler(btn_txt_type, pattern=r"^txt_type_(static|anim|oval|rect|glow)$"),
            ],
            TXT_WAIT_FONT: [
                CallbackQueryHandler(btn_back,     pattern="^back_menu$"),
                CallbackQueryHandler(btn_txt_font, pattern=r"^txt_font_\w+$"),
            ],
            TXT_WAIT_HEIGHT: [
                CallbackQueryHandler(btn_back,       pattern="^back_menu$"),
                CallbackQueryHandler(btn_txt_height, pattern=r"^txt_h_[123]$"),
            ],
            TXT_WAIT_ANIM: [
                CallbackQueryHandler(btn_back,     pattern="^back_menu$"),
                CallbackQueryHandler(btn_txt_anim, pattern=r"^txt_anim_(none|shimmer|pulse|wave|fade|rainbow|scan|glitch)$"),
            ],
            TXT_WAIT_COLOR: [
                CallbackQueryHandler(btn_back,      pattern="^back_menu$"),
                CallbackQueryHandler(btn_txt_color, pattern=r"^txt_color_\w+$"),
            ],
            TXT_WAIT_TEXT: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_txt_text),
            ],
            TXT_WAIT_NAME: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_txt_pack_name),
            ],
        },
        fallbacks=[
            CommandHandler("start", cmd_start),
            MessageHandler(filters.ALL, fallback_start_hint),
        ],
        allow_reentry=True,
        per_message=False,
    )

    app.add_handler(conv)

    while True:
        try:
            log.info("Бот запускается...")
            app.run_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
                close_loop=False,
            )
            break  # нормальное завершение
        except (NetworkError, TimedOut) as e:
            log.warning(f"Сетевая ошибка при старте: {e}. Повтор через 15 сек...")
            import time as _time; _time.sleep(15)
        except Exception as e:
            log.exception(f"Критическая ошибка: {e}")
            raise


if __name__ == "__main__":
    main()

