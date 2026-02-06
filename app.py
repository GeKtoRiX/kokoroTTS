import spaces
from kokoro import KModel, KPipeline
import gradio as gr
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import torch
from typing import Optional
import uuid
import wave
import warnings

@dataclass(frozen=True)
class AppConfig:
    log_level: str
    file_log_level: str
    log_dir: str
    log_file: str
    repo_id: str
    output_dir: str
    output_dir_abs: str
    max_chunk_chars: int
    history_limit: int
    normalize_times: bool
    normalize_numbers: bool
    default_output_format: str
    default_concurrency_limit: Optional[int]
    space_id: str
    is_duplicate: bool
    char_limit: Optional[int]

def _resolve_path(value, base_dir):
    if not os.path.isabs(value):
        return os.path.join(base_dir, value)
    return value

def _parse_int_env(name, default, min_value=None, max_value=None):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value

def load_config():
    base_dir = os.path.dirname(__file__)
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    file_log_level = os.getenv('FILE_LOG_LEVEL', 'DEBUG').upper()
    log_dir = _resolve_path(os.getenv('LOG_DIR', 'logs'), base_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log")
    repo_id = os.getenv('KOKORO_REPO_ID', 'hexgrad/Kokoro-82M')
    output_dir = _resolve_path(os.getenv('OUTPUT_DIR', 'outputs'), base_dir)
    output_dir_abs = os.path.abspath(output_dir)
    max_chunk_chars = _parse_int_env('MAX_CHUNK_CHARS', 500, min_value=50, max_value=2000)
    history_limit = _parse_int_env('HISTORY_LIMIT', 5, min_value=0, max_value=20)
    normalize_times = os.getenv('NORMALIZE_TIMES', '1').strip().lower() not in ('0', 'false', 'no', 'off')
    normalize_numbers = os.getenv('NORMALIZE_NUMBERS', '1').strip().lower() not in ('0', 'false', 'no', 'off')
    default_output_format = os.getenv('DEFAULT_OUTPUT_FORMAT', 'wav').strip().lower()
    default_concurrency_limit = _parse_int_env('DEFAULT_CONCURRENCY_LIMIT', 0, min_value=0)
    if default_concurrency_limit == 0:
        default_concurrency_limit = None
    space_id = os.getenv('SPACE_ID', '')
    is_duplicate = not space_id.startswith('hexgrad/')
    char_limit = None if is_duplicate else 5000
    return AppConfig(
        log_level=log_level,
        file_log_level=file_log_level,
        log_dir=log_dir,
        log_file=log_file,
        repo_id=repo_id,
        output_dir=output_dir,
        output_dir_abs=output_dir_abs,
        max_chunk_chars=max_chunk_chars,
        history_limit=history_limit,
        normalize_times=normalize_times,
        normalize_numbers=normalize_numbers,
        default_output_format=default_output_format,
        default_concurrency_limit=default_concurrency_limit,
        space_id=space_id,
        is_duplicate=is_duplicate,
        char_limit=char_limit
    )

def setup_logging(config):
    logger = logging.getLogger('kokoro_app')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.log_level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    file_handler = logging.FileHandler(config.log_file, encoding='utf-8')
    file_handler.setLevel(config.file_log_level)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(funcName)s | %(message)s')
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.DEBUG)
    warnings_logger.propagate = False
    for handler in list(warnings_logger.handlers):
        warnings_logger.removeHandler(handler)
    warnings_logger.addHandler(file_handler)

    hf_logger = logging.getLogger('huggingface_hub')
    hf_logger.setLevel(logging.DEBUG)
    hf_logger.propagate = False
    for handler in list(hf_logger.handlers):
        hf_logger.removeHandler(handler)
    hf_logger.addHandler(file_handler)
    return logger

CONFIG = load_config()
logger = setup_logging(CONFIG)

def configure_ffmpeg(logger):
    ffmpeg_env = os.getenv('FFMPEG_BINARY')
    ffprobe_env = os.getenv('FFPROBE_BINARY')
    candidates = []
    if sys.prefix:
        candidates.append(os.path.join(sys.prefix, 'Scripts'))
    candidates.append(os.path.join(os.path.dirname(__file__), '.venv', 'Scripts'))
    for folder in candidates:
        if not folder:
            continue
        ffmpeg_path = os.path.join(folder, 'ffmpeg.exe')
        ffprobe_path = os.path.join(folder, 'ffprobe.exe')
        updated = False
        if not ffmpeg_env and os.path.isfile(ffmpeg_path):
            os.environ['FFMPEG_BINARY'] = ffmpeg_path
            ffmpeg_env = ffmpeg_path
            updated = True
        if not ffprobe_env and os.path.isfile(ffprobe_path):
            os.environ['FFPROBE_BINARY'] = ffprobe_path
            ffprobe_env = ffprobe_path
            updated = True
        if updated and folder not in os.environ.get('PATH', ''):
            os.environ['PATH'] = folder + os.pathsep + os.environ.get('PATH', '')
    if ffmpeg_env:
        logger.debug('FFMPEG_BINARY=%s', ffmpeg_env)
    else:
        logger.debug('FFMPEG_BINARY not set; streaming audio may fail without ffmpeg')
    if ffprobe_env:
        logger.debug('FFPROBE_BINARY=%s', ffprobe_env)

configure_ffmpeg(logger)

logger.info('Starting app')
logger.info('Log file: %s', CONFIG.log_file)
logger.debug(
    'Log config: LOG_LEVEL=%s FILE_LOG_LEVEL=%s LOG_DIR=%s OUTPUT_DIR=%s REPO_ID=%s MAX_CHUNK_CHARS=%s HISTORY_LIMIT=%s NORMALIZE_TIMES=%s NORMALIZE_NUMBERS=%s DEFAULT_OUTPUT_FORMAT=%s DEFAULT_CONCURRENCY_LIMIT=%s',
    CONFIG.log_level,
    CONFIG.file_log_level,
    CONFIG.log_dir,
    CONFIG.output_dir,
    CONFIG.repo_id,
    CONFIG.max_chunk_chars,
    CONFIG.history_limit,
    CONFIG.normalize_times,
    CONFIG.normalize_numbers,
    CONFIG.default_output_format,
    CONFIG.default_concurrency_limit
)

CUDA_AVAILABLE = torch.cuda.is_available()
logger.info(
    'Config: SPACE_ID=%s IS_DUPLICATE=%s CHAR_LIMIT=%s CUDA_AVAILABLE=%s',
    CONFIG.space_id,
    CONFIG.is_duplicate,
    CONFIG.char_limit,
    CUDA_AVAILABLE
)
if os.getenv('HF_TOKEN'):
    logger.debug('HF_TOKEN is set')
else:
    logger.debug('HF_TOKEN is not set; hub requests may be rate-limited')
logger.debug('Python version: %s', sys.version.replace('\n', ' '))
logger.debug('Platform: %s', platform.platform())
logger.debug('Torch version: %s', torch.__version__)
if not CONFIG.is_duplicate:
    import kokoro
    import misaki
    logger.debug('Kokoro version: %s', kokoro.__version__)
    logger.debug('Misaki version: %s', misaki.__version__)

SAMPLE_RATE = 24000
OUTPUT_FORMATS = ['wav', 'mp3', 'ogg']
DEFAULT_OUTPUT_FORMAT = CONFIG.default_output_format if CONFIG.default_output_format in OUTPUT_FORMATS else 'wav'
UI_PRIMARY_HUE = os.getenv('UI_PRIMARY_HUE', 'green').strip() or 'green'
APP_THEME = gr.themes.Base(primary_hue=UI_PRIMARY_HUE)


SENTENCE_BREAK_RE = re.compile(r'([.!?]+)(["\')\]]?)(\s+|$)')
TIME_RE = re.compile(r'\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*([AaPp])\.?\s*[Mm]\.?)?\b')
SOFT_BREAK_RE = re.compile(r'[,;:\-]\s+')
MD_LINK_RE = re.compile(r'\[[^\]]*\]\(/[^)\n]*\)')
SLASHED_RE = re.compile(r'(?<![A-Za-z0-9:])/[^\n/]+/')
VOICE_TAG_RE = re.compile(r'\[(?:voice|speaker|spk|mix|voice_mix)\s*=\s*([^\]]+?)\]', re.IGNORECASE)
MULTI_DOT_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+){2,}\b')
ORDINAL_RE = re.compile(r'\b(\d{1,4})(st|nd|rd|th)\b', re.IGNORECASE)
PERCENT_RE = re.compile(r'\b(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?%')
DECIMAL_RE = re.compile(r'\b(\d{1,3}(?:,\d{3})*|\d+)\.(\d+)\b')
INT_RE = re.compile(r'\b(\d{1,3}(?:,\d{3})*|\d+)\b')

ONES = [
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine'
]
TEENS = [
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'
]
TENS = {
    2: 'twenty',
    3: 'thirty',
    4: 'forty',
    5: 'fifty',
    6: 'sixty',
    7: 'seventy',
    8: 'eighty',
    9: 'ninety'
}
ABBREV_TITLES = {
    'mr', 'ms', 'mrs', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'etc'
}
ABBREV_DOTTED = {'e.g', 'i.e'}

def number_to_words_0_59(value):
    if value < 10:
        return ONES[value]
    if value < 20:
        return TEENS[value - 10]
    tens = value // 10
    ones = value % 10
    tens_word = TENS[tens]
    return tens_word if ones == 0 else f"{tens_word} {ONES[ones]}"

def number_to_words_0_99(value):
    if value < 60:
        return number_to_words_0_59(value)
    tens = value // 10
    ones = value % 10
    tens_word = TENS[tens]
    return tens_word if ones == 0 else f"{tens_word} {ONES[ones]}"

def number_to_words_0_999(value):
    if value < 100:
        return number_to_words_0_99(value)
    hundreds = value // 100
    remainder = value % 100
    words = f"{ONES[hundreds]} hundred"
    if remainder:
        words = f"{words} {number_to_words_0_99(remainder)}"
    return words

def number_to_words_0_9999(value):
    if value < 1000:
        return number_to_words_0_999(value)
    thousands = value // 1000
    remainder = value % 1000
    words = f"{ONES[thousands]} thousand"
    if remainder:
        words = f"{words} {number_to_words_0_999(remainder)}"
    return words

ORDINAL_WORDS = {
    'one': 'first',
    'two': 'second',
    'three': 'third',
    'four': 'fourth',
    'five': 'fifth',
    'six': 'sixth',
    'seven': 'seventh',
    'eight': 'eighth',
    'nine': 'ninth',
    'ten': 'tenth',
    'eleven': 'eleventh',
    'twelve': 'twelfth',
    'thirteen': 'thirteenth',
    'fourteen': 'fourteenth',
    'fifteen': 'fifteenth',
    'sixteen': 'sixteenth',
    'seventeen': 'seventeenth',
    'eighteen': 'eighteenth',
    'nineteen': 'nineteenth',
    'twenty': 'twentieth',
    'thirty': 'thirtieth',
    'forty': 'fortieth',
    'fifty': 'fiftieth',
    'sixty': 'sixtieth',
    'seventy': 'seventieth',
    'eighty': 'eightieth',
    'ninety': 'ninetieth',
    'hundred': 'hundredth',
    'thousand': 'thousandth'
}

def ordinalize_words(words):
    parts = words.split()
    if not parts:
        return words
    last = parts[-1]
    if last in ORDINAL_WORDS:
        parts[-1] = ORDINAL_WORDS[last]
    return " ".join(parts)

def ordinal_to_words(value):
    words = number_to_words_0_9999(value)
    return ordinalize_words(words)

def digits_to_words(digits):
    return " ".join(ONES[int(digit)] for digit in digits)

def time_to_words(hour, minute, ampm_letter):
    if ampm_letter:
        hour = hour % 12
        if hour == 0:
            hour = 12
    hour_word = number_to_words_0_59(hour)
    if minute == 0:
        if ampm_letter:
            suffix = 'a.m.' if ampm_letter.lower() == 'a' else 'p.m.'
            return f"{hour_word} {suffix}"
        if hour == 0:
            return 'midnight'
        if hour == 12:
            return 'noon'
        return f"{hour_word} o'clock"
    if minute < 10:
        minute_word = f"oh {ONES[minute]}"
    else:
        minute_word = number_to_words_0_59(minute)
    if ampm_letter:
        suffix = 'a.m.' if ampm_letter.lower() == 'a' else 'p.m.'
        return f"{hour_word} {minute_word} {suffix}"
    return f"{hour_word} {minute_word}"

def _find_skip_spans(text):
    spans = []
    for match in MD_LINK_RE.finditer(text):
        spans.append((match.start(), match.end()))
    for match in SLASHED_RE.finditer(text):
        spans.append((match.start(), match.end()))
    if not spans:
        return []
    spans.sort(key=lambda item: item[0])
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]

def _merge_spans(spans):
    if not spans:
        return []
    spans.sort(key=lambda item: item[0])
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]

def _apply_outside_spans(text, spans, func):
    if not spans:
        return func(text)
    parts = []
    last = 0
    for start, end in spans:
        if start > last:
            parts.append(func(text[last:start]))
        parts.append(text[start:end])
        last = end
    if last < len(text):
        parts.append(func(text[last:]))
    return ''.join(parts)

def _is_within_spans(index, spans):
    for start, end in spans:
        if start <= index < end:
            return True
    return False

def _is_abbrev(text, punct_index):
    if punct_index >= 3:
        dotted = text[punct_index - 3:punct_index].lower()
        if dotted in ABBREV_DOTTED:
            if punct_index == 3 or not text[punct_index - 4].isalnum():
                return True
    i = punct_index - 1
    while i >= 0 and text[i].isalpha():
        i -= 1
    word = text[i + 1:punct_index].lower()
    if not word:
        return False
    if word in ABBREV_TITLES:
        if i < 0 or not text[i].isalnum():
            return True
    return False

def normalize_times(text):
    replacements = 0
    def repl(match):
        nonlocal replacements
        hour = int(match.group(1))
        minute = int(match.group(2))
        ampm = match.group(3)
        replacements += 1
        return time_to_words(hour, minute, ampm)

    skip_spans = _find_skip_spans(text)
    if not skip_spans:
        updated = TIME_RE.sub(repl, text)
    else:
        parts = []
        last = 0
        for start, end in skip_spans:
            if start > last:
                parts.append(TIME_RE.sub(repl, text[last:start]))
            parts.append(text[start:end])
            last = end
        if last < len(text):
            parts.append(TIME_RE.sub(repl, text[last:]))
        updated = ''.join(parts)
    if replacements:
        logger.debug('Normalized times: %s', replacements)
    return updated

def normalize_numbers(text):
    replacements = 0

    def int_to_words(value_str):
        try:
            value = int(value_str.replace(',', ''))
        except ValueError:
            return None
        if value > 9999:
            return None
        return number_to_words_0_9999(value)

    def repl_percent(match):
        nonlocal replacements
        int_part = match.group(1)
        dec_part = match.group(2)
        int_words = int_to_words(int_part)
        if int_words is None:
            return match.group(0)
        if dec_part:
            int_words = f"{int_words} point {digits_to_words(dec_part)}"
        replacements += 1
        return f"{int_words} percent"

    def repl_ordinal(match):
        nonlocal replacements
        value = int(match.group(1))
        if value > 9999:
            return match.group(0)
        replacements += 1
        return ordinal_to_words(value)

    def repl_decimal(match):
        nonlocal replacements
        int_part = match.group(1)
        dec_part = match.group(2)
        int_words = int_to_words(int_part)
        if int_words is None:
            return match.group(0)
        replacements += 1
        return f"{int_words} point {digits_to_words(dec_part)}"

    def repl_int(match):
        nonlocal replacements
        int_words = int_to_words(match.group(1))
        if int_words is None:
            return match.group(0)
        replacements += 1
        return int_words

    skip_spans = _find_skip_spans(text)
    skip_spans.extend(match.span() for match in MULTI_DOT_NUMBER_RE.finditer(text))
    skip_spans = _merge_spans(skip_spans)

    def apply_all(segment):
        segment = PERCENT_RE.sub(repl_percent, segment)
        segment = ORDINAL_RE.sub(repl_ordinal, segment)
        segment = DECIMAL_RE.sub(repl_decimal, segment)
        segment = INT_RE.sub(repl_int, segment)
        return segment

    updated = _apply_outside_spans(text, skip_spans, apply_all)
    if replacements:
        logger.debug('Normalized numbers: %s', replacements)
    return updated

def split_sentences(text):
    sentences = []
    replacements = 0
    start = 0
    skip_spans = _find_skip_spans(text)
    for match in SENTENCE_BREAK_RE.finditer(text):
        match_start = match.start(1)
        if _is_within_spans(match_start, skip_spans):
            continue
        if _is_abbrev(text, match_start):
            continue
        end = match.end(1)
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
        replacements += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    logger.debug('Sentence split: inserted_breaks=%s count=%s', replacements, len(sentences))
    return sentences

def _split_long_piece(piece, max_chars):
    if len(piece) <= max_chars:
        return [piece]
    parts = [p.strip() for p in SOFT_BREAK_RE.split(piece) if p.strip()]
    if len(parts) <= 1:
        words = piece.split()
        out = []
        current = []
        current_len = 0
        for word in words:
            add_len = len(word) + (1 if current else 0)
            if current_len + add_len > max_chars and current:
                out.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += add_len
        if current:
            out.append(" ".join(current))
        return out

    out = []
    current = ''
    for part in parts:
        if not current:
            current = part
        elif len(current) + 2 + len(part) <= max_chars:
            current = f"{current}, {part}"
        else:
            out.append(current)
            current = part
    if current:
        out.append(current)
    return out

def smart_split(text, max_chars, keep_sentences=False):
    text = text.strip()
    if not text:
        return []
    sentences = split_sentences(text)
    if keep_sentences:
        chunks = []
        for sentence in sentences:
            if len(sentence) > max_chars:
                chunks.extend(_split_long_piece(sentence, max_chars))
            else:
                chunks.append(sentence)
        logger.debug('Smart split: sentences=%s chunks=%s', len(sentences), len(chunks))
        return chunks

    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.extend(_split_long_piece(sentence, max_chars))
            continue
        add_len = len(sentence) + (1 if current else 0)
        if current_len + add_len <= max_chars:
            current.append(sentence)
            current_len += add_len
        else:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
    if current:
        chunks.append(" ".join(current))
    logger.debug('Smart split: sentences=%s chunks=%s', len(sentences), len(chunks))
    return chunks

def split_parts(text):
    if '|' not in text:
        return [text]
    parts = [part.strip() for part in text.split('|')]
    parts = [part for part in parts if part]
    if not parts:
        return ['']
    logger.debug('Manual split parts=%s', len(parts))
    return parts

def normalize_voice_input(voice, voice_mix=None):
    raw = voice_mix if voice_mix and str(voice_mix).strip() else voice
    if isinstance(raw, (list, tuple)):
        raw = ','.join(str(v) for v in raw)
    if raw is None:
        return 'af_heart'
    parts = [p.strip() for p in str(raw).split(',') if p.strip()]
    if not parts:
        return 'af_heart'
    lang = parts[0][0] if parts[0] else 'a'
    mismatched = [p for p in parts[1:] if p and p[0] != lang]
    if mismatched:
        logger.warning('Mixed voices across languages; using pipeline for %s: %s', lang, parts)
    return ','.join(parts)

def resolve_voice(voice, voice_mix, mix_enabled):
    if mix_enabled and voice_mix:
        return normalize_voice_input(voice, voice_mix)
    return normalize_voice_input(voice)

def normalize_voice_tag(raw_value, default_voice):
    cleaned = str(raw_value or '').strip().strip('"').strip("'")
    if not cleaned:
        return default_voice
    lowered = cleaned.lower()
    if lowered in ('default', 'auto'):
        return default_voice
    cleaned = cleaned.replace('+', ',')
    parts = [part.strip() for part in cleaned.split(',') if part.strip()]
    if not parts:
        return default_voice
    resolved = []
    for part in parts:
        part_id = CHOICES.get(part, part)
        if part_id not in CHOICES.values():
            logger.warning('Unknown voice tag "%s"; using default voice', part)
            continue
        resolved.append(part_id)
    if not resolved:
        return default_voice
    return normalize_voice_input(','.join(resolved))

def parse_voice_segments(text, default_voice):
    current_voice = default_voice
    segments = []
    last = 0
    for match in VOICE_TAG_RE.finditer(text):
        start, end = match.span()
        chunk = text[last:start]
        if chunk.strip():
            segments.append((current_voice, chunk.strip()))
        current_voice = normalize_voice_tag(match.group(1), default_voice)
        last = end
    tail = text[last:]
    if tail.strip():
        segments.append((current_voice, tail.strip()))
    return segments

def limit_dialogue_parts(parts, char_limit):
    if char_limit is None:
        return parts, False
    remaining = char_limit
    limited_parts = []
    truncated = False
    for segments in parts:
        limited_segments = []
        for voice, text in segments:
            text = text.strip()
            if not text:
                continue
            if remaining <= 0:
                truncated = True
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip()
                truncated = True
            if text:
                limited_segments.append((voice, text))
                remaining -= len(text)
        if limited_segments:
            limited_parts.append(limited_segments)
        if remaining <= 0:
            break
    return limited_parts, truncated

def summarize_voice(parts, default_voice):
    voices = {voice for segments in parts for voice, _ in segments if voice}
    if not voices:
        return default_voice
    if len(voices) == 1:
        return next(iter(voices))
    return 'multi'

class TextNormalizer:
    def __init__(self, char_limit, normalize_times, normalize_numbers):
        self.char_limit = char_limit
        self.normalize_times = normalize_times
        self.normalize_numbers = normalize_numbers

    def preprocess(self, text, normalize_times_enabled=None, normalize_numbers_enabled=None, apply_char_limit=True):
        if apply_char_limit:
            text = text if self.char_limit is None else text.strip()[:self.char_limit]
        if normalize_times_enabled is None:
            normalize_times_enabled = self.normalize_times
        if normalize_numbers_enabled is None:
            normalize_numbers_enabled = self.normalize_numbers
        if normalize_times_enabled:
            text = normalize_times(text)
        if normalize_numbers_enabled:
            text = normalize_numbers(text)
        return text

class AudioWriter:
    def __init__(self, output_dir, sample_rate, logger):
        self.output_dir = output_dir
        self.output_dir_abs = os.path.abspath(output_dir)
        self.sample_rate = sample_rate
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
        self.ffmpeg_path = os.getenv('FFMPEG_BINARY') or shutil.which('ffmpeg')
        self.can_convert = bool(self.ffmpeg_path)
        self.logger.info('Output dir: %s', self.output_dir)
        if self.can_convert:
            self.logger.debug('FFmpeg available: %s', self.ffmpeg_path)
        else:
            self.logger.warning('FFmpeg not found; output formats other than wav will fall back to wav')

    def _sanitize_voice_id(self, voice):
        safe = re.sub(r'[^A-Za-z0-9_-]+', '_', voice).strip('_')
        return safe or 'voice'

    def resolve_output_format(self, output_format):
        fmt = (output_format or 'wav').strip().lower().lstrip('.')
        if fmt not in OUTPUT_FORMATS:
            fmt = 'wav'
        if fmt != 'wav' and not self.can_convert:
            self.logger.warning('Requested output format %s but ffmpeg not available; falling back to wav', fmt)
            fmt = 'wav'
        return fmt

    def _ensure_extension(self, path, output_format):
        base, ext = os.path.splitext(path)
        if ext.lower().lstrip('.') != output_format:
            return f"{base}.{output_format}"
        return path

    def save_wav(self, path, audio_tensor):
        tensor = audio_tensor if isinstance(audio_tensor, torch.Tensor) else torch.as_tensor(audio_tensor)
        tensor = tensor.detach().cpu().flatten()
        tensor = torch.clamp(tensor, -1.0, 1.0)
        int16 = (tensor * 32767.0).to(torch.int16).numpy()
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16.tobytes())

    def _convert_with_ffmpeg(self, src_path, dst_path):
        ffmpeg = self.ffmpeg_path or shutil.which('ffmpeg')
        if not ffmpeg:
            raise FileNotFoundError('ffmpeg not available')
        subprocess.run(
            [ffmpeg, '-y', '-loglevel', 'error', '-i', src_path, dst_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def save_audio(self, path, audio_tensor, output_format):
        output_format = self.resolve_output_format(output_format)
        path = self._ensure_extension(path, output_format)
        if output_format == 'wav':
            self.save_wav(path, audio_tensor)
            return path
        tmp_handle = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=self.output_dir)
        tmp_path = tmp_handle.name
        tmp_handle.close()
        self.save_wav(tmp_path, audio_tensor)
        try:
            self._convert_with_ffmpeg(tmp_path, path)
        except Exception:
            self.logger.exception('Failed to convert audio to %s; keeping WAV', output_format)
            fallback_path = os.path.splitext(path)[0] + '.wav'
            os.replace(tmp_path, fallback_path)
            return fallback_path
        os.remove(tmp_path)
        return path

    def build_output_paths(self, voice, parts_count, output_format):
        output_format = self.resolve_output_format(output_format)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_voice = self._sanitize_voice_id(voice)
        suffix = uuid.uuid4().hex[:8]
        if parts_count <= 0:
            return []
        if parts_count == 1:
            return [os.path.join(self.output_dir, f"{timestamp}_{safe_voice}_{suffix}.{output_format}")]
        return [
            os.path.join(self.output_dir, f"{timestamp}_{safe_voice}_{suffix}_part{index:02d}.{output_format}")
            for index in range(1, parts_count + 1)
        ]

class ModelManager:
    def __init__(self, repo_id, cuda_available, logger):
        self.repo_id = repo_id
        self.cuda_available = cuda_available
        self.logger = logger
        self.models = {}
        self.pipelines = {}
        self.voice_cache = {}
        self._model_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()
        self._voice_lock = threading.Lock()
        self.logger.info('Models and pipelines will load on demand')

    def get_model(self, use_gpu):
        use_gpu = bool(use_gpu and self.cuda_available)
        key = use_gpu
        model = self.models.get(key)
        if model is not None:
            return model
        with self._model_lock:
            model = self.models.get(key)
            if model is None:
                device = 'cuda' if use_gpu else 'cpu'
                self.logger.info('Loading model on %s', device)
                try:
                    model = KModel(repo_id=self.repo_id).to(device).eval()
                except Exception:
                    self.logger.exception('Failed to load model on %s', device)
                    raise
                self.models[key] = model
                self.logger.info('Model ready on %s', device)
        return model

    def _init_pipeline(self, lang_code):
        self.logger.info('Initializing pipeline for language code=%s', lang_code)
        try:
            pipeline = KPipeline(lang_code=lang_code, repo_id=self.repo_id, model=False)
        except Exception:
            self.logger.exception('Failed to initialize pipeline for language code=%s', lang_code)
            raise
        if lang_code == 'a':
            pipeline.g2p.lexicon.golds['kokoro'] = 'kËˆOkÉ™É¹O'
        elif lang_code == 'b':
            pipeline.g2p.lexicon.golds['kokoro'] = 'kËˆQkÉ™É¹Q'
        self.logger.debug('Updated lexicon entries for kokoro in pipeline=%s', lang_code)
        return pipeline

    def get_pipeline(self, voice):
        lang_code = voice[0] if voice else 'a'
        pipeline = self.pipelines.get(lang_code)
        if pipeline is not None:
            return pipeline
        with self._pipeline_lock:
            pipeline = self.pipelines.get(lang_code)
            if pipeline is None:
                pipeline = self._init_pipeline(lang_code)
                self.pipelines[lang_code] = pipeline
        return pipeline

    def get_voice_pack(self, voice):
        pack = self.voice_cache.get(voice)
        if pack is not None:
            return pack
        with self._voice_lock:
            pack = self.voice_cache.get(voice)
            if pack is not None:
                return pack
            pipeline = self.get_pipeline(voice)
            self.logger.debug('Loading voice pack: %s', voice)
            try:
                pack = pipeline.load_voice(voice)
            except Exception:
                self.logger.exception('Failed to load voice pack: %s', voice)
                raise
            self.voice_cache[voice] = pack
            self.logger.debug('Voice pack cached: %s', voice)
        return pack

class KokoroState:
    def __init__(self, model_manager, normalizer, audio_writer, max_chunk_chars, cuda_available, logger):
        self.model_manager = model_manager
        self.normalizer = normalizer
        self.audio_writer = audio_writer
        self.max_chunk_chars = max_chunk_chars
        self.cuda_available = cuda_available
        self.logger = logger
        self.last_saved_paths = []

    def tokenize_first(self, text, voice='af_heart', speed=1, normalize_times_enabled=None, normalize_numbers_enabled=None):
        parts = self._prepare_dialogue_parts(text, voice, normalize_times_enabled, normalize_numbers_enabled)
        if not parts:
            return ''
        first_voice, first_text = parts[0][0]
        pipeline = self.model_manager.get_pipeline(first_voice)
        for _, ps, _ in pipeline(first_text, first_voice, speed):
            return ps
        return ''

    def _preprocess_text(self, text, normalize_times_enabled=None, normalize_numbers_enabled=None, apply_char_limit=True):
        return self.normalizer.preprocess(
            text,
            normalize_times_enabled=normalize_times_enabled,
            normalize_numbers_enabled=normalize_numbers_enabled,
            apply_char_limit=apply_char_limit
        )

    def _prepare_dialogue_parts(self, text, default_voice, normalize_times_enabled, normalize_numbers_enabled):
        parts = split_parts(text)
        dialogue_parts = []
        for part in parts:
            segments = parse_voice_segments(part, default_voice)
            if segments:
                dialogue_parts.append(segments)
        if not dialogue_parts:
            return []
        limited_parts, truncated = limit_dialogue_parts(dialogue_parts, self.normalizer.char_limit)
        if truncated:
            self.logger.info('Input truncated to %s characters (excluding tags)', self.normalizer.char_limit)
        normalized_parts = []
        for segments in limited_parts:
            normalized_segments = []
            for segment_voice, segment_text in segments:
                segment_text = segment_text.strip()
                if not segment_text:
                    continue
                normalized_text = self._preprocess_text(
                    segment_text,
                    normalize_times_enabled=normalize_times_enabled,
                    normalize_numbers_enabled=normalize_numbers_enabled,
                    apply_char_limit=False
                )
                if normalized_text.strip():
                    normalized_segments.append((segment_voice, normalized_text))
            if normalized_segments:
                normalized_parts.append(normalized_segments)
        return normalized_parts

    def _generate_audio_for_text(self, text, voice, speed, use_gpu, pause_seconds):
        keep_sentences = pause_seconds > 0
        sentences = smart_split(text, self.max_chunk_chars, keep_sentences=keep_sentences)
        pipeline = self.model_manager.get_pipeline(voice)
        pack = self.model_manager.get_voice_pack(voice)
        use_gpu = use_gpu and self.cuda_available
        pause_samples = max(0, int(pause_seconds * SAMPLE_RATE))
        pause_tensor = torch.zeros(pause_samples) if pause_samples else None
        segments = []
        first_ps = ''
        model_cpu = None
        if not use_gpu:
            model_cpu = self.model_manager.get_model(False)
        with torch.inference_mode():
            for index, sentence in enumerate(sentences):
                for _, ps, _ in pipeline(sentence, voice, speed):
                    if not first_ps:
                        first_ps = ps
                    ref_s = pack[len(ps)-1]
                    try:
                        if use_gpu:
                            audio = forward_gpu(ps, ref_s, speed)
                        else:
                            audio = model_cpu(ps, ref_s, speed)
                    except gr.exceptions.Error as e:
                        if use_gpu:
                            gr.Warning(str(e))
                            gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                            audio = self.model_manager.get_model(False)(ps, ref_s, speed)
                        else:
                            raise gr.Error(e)
                    audio = audio.detach().cpu().flatten()
                    segments.append(audio)
                if pause_tensor is not None and index < len(sentences) - 1:
                    segments.append(pause_tensor)
        if not segments:
            return None, first_ps
        return torch.cat(segments), first_ps

    def generate_first(self, text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, pause_seconds=0.0, output_format='wav', normalize_times_enabled=None, normalize_numbers_enabled=None, save_outputs=True):
        parts = self._prepare_dialogue_parts(text, voice, normalize_times_enabled, normalize_numbers_enabled)
        self.logger.debug(
            'Generate start: text_len=%s parts=%s voice=%s speed=%s use_gpu=%s pause_seconds=%s output_format=%s',
            len(text),
            len(parts),
            voice,
            speed,
            use_gpu,
            pause_seconds,
            output_format
        )
        pause_samples = max(0, int(pause_seconds * SAMPLE_RATE))
        pause_tensor = torch.zeros(pause_samples) if pause_samples else None
        combined_segments = []
        first_ps = ''
        output_format = self.audio_writer.resolve_output_format(output_format)
        output_voice = summarize_voice(parts, voice)
        output_paths = self.audio_writer.build_output_paths(output_voice, len(parts), output_format) if save_outputs else []
        saved_paths = []
        for index, segments in enumerate(parts):
            part_segments = []
            for segment_index, (segment_voice, segment_text) in enumerate(segments):
                audio_tensor, part_ps = self._generate_audio_for_text(segment_text, segment_voice, speed, use_gpu, pause_seconds)
                if audio_tensor is None:
                    continue
                if not first_ps and part_ps:
                    first_ps = part_ps
                part_segments.append(audio_tensor)
                if pause_tensor is not None and segment_index < len(segments) - 1:
                    part_segments.append(pause_tensor)
            if not part_segments:
                continue
            part_audio = torch.cat(part_segments)
            if save_outputs:
                output_path = output_paths[index]
                try:
                    saved_path = self.audio_writer.save_audio(output_path, part_audio, output_format)
                    saved_paths.append(saved_path)
                except Exception:
                    self.logger.exception('Failed to save output: %s', output_path)
            combined_segments.append(part_audio)
            if pause_tensor is not None and index < len(parts) - 1:
                combined_segments.append(pause_tensor)
        if not combined_segments:
            self.logger.debug('Generate produced no segments')
            self.last_saved_paths = []
            return None, ''
        full_audio = torch.cat(combined_segments)
        if save_outputs:
            self.logger.info('Saved %s file(s) to %s', len(saved_paths), self.audio_writer.output_dir)
            self.logger.debug('Saved files: %s', saved_paths)
        self.last_saved_paths = list(saved_paths)
        self.logger.debug(
            'Generate complete: segments=%s pause_samples=%s audio_samples=%s',
            len(combined_segments),
            pause_samples,
            full_audio.numel()
        )
        return (SAMPLE_RATE, full_audio.numpy()), first_ps

    def generate_all(self, text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, pause_seconds=0.0, normalize_times_enabled=None, normalize_numbers_enabled=None):
        parts = self._prepare_dialogue_parts(text, voice, normalize_times_enabled, normalize_numbers_enabled)
        self.logger.debug(
            'Stream start: text_len=%s parts=%s voice=%s speed=%s use_gpu=%s pause_seconds=%s',
            len(text),
            len(parts),
            voice,
            speed,
            use_gpu,
            pause_seconds
        )
        use_gpu = use_gpu and self.cuda_available
        first = True
        pause_samples = max(0, int(pause_seconds * SAMPLE_RATE))
        pause_audio = torch.zeros(pause_samples).numpy() if pause_samples else None
        segment_index = 0
        model_cpu = None
        if not use_gpu:
            model_cpu = self.model_manager.get_model(False)
        keep_sentences = pause_seconds > 0
        with torch.inference_mode():
            for part_index, segments in enumerate(parts):
                for segment_idx, (segment_voice, segment_text) in enumerate(segments):
                    pipeline = self.model_manager.get_pipeline(segment_voice)
                    pack = self.model_manager.get_voice_pack(segment_voice)
                    sentences = smart_split(segment_text, self.max_chunk_chars, keep_sentences=keep_sentences)
                    for sentence_index, sentence in enumerate(sentences):
                        iterator = iter(pipeline(sentence, segment_voice, speed))
                        current = next(iterator, None)
                        while current is not None:
                            _, ps, _ = current
                            segment_index += 1
                            self.logger.debug('Stream segment=%s tokens=%s', segment_index, len(ps))
                            ref_s = pack[len(ps)-1]
                            try:
                                if use_gpu:
                                    audio = forward_gpu(ps, ref_s, speed)
                                else:
                                    audio = model_cpu(ps, ref_s, speed)
                            except gr.exceptions.Error as e:
                                if use_gpu:
                                    gr.Warning(str(e))
                                    gr.Info('Switching to CPU')
                                    audio = self.model_manager.get_model(False)(ps, ref_s, speed)
                                else:
                                    raise gr.Error(e)
                            audio = audio.detach().cpu()
                            yield SAMPLE_RATE, audio.numpy()
                            if first:
                                first = False
                                yield SAMPLE_RATE, torch.zeros(1).numpy()
                            current = next(iterator, None)
                        if pause_audio is not None:
                            has_next_sentence = sentence_index < len(sentences) - 1
                            has_next_segment = segment_idx < len(segments) - 1
                            has_next_part = part_index < len(parts) - 1
                            if has_next_sentence or has_next_segment or has_next_part:
                                yield SAMPLE_RATE, pause_audio
        self.logger.debug('Stream complete: segments=%s pause_samples=%s', segment_index, pause_samples)

MODEL_MANAGER = ModelManager(CONFIG.repo_id, CUDA_AVAILABLE, logger)
TEXT_NORMALIZER = TextNormalizer(CONFIG.char_limit, CONFIG.normalize_times, CONFIG.normalize_numbers)
AUDIO_WRITER = AudioWriter(CONFIG.output_dir, SAMPLE_RATE, logger)
APP_STATE = KokoroState(
    MODEL_MANAGER,
    TEXT_NORMALIZER,
    AUDIO_WRITER,
    CONFIG.max_chunk_chars,
    CUDA_AVAILABLE,
    logger
)

@spaces.GPU(duration=30)
def forward_gpu(ps, ref_s, speed):
    return APP_STATE.model_manager.get_model(True)(ps, ref_s, speed)

def generate_first(text, voice='af_heart', mix_enabled=False, voice_mix=None, speed=1, use_gpu=CUDA_AVAILABLE, pause_seconds=0.0, output_format='wav', normalize_times_enabled=None, normalize_numbers_enabled=None):
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    return APP_STATE.generate_first(
        text=text,
        voice=voice,
        speed=speed,
        use_gpu=use_gpu,
        pause_seconds=pause_seconds,
        output_format=output_format,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled
    )

# Arena API
def predict(text, voice='af_heart', mix_enabled=False, voice_mix=None, speed=1, normalize_times_enabled=None, normalize_numbers_enabled=None):
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    return APP_STATE.generate_first(
        text=text,
        voice=voice,
        speed=speed,
        use_gpu=False,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
        save_outputs=False
    )[0]

def tokenize_first(text, voice='af_heart', mix_enabled=False, voice_mix=None, speed=1, normalize_times_enabled=None, normalize_numbers_enabled=None):
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    return APP_STATE.tokenize_first(
        text=text,
        voice=voice,
        speed=speed,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled
    )

def generate_all(text, voice='af_heart', mix_enabled=False, voice_mix=None, speed=1, use_gpu=CUDA_AVAILABLE, pause_seconds=0.0, normalize_times_enabled=None, normalize_numbers_enabled=None):
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    yield from APP_STATE.generate_all(
        text=text,
        voice=voice,
        speed=speed,
        use_gpu=use_gpu,
        pause_seconds=pause_seconds,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled
    )

CHOICES = {
'🇺🇸 🚺 Heart ❤️': 'af_heart',
'🇺🇸 🚺 Bella 🔥': 'af_bella',
'🇺🇸 🚺 Nicole 🎧': 'af_nicole',
'🇺🇸 🚺 Aoede': 'af_aoede',
'🇺🇸 🚺 Kore': 'af_kore',
'🇺🇸 🚺 Sarah': 'af_sarah',
'🇺🇸 🚺 Nova': 'af_nova',
'🇺🇸 🚺 Sky': 'af_sky',
'🇺🇸 🚺 Alloy': 'af_alloy',
'🇺🇸 🚺 Jessica': 'af_jessica',
'🇺🇸 🚺 River': 'af_river',
'🇺🇸 🚹 Michael': 'am_michael',
'🇺🇸 🚹 Fenrir': 'am_fenrir',
'🇺🇸 🚹 Puck': 'am_puck',
'🇺🇸 🚹 Echo': 'am_echo',
'🇺🇸 🚹 Eric': 'am_eric',
'🇺🇸 🚹 Liam': 'am_liam',
'🇺🇸 🚹 Onyx': 'am_onyx',
'🇺🇸 🚹 Santa': 'am_santa',
'🇺🇸 🚹 Adam': 'am_adam',
'🇬🇧 🚺 Emma': 'bf_emma',
'🇬🇧 🚺 Isabella': 'bf_isabella',
'🇬🇧 🚺 Alice': 'bf_alice',
'🇬🇧 🚺 Lily': 'bf_lily',
'🇬🇧 🚹 George': 'bm_george',
'🇬🇧 🚹 Fable': 'bm_fable',
'🇬🇧 🚹 Lewis': 'bm_lewis',
'🇬🇧 🚹 Daniel': 'bm_daniel',
}
logger.info('Voices available: %s (lazy loading)', len(CHOICES))

TOKEN_NOTE = '''
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`

💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`

⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`

⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
'''

DIALOGUE_NOTE = '''
Use [voice=af_heart] to switch speakers inside the text.
Mix voices with commas: [voice=af_heart,am_michael].
'''

history_state = None
history_audios = []
clear_history_btn = None

with gr.Blocks(theme=APP_THEME) as generate_tab:
    out_audio = gr.Audio(label='Output Audio', interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button('Generate', variant='primary')
    history_state = gr.State([])
    if CONFIG.history_limit > 0:
        with gr.Accordion('History', open=False):
            clear_history_btn = gr.Button('Clear history', variant='secondary')
            for index in range(1, CONFIG.history_limit + 1):
                history_audios.append(gr.Audio(label=f'History {index}', interactive=False, streaming=False))
    with gr.Accordion('Output Tokens', open=True):
        out_ps = gr.Textbox(interactive=False, show_label=False, info='Tokens used to generate the audio, up to 510 context length.')
        tokenize_btn = gr.Button('Tokenize', variant='secondary')
        gr.Markdown(TOKEN_NOTE)
        predict_btn = gr.Button('Predict', variant='secondary', visible=False)

STREAM_NOTE = ['⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`.']
if CONFIG.char_limit is not None:
    STREAM_NOTE.append(f'✂️ Each stream is capped at {CONFIG.char_limit} characters.')
    STREAM_NOTE.append('🚀 Want more characters? You can [use Kokoro directly](https://huggingface.co/hexgrad/Kokoro-82M#usage) or duplicate this space:')
STREAM_NOTE = '\n\n'.join(STREAM_NOTE)

def toggle_mix_controls(enabled):
    return gr.update(visible=enabled), gr.update(interactive=not enabled)

def update_history(history):
    history = list(history or [])
    saved_paths = getattr(APP_STATE, 'last_saved_paths', []) or []
    if saved_paths:
        for path in reversed(saved_paths):
            if path and os.path.isfile(path):
                history.insert(0, path)
    history = history[:CONFIG.history_limit]
    values = history + [None] * (CONFIG.history_limit - len(history))
    return (history, *values)

def clear_history(history):
    history = list(history or [])
    deleted = 0
    for path in history:
        if not path:
            continue
        try:
            abs_path = os.path.abspath(path)
            if os.path.commonpath([abs_path, CONFIG.output_dir_abs]) != CONFIG.output_dir_abs:
                logger.warning('Skip delete outside output dir: %s', path)
                continue
            if os.path.isfile(abs_path):
                os.remove(abs_path)
                deleted += 1
        except Exception:
            logger.exception('Failed to delete history file: %s', path)
    APP_STATE.last_saved_paths = []
    logger.info('Cleared history: deleted=%s', deleted)
    values = [None] * CONFIG.history_limit
    return ([], *values)

with gr.Blocks(theme=APP_THEME) as stream_tab:
    out_stream = gr.Audio(label='Output Audio Stream', interactive=False, streaming=True, autoplay=True)
    with gr.Row():
        stream_btn = gr.Button('Stream', variant='primary')
        stop_btn = gr.Button('Stop', variant='stop')
    with gr.Accordion('Note', open=True):
        gr.Markdown(STREAM_NOTE)
        gr.DuplicateButton()

API_OPEN = CONFIG.space_id != 'hexgrad/Kokoro-TTS'
API_NAME = None if API_OPEN else False
logger.debug('API_OPEN=%s', API_OPEN)
with gr.Blocks(theme=APP_THEME) as app:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label='Input Text',
                info=f"Up to ~{CONFIG.max_chunk_chars} characters per chunk for Generate, or {'∞' if CONFIG.char_limit is None else CONFIG.char_limit} characters per Stream. Use | to split into separate files. Use [voice=af_heart] to switch speakers."
            )
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Voice', info='Quality and availability vary by language')
                mix_enabled = gr.Checkbox(label='Mix voices', value=False)
            voice_mix = gr.Dropdown(
                list(CHOICES.items()),
                value=[],
                multiselect=True,
                label='Voice mix',
                info='Select multiple voices to average',
                visible=False
            )
            with gr.Row():
                use_gpu = gr.Dropdown(
                    [('ZeroGPU 🚀', True), ('CPU 🐌', False)],
                    value=CUDA_AVAILABLE,
                    label='Hardware',
                    info='GPU is usually faster, but has a usage quota',
                    interactive=CUDA_AVAILABLE
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
            pause_between = gr.Slider(
                minimum=0,
                maximum=2,
                value=0,
                step=0.1,
                label='Pause between sentences (s)',
                info='Applies to Generate and Stream output'
            )
            output_format = gr.Dropdown(
                OUTPUT_FORMATS,
                value=DEFAULT_OUTPUT_FORMAT,
                label='Output format',
                info='Applies to saved files in History/outputs (mp3/ogg require ffmpeg)'
            )
            with gr.Accordion('Text normalization', open=False):
                normalize_times_toggle = gr.Checkbox(
                    label='Normalize times (12:30 -> twelve thirty)',
                    value=CONFIG.normalize_times
                )
                normalize_numbers_toggle = gr.Checkbox(
                    label='Normalize numbers (0-9999, decimals, %, ordinals)',
                    value=CONFIG.normalize_numbers
                )
            with gr.Accordion('Dialog tags', open=False):
                gr.Markdown(DIALOGUE_NOTE)
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ['Generate', 'Stream'])
    mix_enabled.change(fn=toggle_mix_controls, inputs=[mix_enabled], outputs=[voice_mix, voice])
    generate_event = generate_btn.click(
        fn=generate_first,
        inputs=[text, voice, mix_enabled, voice_mix, speed, use_gpu, pause_between, output_format, normalize_times_toggle, normalize_numbers_toggle],
        outputs=[out_audio, out_ps],
        api_name=API_NAME
    )
    generate_event.then(
        fn=update_history,
        inputs=[history_state],
        outputs=[history_state] + history_audios
    )
    if clear_history_btn is not None:
        clear_history_btn.click(fn=clear_history, inputs=[history_state], outputs=[history_state] + history_audios)
    tokenize_btn.click(
        fn=tokenize_first,
        inputs=[text, voice, mix_enabled, voice_mix, speed, normalize_times_toggle, normalize_numbers_toggle],
        outputs=[out_ps],
        api_name=API_NAME
    )
    stream_event = stream_btn.click(
        fn=generate_all,
        inputs=[text, voice, mix_enabled, voice_mix, speed, use_gpu, pause_between, normalize_times_toggle, normalize_numbers_toggle],
        outputs=[out_stream],
        api_name=API_NAME
    )
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(
        fn=predict,
        inputs=[text, voice, mix_enabled, voice_mix, speed, normalize_times_toggle, normalize_numbers_toggle],
        outputs=[out_audio],
        api_name=API_NAME
    )

logger.debug('UI wiring complete')

if __name__ == '__main__':
    logger.info('Launching Gradio app')
    queue_kwargs = {'api_open': API_OPEN}
    if CONFIG.default_concurrency_limit is not None:
        queue_kwargs['default_concurrency_limit'] = CONFIG.default_concurrency_limit
    app.queue(**queue_kwargs).launch(show_api=API_OPEN, ssr_mode=True)
