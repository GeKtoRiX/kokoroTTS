"""Audio backend wrappers used by desktop UI."""

from __future__ import annotations

import os
import sys

try:
    import vlc as _vlc
except Exception:  # pragma: no cover - dependency optional at import time
    _vlc = None


class VlcAudioBackend:
    """Thin libVLC wrapper for audio-only playback."""

    def __init__(self, *, vlc_module=None, platform_name: str | None = None) -> None:
        self._vlc = vlc_module if vlc_module is not None else _vlc
        if self._vlc is None:
            raise RuntimeError("python-vlc is not available")
        platform_value = platform_name if platform_name is not None else sys.platform
        args = ["--no-xlib"] if str(platform_value).startswith("linux") else []
        self.instance = self._vlc.Instance(args)
        self.player = self.instance.media_player_new()
        self.media = None

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if self.media is not None:
            try:
                self.media.release()
            except Exception:
                pass
            self.media = None
        media = self.instance.media_new(os.path.abspath(path))
        self.player.set_media(media)
        self.media = media

    def play(self) -> None:
        rc = int(self.player.play())
        if rc == -1:
            raise RuntimeError("VLC failed to start playback.")

    def pause_toggle(self) -> None:
        self.player.pause()

    def set_pause(self, on: bool) -> None:
        self.player.set_pause(1 if on else 0)

    def stop(self) -> None:
        self.player.stop()

    def is_playing(self) -> bool:
        return bool(self.player.is_playing())

    def set_volume(self, vol_0_100: int) -> None:
        self.player.audio_set_volume(max(0, min(100, int(vol_0_100))))

    def get_time_ms(self) -> int:
        return int(self.player.get_time() or 0)

    def get_length_ms(self) -> int:
        return int(self.player.get_length() or 0)

    def set_time_ms(self, ms: int) -> None:
        self.player.set_time(int(ms))

    def get_state(self):
        return self.player.get_state()

    def release(self) -> None:
        try:
            self.player.stop()
        except Exception:
            pass
        if self.media is not None:
            try:
                self.media.release()
            except Exception:
                pass
            self.media = None
        try:
            self.player.release()
        except Exception:
            pass
        try:
            self.instance.release()
        except Exception:
            pass
