"""
Small, isolated cross-platform compatibility helpers.

Kept in one place so main.py, audio_output.py, and query_loop.py don't each
reinvent slightly different platform-detection logic. Nothing here is
required for the app to run on any single platform -- it exists purely to
smooth over the specific ways Windows/macOS/Linux diverge in practice.
"""

import sys


def ensure_utf8_console() -> None:
    """Reconfigure stdout/stderr to UTF-8 so non-ASCII log output can't
    crash logging on a Windows console still using a legacy code page
    (cmd.exe / older PowerShell default to something like cp1252, not
    UTF-8). This app can genuinely emit non-ASCII text -- a VLM
    description phrased with an accented word, or a Whisper transcription
    hallucination in another script entirely (observed directly in
    testing: a stray Cyrillic phrase came back from a noisy/echoey audio
    clip) -- and writing that through a non-UTF-8 stream raises
    UnicodeEncodeError from inside the logging module itself, which is a
    confusing way to lose a log line.

    In effect a no-op on macOS/Linux, where UTF-8 is already the default.
    Never raises: worst case this silently does nothing and the platform's
    existing encoding behaves as it did before.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # No reconfigure() on very old Python, stream isn't a real
            # TextIOWrapper (piped/redirected in some environments), or
            # some other platform quirk -- none of that should be fatal
            # for a robustness helper.
            pass


def audio_troubleshooting_hint() -> str:
    """One-line, OS-specific pointer for where to look when PyAudio can't
    open an input or output stream. The failure mode -- and the fix --
    looks different enough on each platform that a single generic message
    ("could not open audio device") isn't actually actionable."""
    import platform

    system = platform.system()
    if system == "Windows":
        return (
            "On Windows: open Settings > System > Sound and confirm the "
            "right input/output device is connected and set as default. "
            "If this is the microphone, also check Settings > Privacy & "
            "security > Microphone > let apps access your microphone."
        )
    if system == "Darwin":
        return (
            "On macOS: check System Settings > Sound, and confirm "
            "PortAudio is installed (brew install portaudio). If this is "
            "the microphone, also check System Settings > Privacy & "
            "Security > Microphone."
        )
    return (
        "On Linux: list devices with `aplay -l` (output) or `arecord -l` "
        "(input) to confirm the hardware is visible, and confirm "
        "PortAudio is installed (sudo apt install portaudio19-dev on "
        "Debian/Ubuntu)."
    )
