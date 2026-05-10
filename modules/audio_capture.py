"""
modules/audio_capture.py
────────────────────────
Captures live microphone audio in fixed-duration segments using SoundDevice.
Each captured segment is a NumPy float32 array at SAMPLE_RATE Hz.
"""

import queue
import threading
import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE, CHANNELS, SEGMENT_DURATION


class AudioCapture:
    """Continuously records audio from the default microphone.

    Usage
    -----
    capture = AudioCapture()
    capture.start()
    segment = capture.get_segment()   # blocks until a segment is ready
    capture.stop()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        segment_duration: float = SEGMENT_DURATION,
    ) -> None:
        self.sample_rate     = sample_rate
        self.channels        = channels
        self.segment_samples = int(sample_rate * segment_duration)

        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._buffer: list[np.ndarray]   = []
        self._stream: sd.InputStream | None = None
        self._running = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Open the input stream and begin buffering."""
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop recording and close the stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_segment(self, timeout: float = 10.0) -> np.ndarray | None:
        """Block until a full segment is available, then return it.

        Returns None if the timeout expires or recording has stopped.
        """
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status,
    ) -> None:
        """SoundDevice callback – accumulates samples into fixed segments."""
        if not self._running:
            return
        self._buffer.append(indata.copy().flatten())

        total = sum(len(b) for b in self._buffer)
        if total >= self.segment_samples:
            combined   = np.concatenate(self._buffer)
            segment    = combined[: self.segment_samples]
            remainder  = combined[self.segment_samples :]
            self._buffer = [remainder] if len(remainder) else []
            self._q.put(segment)
