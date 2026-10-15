"""Threaded webcam capture that always serves the newest frame.

``cv2.VideoCapture.read()`` returns frames from a driver-side queue in
order. If the processing loop is slower than the camera, that queue grows
and every frame handed to the tracker is progressively more out of date —
the cursor then lags behind the user's gaze by a delay that creeps upward
the longer the session runs, which no amount of filter tuning can fix.

:class:`FrameGrabber` runs the capture in its own thread, keeps only the
most recent frame, and drops the rest. End-to-end latency becomes one frame
interval regardless of how long inference takes.

Only :func:`open_capture` needs OpenCV, and it imports it lazily, so the
grabber itself can be tested against a fake capture object.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol


class CaptureLike(Protocol):
    """The slice of ``cv2.VideoCapture`` that the grabber relies on."""

    def read(self) -> tuple[bool, Any]: ...
    def release(self) -> None: ...


class FrameGrabber:
    """Serves the most recent frame from a capture device.

    Args:
        capture: Any object with ``read()`` and ``release()``.
        name: Thread name, for debugging.

    The grabber is reentrant-safe for a single consumer: one thread calls
    :meth:`read`, the internal thread produces. Use it as a context manager
    so the thread is always joined and the device always released.
    """

    def __init__(self, capture: CaptureLike, name: str = "fatma-capture") -> None:
        self._capture = capture
        self._name = name
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frame: Any = None
        self._seq = 0
        self._last_read_seq = 0
        self._captured = 0
        self._dropped = 0
        self._stopped = threading.Event()
        self._failed = False
        self._thread: threading.Thread | None = None

    # --- lifecycle -----------------------------------------------------

    def start(self) -> "FrameGrabber":
        """Start the capture thread. Returns self, for chaining."""
        if self._thread is not None:
            raise RuntimeError("FrameGrabber already started")
        self._thread = threading.Thread(
            target=self._run, name=self._name, daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stopped.is_set():
            ok, frame = self._capture.read()
            with self._new_frame:
                if not ok:
                    self._failed = True
                    self._new_frame.notify_all()
                    break
                self._frame = frame
                self._seq += 1
                self._captured += 1
                self._new_frame.notify_all()

    def stop(self) -> None:
        """Stop the thread and release the device. Safe to call twice."""
        self._stopped.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        try:
            self._capture.release()
        except Exception:
            # The device may already be gone; nothing useful to do here and
            # shutdown must not raise over it.
            pass

    def __enter__(self) -> "FrameGrabber":
        if self._thread is None:
            self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # --- consumption ---------------------------------------------------

    def read(self, timeout: float = 2.0) -> tuple[bool, Any]:
        """Wait for a frame newer than the last one returned.

        Blocking on a *new* frame rather than returning the current one
        means the tracker never runs inference twice on the same image.

        Returns:
            ``(True, frame)``, or ``(False, None)`` if the device failed or
            no frame arrived within ``timeout`` seconds.
        """
        with self._new_frame:
            if self._failed and self._seq == self._last_read_seq:
                return False, None
            if self._seq == self._last_read_seq:
                self._new_frame.wait(timeout)
            if self._seq == self._last_read_seq:
                return False, None
            # Every frame between the one last handed over and the newest
            # was captured and skipped. Counting them here is the only
            # place the information exists.
            self._dropped += self._seq - self._last_read_seq - 1
            self._last_read_seq = self._seq
            return True, self._frame

    @property
    def frames_captured(self) -> int:
        """Frames pulled off the device since start."""
        return self._captured

    @property
    def frames_dropped(self) -> int:
        """Frames captured but skipped over — the staleness we avoided.

        A high count is healthy: it means the camera outran the tracker and
        the tracker stayed on current images instead of working through a
        backlog.
        """
        with self._lock:
            return self._dropped

    @property
    def failed(self) -> bool:
        """True once the device stopped returning frames."""
        return self._failed


class DirectCapture:
    """Non-threaded adapter with the same surface as :class:`FrameGrabber`.

    Used when threaded capture is disabled, so the runtime loop does not
    need to branch.
    """

    def __init__(self, capture: CaptureLike) -> None:
        self._capture = capture
        self._captured = 0
        self._failed = False

    def start(self) -> "DirectCapture":
        return self

    def read(self, timeout: float = 2.0) -> tuple[bool, Any]:
        ok, frame = self._capture.read()
        if not ok:
            self._failed = True
            return False, None
        self._captured += 1
        return True, frame

    def stop(self) -> None:
        try:
            self._capture.release()
        except Exception:
            pass

    def __enter__(self) -> "DirectCapture":
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    @property
    def frames_captured(self) -> int:
        return self._captured

    @property
    def frames_dropped(self) -> int:
        return 0

    @property
    def failed(self) -> bool:
        return self._failed


def open_capture(index: int, width: int, height: int, fps: int):
    """Open a webcam and request a capture format.

    Raises:
        RuntimeError: if the device cannot be opened — on macOS this is
            almost always a missing Camera permission rather than a missing
            camera.
    """
    import cv2

    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"cannot open camera {index}. On macOS, grant Camera access to "
            "your terminal in System Settings -> Privacy & Security -> Camera."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    # Ask the driver to keep a single frame queued. Not honoured by every
    # backend (AVFoundation ignores it), which is exactly why FrameGrabber
    # drops stale frames itself rather than relying on this.
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return capture


def negotiated_size(capture) -> tuple[int, int]:
    """What the device actually gave us, which may not be what we asked for.

    Cameras silently fall back to a mode they support. Since capture
    resolution is the dominant term in gaze precision, running at 640x480
    while believing you are at 1280x720 would quietly cost a factor of two
    in accuracy — so the caller reports this rather than assuming.
    """
    import cv2

    return (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
