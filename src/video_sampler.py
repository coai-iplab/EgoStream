import cv2
import collections
import os
from abc import ABC, abstractmethod

class VideoSampler(ABC):
    def __init__(self, video_path):
        self.video_path = video_path

    @abstractmethod
    def __iter__(self):
        pass

class UniformFrameSampler(VideoSampler):
    """Returns ONE frame at a time (the one you use now)."""
    def __init__(self, video_path, target_fps=1, max_frames=10000):
        super().__init__(video_path)
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.query_time=-1

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): return
        
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        skip = int(max(native_fps, 30)) / self.target_fps
        
        count, yielded = 0, 0
        while cap.isOpened() and yielded < self.max_frames:
            ret, frame = cap.read()
            if not ret: break
            if count % int(skip) == 0:
                yield frame # Returns a single numpy array
                yielded += 1
            count += 1
        cap.release()



class SlidingWindowSampler(VideoSampler):
    def __init__(self, video_path, window_size=3, stride=1, target_fps=1,
                 max_frames=1000, until_second=None):
        super().__init__(video_path)
        self.window_size = window_size
        self.stride = stride
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.until_second = until_second

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps <= 0:
                native_fps = 30.0

            skip = max(1, int(max(native_fps, 30.0) / self.target_fps))

            buffer = collections.deque(maxlen=self.window_size)
            tail_buffer = []  # new frames since the last yield

            raw_frame_count = 0
            frames_added_to_buffer = 0
            yielded_batches = 0
            stopped_by_until = False

            while cap.isOpened() and yielded_batches < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if raw_frame_count % skip == 0:
                    ts_sec = raw_frame_count / native_fps

                    if self.until_second is not None and ts_sec > self.until_second:
                        stopped_by_until = True
                        break

                    buffer.append(frame)
                    tail_buffer.append(frame)
                    frames_added_to_buffer += 1

                    if len(buffer) == self.window_size:
                        steps_since_full = frames_added_to_buffer - self.window_size

                        if steps_since_full % self.stride == 0:
                            yield list(buffer)
                            yielded_batches += 1
                            tail_buffer = []  # reset: from here on only the new tail is accumulated

                raw_frame_count += 1

            # final flush: only the frames not yet emitted
            if len(tail_buffer) > 0 and yielded_batches < self.max_frames:
                yield tail_buffer

        finally:
            cap.release()


class IncrementalSlidingWindowSampler:
    """
    Incremental stateful sampler.

    Difference with respect to SlidingWindowSampler:
    - it never restarts from zero
    - it keeps the VideoCapture open
    - it keeps buffer, tail_buffer and internal counters
    - it allows advancing progressively up to increasing timestamps

    Typical usage:
        sampler = IncrementalSlidingWindowSampler(...)
        committed_batches, pending_tail = sampler.advance_until(721.0)
        committed_batches, pending_tail = sampler.advance_until(734.4)
        ...
        sampler.close()

    Semantics:
    - committed_batches: complete "stable" batches, to be appended to the master cache
    - pending_tail: final tail not yet consolidated, to be used only on the question cache
    """

    def __init__(
        self,
        video_path,
        window_size=3,
        stride=1,
        target_fps=1,
        max_frames=1000,
        pre_resize=None,
    ):
        self.video_path = video_path
        self.window_size = window_size
        self.stride = stride
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.pre_resize = tuple(pre_resize) if pre_resize else None

        self.decord_sampling = str(os.environ.get("EGOSTREAM_DECORD_SAMPLING", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.pyav_sampling = str(os.environ.get("EGOSTREAM_PYAV_SAMPLING", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.vr = None
        self.av_container = None
        self.av_frame_iter = None
        self.total_frames = 0
        self.cap = None

        if self.pyav_sampling:
            try:
                import av

                self.av_container = av.open(self.video_path)
                stream = self.av_container.streams.video[0]
                self.av_frame_iter = self.av_container.decode(stream)
                self.opened = True
                native_fps = float(stream.average_rate or stream.base_rate or 0.0)
                self.total_frames = int(stream.frames or 0)
            except Exception:
                self.av_container = None
                self.av_frame_iter = None
                self.opened = False
                native_fps = 0.0
        elif self.decord_sampling:
            try:
                from decord import VideoReader, cpu

                self.vr = VideoReader(self.video_path, ctx=cpu(0))
                self.opened = len(self.vr) > 0
                self.total_frames = len(self.vr)
                native_fps = float(self.vr.get_avg_fps() or 0.0) if self.opened else 0.0
            except Exception:
                self.vr = None
                self.opened = False
                native_fps = 0.0
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            self.opened = self.cap.isOpened()
            native_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.opened else 0.0

        if self.opened:
            if native_fps <= 0:
                native_fps = 30.0
            self.native_fps = native_fps
        else:
            self.native_fps = 30.0

        self.skip = max(1, int(max(self.native_fps, 30.0) / self.target_fps))
        self.seek_sampling = str(os.environ.get("EGOSTREAM_SEEK_SAMPLING", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.next_sample_index = 0

        self.buffer = collections.deque(maxlen=self.window_size)
        self.tail_buffer = []

        self.raw_frame_count = 0
        self.frames_added_to_buffer = 0
        self.yielded_batches = 0
        self.finished = not self.opened
        self.last_until_second = None

    def _store_frame(self, frame):
        if self.pre_resize is not None:
            frame = cv2.resize(frame, self.pre_resize, interpolation=cv2.INTER_AREA)

        self.buffer.append(frame)
        self.tail_buffer.append(frame)
        self.frames_added_to_buffer += 1

    def is_open(self):
        return self.opened and not self.finished

    def advance_until(self, until_second):
        """
        Advances the internal state up to until_second.

        Returns:
            committed_batches: list[list[frame]]
            pending_tail: list[frame]

        Important notes:
        - until_second must be monotonically non-decreasing
        - pending_tail must NOT be consolidated into the master cache
        - pending_tail must only be appended to a temporary copy of the cache
        """
        if until_second is None:
            raise ValueError("until_second must not be None")

        until_second = float(until_second)

        if self.last_until_second is not None and until_second < self.last_until_second:
            raise ValueError(
                f"advance_until requires non-decreasing timestamps: "
                f"got {until_second} after {self.last_until_second}"
            )

        self.last_until_second = until_second

        committed_batches = []

        if self.finished:
            return committed_batches, list(self.tail_buffer)

        if self.pyav_sampling and self.av_frame_iter is not None:
            while not self.finished and self.yielded_batches < self.max_frames:
                try:
                    frame_obj = next(self.av_frame_iter)
                except StopIteration:
                    self.finished = True
                    break

                if self.raw_frame_count % self.skip == 0:
                    ts_sec = self.raw_frame_count / self.native_fps

                    if ts_sec > until_second:
                        break

                    frame = frame_obj.to_ndarray(format="bgr24")
                    self._store_frame(frame)

                    if len(self.buffer) == self.window_size:
                        steps_since_full = self.frames_added_to_buffer - self.window_size

                        if steps_since_full % self.stride == 0:
                            committed_batches.append(list(self.buffer))
                            self.yielded_batches += 1
                            self.tail_buffer = []

                self.raw_frame_count += 1

            return committed_batches, list(self.tail_buffer)

        if self.decord_sampling and self.vr is not None:
            while not self.finished and self.yielded_batches < self.max_frames:
                raw_frame_idx = int(round(self.next_sample_index * self.skip))
                ts_sec = raw_frame_idx / self.native_fps

                if ts_sec > until_second:
                    break
                if raw_frame_idx >= self.total_frames:
                    self.finished = True
                    break

                frame_rgb = self.vr[raw_frame_idx].asnumpy()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                self._store_frame(frame)
                self.raw_frame_count = raw_frame_idx + 1
                self.next_sample_index += 1

                if len(self.buffer) == self.window_size:
                    steps_since_full = self.frames_added_to_buffer - self.window_size

                    if steps_since_full % self.stride == 0:
                        committed_batches.append(list(self.buffer))
                        self.yielded_batches += 1
                        self.tail_buffer = []

            return committed_batches, list(self.tail_buffer)

        if self.seek_sampling:
            while not self.finished and self.yielded_batches < self.max_frames:
                raw_frame_idx = int(round(self.next_sample_index * self.skip))
                ts_sec = raw_frame_idx / self.native_fps

                if ts_sec > until_second:
                    break

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_idx)
                ret, frame = self.cap.read()
                if not ret:
                    self.finished = True
                    break

                self._store_frame(frame)
                self.raw_frame_count = raw_frame_idx + 1
                self.next_sample_index += 1

                if len(self.buffer) == self.window_size:
                    steps_since_full = self.frames_added_to_buffer - self.window_size

                    if steps_since_full % self.stride == 0:
                        committed_batches.append(list(self.buffer))
                        self.yielded_batches += 1
                        self.tail_buffer = []

            return committed_batches, list(self.tail_buffer)

        while self.cap.isOpened() and not self.finished and self.yielded_batches < self.max_frames:
            ret, frame = self.cap.read()
            if not ret:
                self.finished = True
                break

            if self.raw_frame_count % self.skip == 0:
                ts_sec = self.raw_frame_count / self.native_fps

                if ts_sec > until_second:
                    break

                self._store_frame(frame)

                if len(self.buffer) == self.window_size:
                    steps_since_full = self.frames_added_to_buffer - self.window_size

                    if steps_since_full % self.stride == 0:
                        committed_batches.append(list(self.buffer))
                        self.yielded_batches += 1
                        self.tail_buffer = []

            self.raw_frame_count += 1

        pending_tail = list(self.tail_buffer)
        return committed_batches, pending_tail

    def finish(self):
        """
        Consumes the rest of the video until the end and returns the last batches/tail.
        Only useful if you should ever need a final query at EOF.
        """
        if self.finished:
            return [], list(self.tail_buffer)

        committed_batches = []

        while self.cap.isOpened() and not self.finished and self.yielded_batches < self.max_frames:
            ret, frame = self.cap.read()
            if not ret:
                self.finished = True
                break

            if self.raw_frame_count % self.skip == 0:
                self.buffer.append(frame)
                self.tail_buffer.append(frame)
                self.frames_added_to_buffer += 1

                if len(self.buffer) == self.window_size:
                    steps_since_full = self.frames_added_to_buffer - self.window_size
                    if steps_since_full % self.stride == 0:
                        committed_batches.append(list(self.buffer))
                        self.yielded_batches += 1
                        self.tail_buffer = []

            self.raw_frame_count += 1

        return committed_batches, list(self.tail_buffer)

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.vr = None
        if self.av_container is not None:
            self.av_container.close()
        self.av_container = None
        self.av_frame_iter = None
        self.finished = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
