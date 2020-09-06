from typing import Generator, Optional, Tuple

import logging
import os
import subprocess

import cv2
import numpy as np


def video_fps(video_filename: str) -> float:
    """Return a video's frame rate.

    Args:
        video_filename: The name of the video file.

    Returns:
        A float indicating the number of frames per second in the video.

    Raises:
        FileNotFoundError: If the video doesn't exist.
    """
    if not os.path.isfile(video_filename):
        raise FileNotFoundError(f"{video_filename} does not exist.")
    cap = cv2.VideoCapture(video_filename)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps


def video_dimensions(video_filename: str) -> Tuple[int, int]:
    """Return the (height, width) of a video.

    Args:
        video_filename: The name of the video file.

    Returns:
        A tuple containing the height and width of the frames of the video.

    Raises:
        FileNotFoundError: If the video doesn't exist.
    """
    if not os.path.isfile(video_filename):
        raise FileNotFoundError("The video file does not exist.")
    cap = cv2.VideoCapture(video_filename)
    if cap.isOpened():
        success, frame_bgr = cap.read()
    dims = (frame_bgr.shape[0], frame_bgr.shape[1])
    cap.release()
    return dims


def video_to_frames(
    video_filename: str,
    fps: int,
    resize: Optional[Tuple[int, int]] = None,
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """Returns all frames from a video.

    Args:
        video_filename: The name of the video file.
        fps: The fps of the video. If 0, it will be inferred from the metadata
            of the video.
        resize: If the frames are to be resized, this specifies the height and
            width. Set to `None` to keep original resolution.

    Returns:
        An iterator over a tuple of numpy arrays and ints, where each array is
        an RGB uint8 frame and each int is its associated timestamp in seconds.

    Raises:
        ValueError: If fps is greater than the rate of the video.
        FileNotFoundError: If the video doesn't exist.
    """
    # Check that the file exists.
    if not os.path.isfile(video_filename):
        raise FileNotFoundError("The video file does not exist.")

    logging.debug("Loading video from {}".format(video_filename))
    cap = cv2.VideoCapture(video_filename)

    # Determine sampling rate based on fps.
    if fps == 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        keep_frequency = 1
    else:
        if fps > video_fps(video_filename):
            raise ValueError(
                "Cannot sample at a frequency higher than FPS of video."
            )
        keep_frequency = int(float(cap.get(cv2.CAP_PROP_FPS)) / fps)
    logging.debug("Sampling at {} fps.".format(fps))

    try:  # Sample and process frames at given fps.
        counter = 0
        if cap.isOpened():
            while True:
                success, frame_bgr = cap.read()
                if not success:
                    break
                if not counter % keep_frequency:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if resize is not None:
                        frame_rgb = cv2.resize(
                            frame_rgb, tuple(reversed(resize))
                        )
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    counter += 1
                    yield frame_rgb, timestamp
    finally:  # Release resources.
        cap.release()


def video_to_audio(
    video_filename: str,
    sampling_rate: int = 44100
) -> Generator[Tuple[np.ndarray, int], None, None]:
    """Returns all audio chunks from a video.

    Args:
        video_filename: The name of the video file.
        sampling_rate: The number of audio samples per second.

    Returns:
        An iterator over a tuple of numpy arrays and ints, where each array is a
        buffer of floats in the range [-1, 1] and each int is its associated
        timestamp in seconds.

    Raises:
        FileNotFoundError: If the video doesn't exist.
        OSError: If the audio stream is empty.
    """
    # check that the file exists
    if not os.path.isfile(video_filename):
        raise FileNotFoundError("Failed to read the video.")

    logging.debug("Loading audio from {}".format(video_filename))
    audio_reader, audio_writer = os.pipe()
    try:
        args = [
            "ffmpeg",
            "-i",
            video_filename,
            "-f",
            "s16le",  # signed 16-bit little-endian
            "-vn",  # disable video
            "-ar",
            str(sampling_rate),  # set the sampling rate
            "-ac",
            "1",  # mix stereo audio down to mono
            "pipe:{}".format(audio_writer),  # pipe output
        ]
        ffmpeg_proc = subprocess.Popen(
            args,
            pass_fds=(audio_writer,),
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        os.close(audio_writer)
        audio_writer = -1
        buffer_size = 2 * sampling_rate
        reader = os.fdopen(audio_reader, "rb")
        counter = 0
        while True:
            buf = reader.read(buffer_size)
            if not len(buf):
                raise OSError("Audio stream is empty.")
            audio = np.frombuffer(buf, dtype="int16")
            # Convert to [-1, 1] range.
            audio = audio.astype("float64") / 2 ** 15
            yield audio, counter
            counter += 1
            # Done sampling video.
            if len(buf) < buffer_size:
                logging.debug("End of video file reached.")
                break
        ffmpeg_proc.wait()
    finally:  # Release resources.
        os.close(audio_reader)
        if audio_writer >= 0:
            os.close(audio_writer)
