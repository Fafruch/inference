import argparse
from threading import Thread
from typing import Optional

import cv2
import supervision as sv

from inference.core.interfaces.camera.entities import StatusUpdate
from inference.core.interfaces.camera.utils import get_video_frames_generator
from inference.core.interfaces.camera.video_source import VideoSource, BufferFillingStrategy
from inference.core.utils.preprocess import letterbox_image

STOP = False


def main(stream_uri: str, max_fps: Optional[int] = None) -> None:
    global STOP
    camera = VideoSource.init(video_reference=stream_uri, buffer_filling_strategy=BufferFillingStrategy.DROP_LATEST)
    camera.start()
    control_thread = Thread(target=command_thread, args=(camera,))
    control_thread.start()
    fps_monitor = sv.FPSMonitor()
    for _, _, frame in get_video_frames_generator(stream=camera, max_fps=max_fps):
        fps_monitor.tick()
        fps_value = fps_monitor()
        resized_frame = letterbox_image(frame, desired_size=(1280, 720))
        resized_frame = cv2.putText(
            resized_frame, f"FPS: {round(fps_value, 2)}", (9, 709),
            cv2.FONT_HERSHEY_SIMPLEX, 1.02, (0, 0, 0), 5
        )
        resized_frame = cv2.putText(
            resized_frame, f"FPS: {round(fps_value, 2)}", (10, 710),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        cv2.imshow("Stream", resized_frame)
        _ = cv2.waitKey(1)
    STOP = True
    print("DONE")
    cv2.destroyAllWindows()
    control_thread.join()


def command_thread(camera: VideoSource) -> None:
    global STOP
    while not STOP:
        key = input()
        if key == "q":
            continue
        elif key == "i":
            print(camera.describe_source())
        elif key == "t":
            camera.terminate()
            STOP = True
        elif key == "p":
            camera.pause()
        elif key == "m":
            camera.mute()
        elif key == "r":
            camera.resume()
        elif key == "re":
            camera.restart()


def dump_status_update(status_update: StatusUpdate) -> None:
    print(status_update)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Camera demo parser")
    parser.add_argument(
        "--stream_uri", help="URI of video stream", type=str, required=True
    )
    parser.add_argument(
        "--max_fps", help="Limit FPS", type=int, required=False, default=None
    )
    args = parser.parse_args()
    main(stream_uri=args.stream_uri, max_fps=args.max_fps)
