import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import pyrealsense2 as rs
import lz4.frame
import threading


class RealSenseCamera:
    def __init__(self, img_shape, fps, serial_number=True, enable_depth=True):
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.init_realsense()

    def init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number is not None:
            config.enable_device(self.serial_number)

        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self.enable_depth:
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def release(self):
        self.pipeline.stop()


class WristDepthFetcher:
    def __init__(self, serial_number, img_shape, fps):
        self.serial_number = serial_number
        self.img_shape = img_shape
        self.fps = fps
        self.depth_frame = None
        self.lock = threading.Lock()
        self.running = True

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.depth, img_shape[1], img_shape[0], rs.format.z16, fps)

        self.pipeline.start(config)
        self.dec_filter = rs.decimation_filter()
        self.dec_filter.set_option(rs.option.filter_magnitude, 1)  # 1 = 해상도 유지

        self.spat_filter = rs.spatial_filter()
        self.spat_filter.set_option(rs.option.filter_magnitude, 2)
        self.spat_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spat_filter.set_option(rs.option.filter_smooth_delta, 20)

        self.temp_filter = rs.temporal_filter()
        self.temp_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temp_filter.set_option(rs.option.filter_smooth_delta, 30)

        self.hole_filter = rs.hole_filling_filter()
        self.hole_filter.set_option(rs.option.holes_fill, 1)

        self.thread = threading.Thread(target=self.update_depth, daemon=True)
        self.thread.start()

    def update_depth(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if depth:
                depth = self.dec_filter.process(depth)
                depth = self.spat_filter.process(depth)
                depth = self.temp_filter.process(depth)
                depth = self.hole_filter.process(depth)

                depth_np = np.asanyarray(depth.get_data())
                depth_np = cv2.resize(depth_np, (self.img_shape[1], self.img_shape[0]), interpolation=cv2.INTER_NEAREST)

                with self.lock:
                    self.depth_frame = depth_np
            # if depth:
            #     with self.lock:
            #         self.depth_frame = np.asanyarray(depth.get_data())

    def get_depth(self):
        with self.lock:
            return self.depth_frame.copy() if self.depth_frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.pipeline.stop()


class ImageServer:
    def __init__(self, config, port=5555, Unit_Test=False):
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'realsense')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])

        self.wrist_camera_type = config.get('wrist_camera_type', 'realsense')
        self.wrist_image_shape = config.get('wrist_camera_image_shape', [480, 640])
        self.wrist_camera_id_numbers = config.get('wrist_camera_id_numbers', [])

        self.port = port
        self.Unit_Test = Unit_Test

        self.head_cameras = [RealSenseCamera(self.head_image_shape, self.fps, sn, enable_depth=False)
                             for sn in self.head_camera_id_numbers]

        self.wrist_cameras = [RealSenseCamera(self.wrist_image_shape, self.fps, sn, enable_depth=False)
                              for sn in self.wrist_camera_id_numbers]

        self.wrist_depth_fetchers = [WristDepthFetcher(sn, self.wrist_image_shape, self.fps)
                                     for sn in self.wrist_camera_id_numbers]

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

    def _close(self):
        for cam in self.head_cameras:
            cam.release()
        for cam in self.wrist_cameras:
            cam.release()
        for fetcher in self.wrist_depth_fetchers:
            fetcher.stop()
        self.socket.close()
        self.context.term()

    def send_process(self):
        try:
            while True:
                head_frames = [cam.get_frame() for cam in self.head_cameras]
                if any(f is None for f in head_frames):
                    print("[Image Server] Error reading head camera.")
                    break
                head_color = cv2.hconcat(head_frames)

                wrist_frames = [cam.get_frame() for cam in self.wrist_cameras]
                wrist_depths = [fetcher.get_depth() for fetcher in self.wrist_depth_fetchers]
                if any(f is None for f in wrist_frames):
                    print("[Image Server] Error reading wrist camera.")
                    break
                wrist_color = cv2.hconcat(wrist_frames)
                wrist_depth = cv2.hconcat(wrist_depths)

                _, wrist_buffer = cv2.imencode('.jpg', wrist_color)
                _, buffer = cv2.imencode('.jpg', head_color)
                wrist_depth_bytes = wrist_depth.tobytes()
                wrist_depth_compressed = lz4.frame.compress(wrist_depth_bytes)
                jpg_bytes = buffer.tobytes()
                wrist_jpg_bytes = wrist_buffer.tobytes()

                message = [jpg_bytes, wrist_jpg_bytes, wrist_depth_compressed]
                self.socket.send_multipart(message)
        except KeyboardInterrupt:
            print("[Image Server] Interrupted.")
        finally:
            self._close()


if __name__ == "__main__":
    config = {
        'fps': 30,
        'head_camera_type': 'realsense',
        'head_camera_image_shape': [480, 848],
        'head_camera_id_numbers': ['339222071291'],
        'wrist_camera_type': 'realsense',
        'wrist_camera_image_shape': [480, 640],
        'wrist_camera_id_numbers': ['922612070565']
    }
    server = ImageServer(config)
    server.send_process()
