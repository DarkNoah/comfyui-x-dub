import numpy as np
import onnxruntime as ort

from .detection import inference_detector
from .estimation import inference_pose


class ONNXWholebodyDetector:
    def __init__(self, detector_path, pose_path, device="cuda"):
        if str(device).startswith("cuda") and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        available = set(ort.get_available_providers())
        if str(device).startswith("cuda") and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.detector = ort.InferenceSession(str(detector_path), providers=providers)
        self.pose = ort.InferenceSession(str(pose_path), providers=providers)
        self.providers = providers

    def __call__(self, image_np_hwc, box_ext=None):
        bboxes = np.asarray(box_ext, dtype=np.float32) if box_ext is not None else inference_detector(
            self.detector,
            image_np_hwc,
            model_type="ONNX",
            nms_threshold=0.7,
            score_threshold=0.1,
        )
        if bboxes is None or len(bboxes) == 0:
            bboxes = np.array([[0, 0, image_np_hwc.shape[1], image_np_hwc.shape[0]]], dtype=np.float32)
        elif len(bboxes) > 1:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            bboxes = bboxes[[int(np.argmax(areas))]]

        keypoints, scores = inference_pose(self.pose, bboxes, image_np_hwc, model_type="ONNX")
        if keypoints.shape[0] == 0:
            raise RuntimeError("ONNX DWPose did not detect a person")

        keypoints_info = np.concatenate((keypoints[:1], scores[:1, :, None]), axis=-1)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2] = np.minimum(keypoints_info[:, 5, 2], keypoints_info[:, 6, 2])
        keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        keypoints_info[:, openpose_idx] = keypoints_info[:, mmpose_idx]

        return keypoints_info[..., :2], keypoints_info[..., 2], bboxes
