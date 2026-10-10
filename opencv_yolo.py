"""Small CPU-only YOLO ONNX runner using OpenCV DNN (no PyTorch runtime)."""

from pathlib import Path


class OpenCVDNNYolo:
    def __init__(self, model_path: str, image_size: int = 416):
        import cv2

        if Path(model_path).suffix.lower() != ".onnx":
            raise ValueError(
                "OpenCV DNN requires a YOLO .onnx model; the .pt PyTorch weights are not supported."
            )
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. From the project folder, download the 12.8 MB YOLOv8n ONNX file with: "
                "curl -L -o yolov8n.onnx https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"
            )
        if image_size < 160 or image_size % 32:
            raise ValueError("image size must be at least 160 and divisible by 32")
        self.cv2 = cv2
        self.image_size = image_size
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # OpenCV's DNN backend defaults to CPU; setting DNN_TARGET_CPU emits
        # an unsupported-target warning on OpenCV 5's graph engine.

    def detect(self, frame, confidence=0.4, nms_threshold=0.45):
        try:
            return self._detect_at_size(frame, confidence, nms_threshold, self.image_size)
        except self.cv2.error:
            if self.image_size == 640:
                raise
            # Many exported YOLO ONNX files have fixed 640x640 input dimensions.
            # Retry once at that size instead of terminating the camera loop.
            print(f"[WARN] Model has a fixed input shape; retrying YOLO inference at 640px (requested {self.image_size}px).")
            self.image_size = 640
            return self._detect_at_size(frame, confidence, nms_threshold, self.image_size)

    def _detect_at_size(self, frame, confidence, nms_threshold, size):
        cv2 = self.cv2
        original_height, original_width = frame.shape[:2]
        scale = min(size / original_width, size / original_height)
        resized_width = int(round(original_width * scale))
        resized_height = int(round(original_height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        left = (size - resized_width) // 2
        top = (size - resized_height) // 2
        canvas = cv2.copyMakeBorder(
            resized, top, size - resized_height - top, left,
            size - resized_width - left, cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()

        # YOLOv8 exports [1, 4+classes, candidates]; YOLOv5 exports
        # [1, candidates, 5+classes]. Both formats are supported.
        predictions = output[0]
        if predictions.ndim != 2:
            raise ValueError("unsupported YOLO ONNX output shape")
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] in (5, 6, 84, 85):
            predictions = predictions.T
        feature_count = predictions.shape[1]
        has_objectness = feature_count == 85
        person_index = 5 if has_objectness else 4
        if feature_count <= person_index:
            raise ValueError(f"unsupported YOLO ONNX feature count: {feature_count}")

        boxes = []
        scores = []
        for row in predictions:
            score = float(row[4]) * float(row[person_index]) if has_objectness else float(row[person_index])
            if score < confidence:
                continue
            center_x, center_y, width, height = map(float, row[:4])
            x1 = int(max(0, min(original_width - 1, (center_x - width / 2 - left) / scale)))
            y1 = int(max(0, min(original_height - 1, (center_y - height / 2 - top) / scale)))
            x2 = int(max(x1 + 1, min(original_width, (center_x + width / 2 - left) / scale)))
            y2 = int(max(y1 + 1, min(original_height, (center_y + height / 2 - top) / scale)))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)

        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, nms_threshold)
        if len(indices) == 0:
            return []
        if hasattr(indices, "flatten"):
            indices = indices.flatten()
        else:
            indices = [item[0] if isinstance(item, (tuple, list)) else item for item in indices]
        return [
            (boxes[int(i)][0], boxes[int(i)][1],
             boxes[int(i)][0] + boxes[int(i)][2], boxes[int(i)][1] + boxes[int(i)][3],
             scores[int(i)])
            for i in indices
        ]
