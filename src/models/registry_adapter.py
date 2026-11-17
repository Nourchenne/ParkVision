"""Portable MLflow inference contract for an existing ParkVision YOLO checkpoint."""

import json

import mlflow.pyfunc


class ParkVisionDetector(mlflow.pyfunc.PythonModel):
    def __init__(self, confidence, nms_iou, image_size):
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.image_size = image_size

    def load_context(self, context):
        from ultralytics import YOLO

        self.detector = YOLO(context.artifacts["checkpoint"])
        names = self.detector.names
        labels = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        if labels != ["space-empty", "space-occupied"]:
            raise ValueError(f"unexpected model classes: {labels}")

    def predict(self, context, model_input, params=None):
        import pandas as pd

        if list(model_input.columns) != ["image_path"] or model_input.empty:
            raise ValueError("input must contain one image_path column and at least one row")
        rows = []
        for path in model_input["image_path"]:
            result = self.detector.predict(
                source=str(path), conf=self.confidence, iou=self.nms_iou,
                imgsz=self.image_size, device="cpu", verbose=False,
            )[0]
            boxes = result.boxes
            classes = [int(v) for v in boxes.cls.tolist()]
            if any(value not in (0, 1) for value in classes):
                raise ValueError("prediction contains an unexpected class")
            detections = [
                {"class_id": class_id, "confidence": float(score),
                 "xyxy": [float(value) for value in coordinates]}
                for class_id, score, coordinates in zip(
                    classes, boxes.conf.tolist(), boxes.xyxy.tolist()
                )
            ]
            rows.append({"image_path": str(path), "empty_count": classes.count(0),
                         "occupied_count": classes.count(1),
                         "detections_json": json.dumps(detections, separators=(",", ":"))})
        return pd.DataFrame(rows)
