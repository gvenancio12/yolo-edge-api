"""Testes unitários das métricas usadas na Aula 5."""

import numpy as np

from preprocessing.utils.roboflow_evaluate import Detection, average_precision, iou_xyxy, map50


def test_iou_is_one_for_identical_boxes():
    box = np.array([1, 2, 5, 6], dtype=float)
    assert iou_xyxy(box, box) == 1.0


def test_map50_is_one_for_perfect_prediction():
    truth = {"frame.jpg": {0: [np.array([0, 0, 10, 10], dtype=float)]}}
    predictions = [Detection("frame.jpg", 0, 0.9, np.array([0, 0, 10, 10], dtype=float))]
    assert average_precision(predictions, truth, 0) == 1.0
    assert map50(predictions, truth, 1) == 1.0


def test_map50_is_zero_when_iou_is_below_threshold():
    truth = {"frame.jpg": {0: [np.array([0, 0, 10, 10], dtype=float)]}}
    predictions = [Detection("frame.jpg", 0, 0.9, np.array([20, 20, 30, 30], dtype=float))]
    assert map50(predictions, truth, 1) == 0.0
