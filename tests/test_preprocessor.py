"""Testes do módulo reutilizável de pré-processamento."""

import numpy as np

from preprocessing.preprocessor import (
    CONFIG_DEFAULT,
    CONFIG_LOW_LIGHT,
    PreprocessConfig,
    Preprocessor,
)


def make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Gera um frame determinístico no formato BGR usado pelo OpenCV."""
    return np.full((h, w, 3), (12, 34, 56), dtype=np.uint8)


class TestPreprocessorOutput:
    def test_output_shape_letterbox(self):
        result = Preprocessor(PreprocessConfig(infer_size=416)).process(make_frame())
        assert result.frame.shape == (416, 416, 3)

    def test_output_dtype_uint8_without_normalization(self):
        result = Preprocessor(PreprocessConfig(normalize=False)).process(make_frame())
        assert result.frame.dtype == np.uint8

    def test_output_dtype_float32_when_normalized(self):
        result = Preprocessor(PreprocessConfig(normalize=True)).process(make_frame())
        assert result.frame.dtype == np.float32
        assert result.frame.max() <= 1.0

    def test_scale_and_original_size_are_recorded(self):
        result = Preprocessor(PreprocessConfig(infer_size=416)).process(make_frame())
        assert result.scale > 0
        assert result.orig_size == (480, 640)

    def test_square_frame_has_no_letterbox_padding(self):
        result = Preprocessor(PreprocessConfig(infer_size=416)).process(
            make_frame(h=416, w=416)
        )
        assert result.pad_w == 0
        assert result.pad_h == 0


class TestBboxAdjustment:
    def test_adjust_removes_letterbox_offset(self):
        processor = Preprocessor(PreprocessConfig(infer_size=416))
        result = processor.process(make_frame())
        boxes_original = processor.adjust_boxes(
            np.array([[10, 50, 100, 200]], dtype=float), result
        )
        assert boxes_original[0, 1] < 50

    def test_adjust_without_letterbox_uses_both_axis_scales(self):
        processor = Preprocessor(PreprocessConfig(infer_size=416, use_letterbox=False))
        result = processor.process(make_frame())
        boxes_original = processor.adjust_boxes(
            np.array([[0, 0, 416, 416]], dtype=float), result
        )
        assert result.scale_x != result.scale_y
        np.testing.assert_allclose(boxes_original[0, 2:], [640, 480])


class TestPreprocessorConfigs:
    def test_presets_keep_the_expected_processing_modes(self):
        assert CONFIG_DEFAULT.infer_size == 320
        assert not CONFIG_DEFAULT.clahe
        assert CONFIG_LOW_LIGHT.clahe
        assert Preprocessor(CONFIG_LOW_LIGHT).process(make_frame()).frame.shape[2] == 3
