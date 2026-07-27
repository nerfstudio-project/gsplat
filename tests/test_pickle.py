"""Tests for pickle/torch.save serialization of gsplat C++ custom classes.

Verifies that UnscentedTransformParameters, FThetaCameraDistortionParameters,
and BivariateWindshieldModelParameters round-trip through pickle and torch.save.
"""

import io
import pickle

import pytest
import torch

from gsplat.cuda._wrapper import (
    BivariateWindshieldModelParameters,
    ExternalDistortionReferencePolynomial,
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    UnscentedTransformParameters,
    has_3dgut,
    has_camera_wrappers,
)


@pytest.mark.skipif(not has_3dgut(), reason="3DGUT not built")
class TestUnscentedTransformParametersPickle:
    def _make(self, **kwargs):
        return UnscentedTransformParameters(**kwargs)

    def test_pickle_defaults(self):
        # Assert literal defaults (not obj.X == obj.X) so a broken __setstate__
        # that returns a default-constructed instance is still caught.
        obj = self._make()
        data = pickle.dumps(obj)
        restored = pickle.loads(data)
        assert restored.alpha == pytest.approx(0.1)
        assert restored.beta == pytest.approx(2.0)
        assert restored.kappa == pytest.approx(0.0)
        assert restored.in_image_margin_factor == pytest.approx(0.1)
        assert restored.require_all_sigma_points_valid is False

    def test_pickle_custom_values(self):
        obj = self._make(
            alpha=0.5,
            beta=3.0,
            kappa=1.0,
            in_image_margin_factor=0.2,
            require_all_sigma_points_valid=True,
        )
        data = pickle.dumps(obj)
        restored = pickle.loads(data)
        assert restored.alpha == pytest.approx(0.5)
        assert restored.beta == pytest.approx(3.0)
        assert restored.kappa == pytest.approx(1.0)
        assert restored.in_image_margin_factor == pytest.approx(0.2)
        assert restored.require_all_sigma_points_valid is True

    def test_torch_save_load(self):
        obj = self._make(alpha=0.3, kappa=2.0)
        buf = io.BytesIO()
        torch.save(obj, buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=False)
        assert restored.alpha == pytest.approx(0.3)
        assert restored.kappa == pytest.approx(2.0)

    def test_torch_save_in_dict(self):
        """Verify the class survives torch.save as part of a larger dict."""
        obj = self._make(beta=5.0)
        payload = {"ut_params": obj, "other": torch.tensor([1.0, 2.0])}
        buf = io.BytesIO()
        torch.save(payload, buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=False)
        assert restored["ut_params"].beta == pytest.approx(5.0)
        assert torch.equal(restored["other"], payload["other"])


@pytest.mark.skipif(not has_camera_wrappers(), reason="Camera wrappers not built")
class TestFThetaCameraDistortionParametersPickle:
    def _make(self, **kwargs):
        defaults = dict(
            reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=[0.0, 8.4335e-03, 2.3174e-06, 0.0, 0.0, 0.0],
            angle_to_pixeldist_poly=[0.0, 118.43, -2.56, 0.0, 0.0, 0.0],
            max_angle=1.5,
            linear_cde=[0.9997, 1.87e-05, 1.77e-05],
        )
        defaults.update(kwargs)
        return FThetaCameraDistortionParameters(**defaults)

    def test_pickle_roundtrip(self):
        obj = self._make()
        data = pickle.dumps(obj)
        restored = pickle.loads(data)
        assert restored.reference_poly == obj.reference_poly
        assert restored.max_angle == pytest.approx(obj.max_angle, rel=1e-5)
        for i in range(6):
            assert restored.pixeldist_to_angle_poly[i] == pytest.approx(
                obj.pixeldist_to_angle_poly[i], abs=1e-5
            )
            assert restored.angle_to_pixeldist_poly[i] == pytest.approx(
                obj.angle_to_pixeldist_poly[i], abs=1e-2
            )
        for i in range(3):
            assert restored.linear_cde[i] == pytest.approx(obj.linear_cde[i], abs=1e-5)

    def test_torch_save_load(self):
        obj = self._make(max_angle=2.0)
        buf = io.BytesIO()
        torch.save(obj, buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=False)
        assert restored.max_angle == pytest.approx(2.0, rel=1e-5)

    def test_torch_save_in_dict(self):
        obj = self._make()
        payload = {"ftheta_coeffs": obj, "width": 1920}
        buf = io.BytesIO()
        torch.save(payload, buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=False)
        assert restored["ftheta_coeffs"].reference_poly == obj.reference_poly
        assert restored["width"] == 1920


@pytest.mark.skipif(not has_camera_wrappers(), reason="Camera wrappers not built")
class TestBivariateWindshieldModelParametersPickle:
    def _make(self):
        obj = BivariateWindshieldModelParameters()
        obj.horizontal_poly = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        obj.vertical_poly = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        obj.horizontal_poly_inverse = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        obj.vertical_poly_inverse = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        obj.reference_poly = ExternalDistortionReferencePolynomial.FORWARD
        return obj

    def test_pickle_roundtrip(self):
        obj = self._make()
        data = pickle.dumps(obj)
        restored = pickle.loads(data)
        assert torch.equal(restored.horizontal_poly, obj.horizontal_poly)
        assert torch.equal(restored.vertical_poly, obj.vertical_poly)
        assert torch.equal(
            restored.horizontal_poly_inverse, obj.horizontal_poly_inverse
        )
        assert torch.equal(restored.vertical_poly_inverse, obj.vertical_poly_inverse)
        assert restored.reference_poly == obj.reference_poly

    def test_torch_save_load(self):
        obj = self._make()
        buf = io.BytesIO()
        torch.save(obj, buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=False)
        assert torch.equal(restored.horizontal_poly, obj.horizontal_poly)
        assert restored.reference_poly == obj.reference_poly

    def test_torch_save_in_dict(self):
        obj = self._make()
        payload = {
            "external_distortion_coeffs": obj,
            "viewmats": torch.eye(4).unsqueeze(0),
        }
        buf = io.BytesIO()
        torch.save(payload, buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=False)
        assert torch.equal(
            restored["external_distortion_coeffs"].horizontal_poly,
            obj.horizontal_poly,
        )
        assert torch.equal(restored["viewmats"], payload["viewmats"])
