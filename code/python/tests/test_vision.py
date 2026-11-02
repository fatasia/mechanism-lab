import numpy as np
import torch

from phm_core.vision import PatchEncoder, PatchMemoryScore, kcenter_memory


def test_patch_projection_and_maps_keep_float32_and_expected_shape():
    torch.manual_seed(42)
    encoder = PatchEncoder(128).eval()
    assert encoder.projection.dtype == torch.float32
    with torch.inference_mode():
        feature = encoder(torch.zeros(1, 3, 224, 224))
        assert feature.shape == (1, 14, 14, 128)
        memory = feature.flatten(0, 2).numpy()[::20]
        score, heatmap = PatchMemoryScore(encoder, memory)(torch.zeros(1, 3, 224, 224))
    assert heatmap.shape == (1, 14, 14) and torch.isfinite(score).all()


def test_bounded_coreset_keeps_distant_training_points():
    patches = np.array([[0, 0], [0.01, 0], [10, 10], [10.01, 10]], dtype="float32")
    memory = kcenter_memory(patches, 2)
    assert memory.shape == (2, 2)
    np.testing.assert_array_equal(memory[0], patches[0])
    assert np.linalg.norm(memory[1]) > 10
