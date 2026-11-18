import numpy as np
import unittest
import torch
from bev_osm.diffusion import Generator
from bev_osm.codec import MapAutoencoder
from bev_osm.metrics import Metrics, evaluate
from bev_osm.data import CLASSES
from bev_osm.train import StepBatches


def batch():
    return dict(
        bev=torch.randn(1, 128, 100, 100),
        support=torch.ones(1, 1, 100, 100),
        eval_mask=torch.ones(1, 1, 100, 100),
        target=(torch.rand(1, 6, 100, 100) > 0.8).float(),
        id=["sample-a"],
    )


def test_gradients_and_no_aux(kind):
    torch.set_num_threads(2)
    torch.manual_seed(1)
    net = Generator(kind, MapAutoencoder() if kind == "latent_x0" else None)
    b = batch()
    loss = net.loss(b)
    loss.backward()
    assert torch.isfinite(loss) and net.net.out.weight.grad.abs().sum() > 0
    assert not any("aux" in n for n, _ in net.named_parameters())
    if net.codec is not None:
        assert all(p.grad is None for p in net.codec.parameters())
    net.eval()
    p = net.sample(b, nfe=2)
    b["target"].zero_()
    b["eval_mask"].zero_()
    q = net.sample(b, nfe=2)
    assert p.shape == (1, 6, 100, 100) and torch.equal(p, q)


def test_resume_batches():
    all_batches = list(StepBatches(100, 4, 0, 20, 41))
    resumed = list(StepBatches(100, 4, 9, 20, 41))
    assert all_batches[9:] == resumed


def test_streaming_metrics_equal_original():
    rng = np.random.default_rng(4)
    y = np.zeros((2, 6, 100, 100), bool)
    y[:, :, 20:50, 40:60] = 1
    p = y.astype(float)
    p[1, :, 30:40, 35:45] = 1
    m = np.ones((2, 1, 100, 100))
    a = evaluate(p, y, m, CLASSES)
    metric = Metrics(CLASSES)
    metric.update(p[:1], y[:1], m[:1])
    metric.update(p[1:], y[1:], m[1:])
    b = metric.result()
    for x, z in zip(a, b):
        for k in ("iou", "boundary_f1_0.5m", "boundary_chamfer_m", "cldice"):
            assert (x[k] is None and z[k] is None) or np.isclose(x[k], z[k])


class Contracts(unittest.TestCase):
    def test_models(self):
        for kind in ("x0", "fm", "latent_x0"):
            with self.subTest(kind=kind):
                test_gradients_and_no_aux(kind)

    def test_resume(self):
        test_resume_batches()

    def test_metrics(self):
        test_streaming_metrics_equal_original()


if __name__ == "__main__":
    unittest.main()
