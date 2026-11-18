import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label


def skeleton(a):
    """Zhang-Suen thinning; zero-padded, 8-neighbour binary skeleton."""
    a = np.pad(a.astype(bool), 1)
    for _ in range(100):
        changed = False
        for phase in (0, 1):
            p = [
                a[:-2, 1:-1],
                a[:-2, 2:],
                a[1:-1, 2:],
                a[2:, 2:],
                a[2:, 1:-1],
                a[2:, :-2],
                a[1:-1, :-2],
                a[:-2, :-2],
            ]
            n = sum(v.astype(np.uint8) for v in p)
            transitions = sum(
                (~p[j] & p[(j + 1) % 8]).astype(np.uint8) for j in range(8)
            )
            cond = (
                (~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6]))
                if phase == 0
                else (~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6]))
            )
            remove = a[1:-1, 1:-1] & (n >= 2) & (n <= 6) & (transitions == 1) & cond
            changed |= bool(remove.any())
            a[1:-1, 1:-1][remove] = False
        if not changed:
            break
    return a[1:-1, 1:-1]


skeletonize = skeleton


def evaluate(prob, gt, visibility, classes):
    decoded = prob >= 0.5
    gt = gt > 0
    mask = visibility[:, 0] > 0
    rows = []
    for k, name in enumerate(classes):
        inter = union = predn = gtn = 0
        distances = []
        bf = {0.5: [], 1.0: []}
        top = []
        components = []
        positive = 0
        for i in range(len(gt)):
            valid = mask[i]
            p = decoded[i, k] & valid
            g = gt[i, k] & valid
            positive += int(g.any())
            inter += int((p & g).sum())
            union += int((p | g).sum())
            predn += int(p.sum())
            gtn += int(g.sum())
            interior = binary_erosion(valid, iterations=2, border_value=0)
            pe = (decoded[i, k] ^ binary_erosion(decoded[i, k])) & interior
            ge = (gt[i, k] ^ binary_erosion(gt[i, k])) & interior
            if ge.any():
                for tolerance in bf:
                    if pe.any():
                        precision = np.mean(
                            distance_transform_edt(~ge)[pe] * 0.5 <= tolerance
                        )
                        recall = np.mean(
                            distance_transform_edt(~pe)[ge] * 0.5 <= tolerance
                        )
                        bf[tolerance].append(
                            float(
                                2 * precision * recall / max(precision + recall, 1e-9)
                            )
                        )
                    else:
                        bf[tolerance].append(0.0)
            if pe.any() and ge.any():
                distances.append(
                    float(
                        0.25
                        * (
                            distance_transform_edt(~ge)[pe].mean()
                            + distance_transform_edt(~pe)[ge].mean()
                        )
                    )
                )
            if name == "road" and (g & interior).any():
                sp = skeletonize(p) & interior
                sg = skeletonize(g) & interior
                precision = float((sp & g).sum() / max(sp.sum(), 1))
                recall = float((sg & p).sum() / max(sg.sum(), 1))
                top.append(2 * precision * recall / max(precision + recall, 1e-9))
                components.append(
                    {
                        "sample": i,
                        "pred": int(label(p & interior)[1]),
                        "gt": int(label(g & interior)[1]),
                    }
                )
        rows.append(
            {
                "class": str(name),
                "positive_samples": positive,
                "gt_pixels": gtn,
                "pred_pixels": predn,
                "iou": inter / union if gtn and union else None,
                "dice": 2 * inter / (predn + gtn) if gtn else None,
                "boundary_f1_0.5m": float(np.mean(bf[0.5])) if bf[0.5] else None,
                "boundary_f1_1m": float(np.mean(bf[1.0])) if bf[1.0] else None,
                "boundary_chamfer_m": float(np.mean(distances)) if distances else None,
                "boundary_chamfer_evaluable_samples": len(distances),
                "cldice": float(np.mean(top)) if top else None,
                "components": components,
            }
        )
    return rows


class Metrics:
    """Streaming exact global IoU and per-image macro boundary metrics."""

    def __init__(self, classes):
        self.classes = classes
        self.rows = [dict(gt=0, pred=0, inter=0, sums={}, counts={}) for _ in classes]
        self.samples = 0

    def update(self, p, y, m):
        for i in range(len(p)):
            rows = evaluate(p[i : i + 1], y[i : i + 1], m[i : i + 1], self.classes)
            self.samples += 1
            for k, r in enumerate(rows):
                v = self.rows[k]
                valid = m[i, 0] > 0
                pp = (p[i, k] >= 0.5) & valid
                gg = (y[i, k] > 0) & valid
                v["gt"] += int(gg.sum())
                v["pred"] += int(pp.sum())
                v["inter"] += int((pp & gg).sum())
                for key in (
                    "boundary_f1_0.5m",
                    "boundary_f1_1m",
                    "boundary_chamfer_m",
                    "cldice",
                ):
                    if r[key] is not None:
                        v["sums"][key] = v["sums"].get(key, 0) + r[key]
                        v["counts"][key] = v["counts"].get(key, 0) + 1

    def result(self):
        rows = []
        for name, v in zip(self.classes, self.rows):
            union = v["gt"] + v["pred"] - v["inter"]
            r = dict(
                class_name=name,
                gt_pixels=v["gt"],
                pred_pixels=v["pred"],
                iou=v["inter"] / union if v["gt"] and union else None,
            )
            for key in (
                "boundary_f1_0.5m",
                "boundary_f1_1m",
                "boundary_chamfer_m",
                "cldice",
            ):
                r[key] = (
                    v["sums"].get(key, 0) / v["counts"][key]
                    if v["counts"].get(key)
                    else None
                )
            rows.append(r)
        return rows


def geometry_score(rows):
    vals = [
        0.5 * r["iou"] + 0.5 * r["boundary_f1_0.5m"]
        for r in rows
        if r["class_name"] in ("road", "sidewalk", "building")
        and r["iou"] is not None
        and r["boundary_f1_0.5m"] is not None
    ]
    return float(np.mean(vals)) if vals else 0.0
