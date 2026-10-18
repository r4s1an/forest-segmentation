import numpy as np


def starts(n, tile, stride):
    """Window start offsets that always cover the far edge."""
    r = list(range(0, n - tile + 1, stride))
    if r[-1] != n - tile:
        r.append(n - tile)
    return r


def split_blocks(H, W, train_x1=1100, val_y1=1000, gap=256):
    """(r0, r1, c0, c1) per block.

    Train = west strip. The east strip is cut into val (top) and test (bottom).
    Every pair of blocks is separated by `gap` pixels to reduce spatial leakage.
    """
    east_x0 = train_x1 + gap
    return {
        "train": (0, H, 0, train_x1),
        "val": (0, val_y1, east_x0, W),
        "test": (val_y1 + gap, H, east_x0, W),
    }


def make_tiles(X, Y, block, tile=256, stride=128, max_nodata=0.05):
    """Cut tiles inside one block. X is (C,H,W), Y is (H,W). Skips tiles with nodata holes."""
    r0, r1, c0, c1 = block
    xs, ys = [], []
    for r in range(r0, r1 - tile + 1, stride):
        for c in range(c0, c1 - tile + 1, stride):
            x, y = X[:, r:r + tile, c:c + tile], Y[r:r + tile, c:c + tile]
            if (x.sum(0) == 0).mean() > max_nodata:
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        return np.empty((0, X.shape[0], tile, tile), X.dtype), np.empty((0, tile, tile), Y.dtype)
    return np.stack(xs), np.stack(ys)
