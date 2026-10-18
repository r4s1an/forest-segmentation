import numpy as np
from rasterio.transform import from_origin

from forest_seg.metrics import scores
from forest_seg.tiling import make_tiles, split_blocks, starts
from forest_seg.vectorize import mask_to_polygons

H, W = 2276, 2232


def _gap(a, b):
    rows = max(a[0] - b[1], b[0] - a[1])
    cols = max(a[2] - b[3], b[2] - a[3])
    return max(rows, cols)


def test_starts_cover_far_edge():
    s = starts(600, 256, 128)
    assert s[0] == 0 and s[-1] == 600 - 256


def test_blocks_are_separated_by_buffer():
    b = split_blocks(H, W)
    for k1, k2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert _gap(b[k1], b[k2]) >= 256, (k1, k2)


def test_blocks_stay_inside_scene():
    for r0, r1, c0, c1 in split_blocks(H, W).values():
        assert 0 <= r0 < r1 <= H and 0 <= c0 < c1 <= W


def test_tiles_stay_inside_their_block():
    X = np.ones((4, H, W), "float32")
    X[0] = np.arange(W)[None, :] + 1   # channel 0 encodes the column index
    X[1] = np.arange(H)[:, None] + 1   # channel 1 encodes the row index
    Y = np.zeros((H, W), "uint8")
    blocks = split_blocks(H, W)
    for name, (r0, r1, c0, c1) in blocks.items():
        Xt, Yt = make_tiles(X, Y, blocks[name])
        assert len(Xt) > 0 and Xt.shape[1:] == (4, 256, 256) and Yt.shape[1:] == (256, 256)
        assert Xt[:, 0].min() > c0 and Xt[:, 0].max() <= c1, name
        assert Xt[:, 1].min() > r0 and Xt[:, 1].max() <= r1, name


def test_nodata_tiles_are_skipped():
    X = np.ones((4, 256, 256), "float32")
    X[:, :100, :] = 0   # about 39% nodata
    Xt, _ = make_tiles(X, np.zeros((256, 256), "uint8"), (0, 256, 0, 256))
    assert len(Xt) == 0


def test_iou_values():
    s = scores(tp=50, fp=10, fn=10, tn=30)
    assert abs(s["iou_forest"] - 50 / 70) < 1e-9
    assert abs(s["iou_nonforest"] - 30 / 50) < 1e-9


def test_polygon_area_in_hectares():
    t = from_origin(0, 1000, 10, 10)             # 10 m pixels
    big = np.ones((100, 100), "uint8")           # 1 km x 1 km = 100 ha
    assert abs(mask_to_polygons(big, t, "EPSG:32639").ha.sum() - 100) < 1e-6
    speck = np.zeros((100, 100), "uint8")
    speck[:5, :5] = 1                            # 0.25 ha
    assert len(mask_to_polygons(speck, t, "EPSG:32639")) == 0
