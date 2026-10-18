import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape


def mask_to_polygons(mask, transform, crs, min_ha=1.0):
    """Binary mask -> GeoDataFrame of forest polygons with area in hectares (crs must be metric)."""
    mask = mask.astype("uint8")
    geoms = [shape(g) for g, _ in shapes(mask, mask=mask == 1, transform=transform)]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["ha"] = gdf.area / 1e4
    return gdf[gdf.ha >= min_ha].copy()
