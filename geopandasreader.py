import unicodedata
import geopandas as gpd
import numpy as np
import Config
from rasterio.features import rasterize
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


def _normalize(text):
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


class GeoDataLoader:
    def __init__(self, working_crs, class_to_color):
        self.working_crs = working_crs
        self.class_to_color = {
            _normalize(key): value for key, value in class_to_color.items()
        }

    def read_layer(self, filepath, columns=None, bbox=None):
        layer = gpd.read_file(filepath, bbox=bbox)
        if layer.empty:
            raise ValueError(f"Camada vazia: {filepath}")
        if layer.crs is None:
            raise ValueError(f"Shapefile sem CRS: {filepath}")
        if columns is not None:
            keep = [
                column for column in columns if column in layer.columns] + ["geometry"]
            layer = layer[keep]
        return layer.to_crs(self.working_crs)

    def classify_vegetation(self, layer, class_column):
        if class_column not in layer.columns:
            raise ValueError(f"Coluna '{class_column}' não encontrada")

        def map_color(value):
            normalized = _normalize(value)
            for key, color in self.class_to_color.items():
                if key in normalized:
                    return color
            return "white"

        result = layer.copy()
        result["sim_color"] = result[class_column].apply(map_color)
        return result[["sim_color", "geometry"]]

    @staticmethod
    def _square_grid_transform(bounds, width, height):
        """Centraliza os bounds em uma grade width x height de células quadradas."""
        if width <= 0 or height <= 0:
            raise ValueError("width e height devem ser positivos")
        minx, miny, maxx, maxy = map(float, bounds)
        extent_width, extent_height = maxx - minx, maxy - miny
        if extent_width <= 0 or extent_height <= 0:
            raise ValueError(
                "A extensão do shapefile deve ter largura e altura positivas")
        cell_size = max(extent_width / width, extent_height / height)
        grid_width, grid_height = width * cell_size, height * cell_size
        grid_minx = minx - (grid_width - extent_width) / 2
        grid_maxy = maxy + (grid_height - extent_height) / 2
        return from_origin(grid_minx, grid_maxy, cell_size, cell_size), cell_size

    def rasterize_classes(self, layer, width, height):
        """Rasteriza classes e reserva a margem adicionada como barreira."""
        bounds = tuple(layer.total_bounds)
        minx, miny, maxx, maxy = map(float, bounds)
        transform, cell_size = self._square_grid_transform(
            bounds, width, height)
        shapes = (
            (geometry, Config.COLOR_TO_TERRAIN_CODE[color])
            for geometry, color in zip(layer.geometry, layer["sim_color"])
            if geometry is not None and not geometry.is_empty
        )
        terrain_array = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            # Ausência de polígono no shapefile não é vegetação: é área sem dados.
            fill=Config.COLOR_TO_TERRAIN_CODE["white"],
            dtype=np.uint8,
        )

        x_centers = transform.c + (np.arange(width) + 0.5) * cell_size
        y_centers = transform.f - (np.arange(height) + 0.5) * cell_size
        outside_original_bounds = (
            (x_centers[None, :] < minx)
            | (x_centers[None, :] > maxx)
            | (y_centers[:, None] < miny)
            | (y_centers[:, None] > maxy)
        )
        terrain_array[outside_original_bounds] = Config.COLOR_TO_TERRAIN_CODE["white"]
        return terrain_array, transform

    def load_terrain_array(self, filepath, class_column, width, height, bbox=None):
        layer = self.read_layer(filepath, columns=[class_column], bbox=bbox)
        classified_layer = self.classify_vegetation(layer, class_column)
        return self.rasterize_classes(classified_layer, width=width, height=height)

    def load_elevation_array(
        self,
        filepath,
        destination_transform,
        width,
        height,
        allow_missing=False,
    ):

        destination = np.full((height, width), np.nan, dtype=np.float32)
        with rasterio.open(filepath) as source:
            if source.crs is None:
                raise ValueError("O raster de elevação não possui CRS")
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=destination_transform,
                dst_crs=self.working_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        if not allow_missing and not np.isfinite(destination).all():
            raise ValueError(
                "O raster de elevação não cobre toda a extensão da grade")
        return destination
