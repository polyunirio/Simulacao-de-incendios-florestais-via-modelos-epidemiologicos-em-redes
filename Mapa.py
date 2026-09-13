"""Versão paralela do mapa: relevo interpolado e norte para cima."""

import csv
import os
import random

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from rasterio.fill import fillnodata

import Config


class Mapa:
    def __init__(
        self,
        grid_size_x=Config.GRID_SIZE_X,
        grid_size_y=Config.GRID_SIZE_Y,
        csv_file=None,
        shapefile_info=None,
        use_diagonals=Config.USE_DIAGONAL_CONNECTIONS,
    ):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        if self.grid_size_x <= 0 or self.grid_size_y <= 0:
            raise ValueError("grid_size_x e grid_size_y devem ser positivos")
        self.use_diagonals = use_diagonals
        self.elevation_raster_file = (
            Config.ELEVATION_RASTER_FILE if Config.USE_ELEVATION else None
        )
        self.cell_size_m = float(Config.TARGET_CELL_SIZE)
        self.grid_transform = None
        self.elevation_array = None
        # True nas células com elevação utilizável; após interpolação, todas as
        # células recebem elevação. A origem do dado fica em
        # elevation_interpolated_mask.
        self.elevation_valid_mask = None
        self.elevation_interpolated_mask = None
        self.node_colors = {}
        self._rebuild_graph()
        self.terrain_code_array = np.full(
            (self.grid_size_y, self.grid_size_x),
            Config.COLOR_TO_TERRAIN_CODE["white"],
            dtype=np.uint8,
        )
        if shapefile_info is not None:
            self.load_map_from_shapefile(**shapefile_info)
        elif csv_file and os.path.exists(csv_file):
            self.load_map_from_csv(csv_file)
        else:
            self.create_geographic_pattern()

    def _rebuild_graph(self):
        self.graph = nx.grid_2d_graph(self.grid_size_x, self.grid_size_y)
        if self.use_diagonals:
            diagonal_edges = []
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    if x < self.grid_size_x - 1 and y < self.grid_size_y - 1:
                        diagonal_edges.append(((x, y), (x + 1, y + 1)))
                    if x > 0 and y < self.grid_size_y - 1:
                        diagonal_edges.append(((x, y), (x - 1, y + 1)))
            self.graph.add_edges_from(diagonal_edges)

    def _set_grid_geometry(self, transform):
        if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
            raise ValueError("O raster não pode ter rotação")
        width_m, height_m = abs(float(transform.a)), abs(float(transform.e))
        if not np.isclose(width_m, height_m, rtol=1e-9, atol=1e-6):
            raise ValueError(
                "A grade do shapefile não gerou células quadradas: "
                f"{width_m} m x {height_m} m"
            )
        self.grid_transform = transform
        self.cell_size_m = width_m

    @staticmethod
    def _raster_north_up_to_graph_coordinates(array):
        """Converte linhas raster (norte→sul) para nós da rede (sul→norte).

        Rasterio representa a primeira linha no norte. Já os nós (x, y) são
        desenhados pelo NetworkX com y crescente para cima. Inverter somente o
        eixo das linhas preserva x e faz tanto a figura quanto o relevo usarem
        y crescente em direção ao norte.
        """
        return np.ascontiguousarray(np.asarray(array)[::-1, :])

    def _sync_node_colors_from_array(self):
        self.node_colors = {
            (x, y): Config.TERRAIN_CODE_TO_COLOR.get(
                int(self.terrain_code_array[y, x]), "white"
            )
            for y in range(self.grid_size_y)
            for x in range(self.grid_size_x)
        }

    def _load_elevation(self, geo_loader):
        if self.elevation_raster_file is None:
            self.elevation_array = None
            self.elevation_valid_mask = None
            self.elevation_interpolated_mask = None
            return
        elevation = geo_loader.load_elevation_array(
            self.elevation_raster_file,
            destination_transform=self.grid_transform,
            width=self.grid_size_x,
            height=self.grid_size_y,
            allow_missing=True,
        )
        # O raster chega com a primeira linha ao norte; a rede usa y=0 ao sul.
        elevation = self._raster_north_up_to_graph_coordinates(elevation)
        raster_valid_mask = np.isfinite(elevation)
        missing_count = int((~raster_valid_mask).sum())
        if not raster_valid_mask.any():
            raise ValueError(
                "O raster de elevação não possui nenhuma célula válida na grade")

        if missing_count:
            # Preenche lacunas por interpolação ponderada pela distância a partir
            # das células do MDE que possuem altitude. Assim, a transição do
            # relevo é contínua nas áreas sem dado, em vez de assumir terreno plano.
            elevation = fillnodata(
                image=np.ascontiguousarray(elevation, dtype=np.float32),
                mask=np.ascontiguousarray(raster_valid_mask, dtype=np.uint8),
                max_search_distance=float(np.hypot(*elevation.shape)),
                smoothing_iterations=0,
            )
            if not np.isfinite(elevation).all():
                raise ValueError(
                    "Não foi possível interpolar todas as células sem elevação"
                )
            print(
                f"Elevação interpolada em {missing_count} células sem dado."
            )
        self.elevation_interpolated_mask = ~raster_valid_mask
        self.elevation_valid_mask = np.isfinite(elevation)
        self.elevation_array = elevation

    def create_geographic_pattern(self, seed=None):
        if seed is not None:
            random.seed(seed)
        colors = list(Config.TERRAIN_CODE_TO_COLOR.values())
        weights = [0.1] + [0.9 / (len(colors) - 1)] * (len(colors) - 1)
        for y in range(self.grid_size_y):
            for x in range(self.grid_size_x):
                color = random.choices(colors, weights=weights, k=1)[0]
                self.terrain_code_array[y,
                                        x] = Config.COLOR_TO_TERRAIN_CODE[color]
        self._sync_node_colors_from_array()

    def load_map_from_csv(self, csv_file):
        with open(csv_file, "r", newline="", encoding="utf-8") as file:
            matrix = list(csv.reader(file))
        if not matrix or not matrix[0]:
            raise ValueError("CSV de terreno vazio")
        rows, columns = len(matrix), len(matrix[0])
        if any(len(row) != columns for row in matrix):
            raise ValueError(
                "Todas as linhas do CSV devem ter o mesmo tamanho")
        self.grid_size_x, self.grid_size_y = columns, rows
        self._rebuild_graph()
        self.terrain_code_array = np.full(
            (rows,
             columns), Config.COLOR_TO_TERRAIN_CODE["white"], dtype=np.uint8
        )
        valid_colors = set(Config.COLOR_TO_TERRAIN_CODE)
        for y in range(rows):
            for x in range(columns):
                color = matrix[y][x].strip()
                if color in valid_colors:
                    self.terrain_code_array[y,
                                            x] = Config.COLOR_TO_TERRAIN_CODE[color]
        # CSV também segue a convenção visual usual: primeira linha no topo.
        self.terrain_code_array = self._raster_north_up_to_graph_coordinates(
            self.terrain_code_array
        )
        self._sync_node_colors_from_array()

    def load_map_from_shapefile(
        self,
        filepath,
        class_column,
        bbox=None,
        bounds=None,
        working_crs=Config.WORKING_CRS,
        target_cell_size=Config.TARGET_CELL_SIZE,
        class_to_color=Config.VEGETATION_CLASS_TO_COLOR,
        show_debug=True,
    ):
        from geopandasreader import GeoDataLoader

        if bbox is None:
            bbox = bounds
        loader = GeoDataLoader(working_crs=working_crs,
                               class_to_color=class_to_color)
        terrain_array, transform = loader.load_terrain_array(
            filepath=filepath,
            class_column=class_column,
            width=self.grid_size_x,
            height=self.grid_size_y,
            bbox=bbox,
        )
        if terrain_array.shape != (self.grid_size_y, self.grid_size_x):
            raise RuntimeError(
                "Leitor retornou uma grade com dimensões inesperadas")
        # A grade interna segue as coordenadas da rede: y aumenta para o norte.
        # Sem esta conversão, a exibição e os declives ficam espelhados no eixo Y.
        self.terrain_code_array = self._raster_north_up_to_graph_coordinates(
            terrain_array
        )
        self._set_grid_geometry(transform)
        self._load_elevation(loader)
        self._sync_node_colors_from_array()
        if show_debug:
            print(f"Mapa carregado do shapefile: {filepath}")
            print(
                f"Grade escolhida: {self.grid_size_x} x {self.grid_size_y}; "
                f"células: {self.cell_size_m:.3f} m; "
                f"vizinhança: {'8' if self.use_diagonals else '4'}"
            )

    def save_map_to_csv(self, csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for y in range(self.grid_size_y):
                writer.writerow(
                    Config.TERRAIN_CODE_TO_COLOR.get(
                        int(self.terrain_code_array[self.grid_size_y -
                            1 - y, x]), "white"
                    )
                    for x in range(self.grid_size_x)
                )

    def get_flat_terrain_codes(self):
        return self.terrain_code_array.T.reshape(-1).astype(np.uint8, copy=False)

    def visualize(self, title="Mapa de Terreno"):
        figure, axis = plt.subplots(
            figsize=(Config.DEFAULT_VIEWSIZE_X, Config.DEFAULT_VIEWSIZE_Y)
        )
        positions = {node: node for node in self.graph.nodes()}
        nx.draw_networkx_nodes(
            self.graph,
            positions,
            node_color=[self.node_colors[node] for node in self.graph.nodes()],
            node_size=20 / max(self.grid_size_x, self.grid_size_y),
            ax=axis,
        )
        nx.draw_networkx_edges(self.graph, positions, alpha=0.1, ax=axis)
        legend_elements = [
            Line2D(
                [0], [0], marker="o", color="w",
                label=Config.TERRAIN_CODE_TO_LABEL[code],
                markerfacecolor=color,
                markeredgecolor="lightgray" if color == "white" else "none",
                markersize=10,
            )
            for code, color in Config.TERRAIN_CODE_TO_COLOR.items()
        ]
        axis.set_title(title)
        axis.legend(handles=legend_elements, loc="center left",
                    bbox_to_anchor=(1.0, 0.5))
        # A grade usa índices quadrados. Sem aspecto igual, Matplotlib a ajusta
        # para caber na figura retangular e transforma quadrados em retângulos.
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-0.5, self.grid_size_x - 0.5)
        axis.set_ylim(-0.5, self.grid_size_y - 0.5)
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(Config.TERRAIN_MAP_FILE)
        plt.show()


def build_mapa(grid_size_x=Config.GRID_SIZE_X, grid_size_y=Config.GRID_SIZE_Y):
    if Config.USE_SHAPEFILE:
        return Mapa(
            grid_size_x=grid_size_x,
            grid_size_y=grid_size_y,
            shapefile_info={
                "filepath": Config.VEGETATION_SHAPEFILE,
                "class_column": Config.VEGETATION_CLASS_COLUMN,
                "bbox": None,
            },
        )
    return Mapa(grid_size_x=grid_size_x, grid_size_y=grid_size_y)
