import os
import time
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm, to_rgba
import matplotlib.pyplot as plt
import numpy as np
import Config


class Visualizer:
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2

    def __init__(self, terrain_map, simulation_results):
        self.terrain_map = terrain_map
        self.simulation_results = simulation_results
        self.graph = terrain_map.graph
        self.node_list = list(self.graph.nodes())
        self.n_nodes = len(self.node_list)
        self.node_to_idx = {node: index for index,
                            node in enumerate(self.node_list)}
        self.positions = np.asarray(self.node_list, dtype=np.float32)
        self.base_node_colors = np.asarray(
            [to_rgba(terrain_map.node_colors[node])
             for node in self.node_list],
            dtype=np.float32,
        )
        self.infected_prob = None
        self.recovered_prob = None
        self.infected_counts = None
        self.recovered_counts = None
        self.count_simulations = None

    def set_probability_matrices(self, infected_prob, recovered_prob):
        infected = np.asarray(infected_prob)
        recovered = np.asarray(recovered_prob)
        if infected.shape != recovered.shape:
            raise ValueError(
                "Matrizes de infectados e recuperados devem ter a mesma forma")
        if infected.ndim != 2 or infected.shape[1] != self.n_nodes:
            raise ValueError(
                f"Matrizes devem ter forma [passos, {self.n_nodes}]; recebeu {infected.shape}"
            )
        self.infected_prob = infected
        self.recovered_prob = recovered
        self.infected_counts = None
        self.recovered_counts = None
        self.count_simulations = None

    def set_probability_counts(
        self,
        infected_counts,
        recovered_counts,
        num_simulations,
    ):
        """Recebe contagens inteiras por passo, sem criar matrizes float grandes."""
        infected = np.asarray(infected_counts)
        recovered = np.asarray(recovered_counts)
        if infected.shape != recovered.shape:
            raise ValueError(
                "Contagens de infectados e recuperados devem ter a mesma forma")
        if infected.ndim != 2 or infected.shape[1] != self.n_nodes:
            raise ValueError(
                f"Contagens devem ter forma [passos, {self.n_nodes}]; recebeu {infected.shape}"
            )
        if num_simulations <= 0:
            raise ValueError("num_simulations deve ser positivo")
        if np.any(infected < 0) or np.any(recovered < 0):
            raise ValueError("Contagens de estado não podem ser negativas")
        if np.any(infected > num_simulations) or np.any(recovered > num_simulations):
            raise ValueError(
                "Contagens não podem exceder o número de simulações")

        self.infected_counts = infected
        self.recovered_counts = recovered
        self.count_simulations = int(num_simulations)
        self.infected_prob = None
        self.recovered_prob = None

    def _require_probability_matrices(self):
        has_probabilities = (
            self.infected_prob is not None and self.recovered_prob is not None
        )
        has_counts = (
            self.infected_counts is not None and self.recovered_counts is not None
        )
        if not has_probabilities and not has_counts:
            raise ValueError(
                "Defina as matrizes de probabilidade com set_probability_matrices().")

    def _state_masks_for_step(self, step, threshold):
        if self.infected_counts is not None:
            threshold_count = int(np.ceil(threshold * self.count_simulations))
            infected = self.infected_counts[step]
            recovered = self.recovered_counts[step]
            already_burned = infected + recovered
            likely_burning = (already_burned >= threshold_count) & (
                recovered < threshold_count
            )
            likely_burned_out = recovered >= threshold_count
            return likely_burning, likely_burned_out

        infected = self.infected_prob[step]
        recovered = self.recovered_prob[step]
        already_burned = infected + recovered
        likely_burning = (already_burned >= threshold) & (
            recovered < threshold)
        likely_burned_out = recovered >= threshold
        return likely_burning, likely_burned_out

    def _colors_for_step(self, step, threshold):
        colors = self.base_node_colors.copy()
        likely_burning, likely_burned_out = self._state_masks_for_step(
            step, threshold)
        colors[likely_burning] = to_rgba("black")
        colors[likely_burned_out] = to_rgba("gray")
        return colors

    def _burning_probability_values(self, step):
        """Retorna somente os nós com probabilidade positiva de estado I."""
        if self.infected_counts is not None:
            counts = self.infected_counts[step]
            active_indices = np.flatnonzero(counts)
            probabilities = counts[active_indices] / self.count_simulations
            return active_indices, probabilities

        probabilities = self.infected_prob[step]
        active_indices = np.flatnonzero(probabilities > 0.0)
        return active_indices, probabilities[active_indices]

    def _format_time(self, step):
        seconds = step * Config.FIRE_TIME_STEP_S
        if seconds < 60:
            return f"{seconds:.0f} s"

        total_minutes = round(seconds / 60)
        if total_minutes < 60:
            return f"{total_minutes} min"

        hours, minutes = divmod(total_minutes, 60)
        if minutes:
            return f"{hours} h {minutes:02d} min"
        return f"{hours} h"

    def _draw_static_graph(self, axis, node_size, draw_edges):
        edge_artist = None
        if draw_edges:
            segments = self._grid_line_segments()
            edge_artist = LineCollection(
                segments, colors="black", linewidths=0.15, alpha=0.10, zorder=1
            )
            axis.add_collection(edge_artist)
        nodes_artist = axis.scatter(
            self.positions[:, 0], self.positions[:,
                                                 1], s=node_size, marker="o",
            facecolors=self.base_node_colors, edgecolors="none", zorder=2,
        )
        axis.set_aspect("equal")
        # A área do mapa é quadrada; ancora-a à esquerda para reservar a área
        # livre da direita para a legenda, em vez de centralizá-la na figura.
        axis.set_anchor("W")
        axis.set_xlim(-0.5, self.terrain_map.grid_size_x - 0.5)
        axis.set_ylim(-0.5, self.terrain_map.grid_size_y - 0.5)
        axis.axis("off")
        return edge_artist, nodes_artist

    def _grid_line_segments(self):
        """Representa milhões de arestas por linhas contínuas equivalentes.

        Em uma grade regular, as arestas colineares se tocam ponta a ponta. Para
        a imagem estática, uma linha por fileira/coluna/diagonal produz o mesmo
        traçado visual sem construir uma lista Python para cada aresta.
        """
        maximum_x = self.terrain_map.grid_size_x - 1
        maximum_y = self.terrain_map.grid_size_y - 1

        horizontal_y = np.arange(maximum_y + 1, dtype=np.float32)
        horizontal = np.stack(
            (
                np.column_stack((np.zeros_like(horizontal_y), horizontal_y)),
                np.column_stack(
                    (np.full_like(horizontal_y, maximum_x), horizontal_y)),
            ),
            axis=1,
        )
        vertical_x = np.arange(maximum_x + 1, dtype=np.float32)
        vertical = np.stack(
            (
                np.column_stack((vertical_x, np.zeros_like(vertical_x))),
                np.column_stack(
                    (vertical_x, np.full_like(vertical_x, maximum_y))),
            ),
            axis=1,
        )
        segment_sets = [horizontal, vertical]

        if self.terrain_map.use_diagonals:
            # Diagonais com inclinação +1: inícios na borda oeste e sul.
            plus_left_y = np.arange(maximum_y + 1, dtype=np.float32)
            plus_left_length = np.minimum(maximum_x, maximum_y - plus_left_y)
            plus_left = np.stack(
                (
                    np.column_stack((np.zeros_like(plus_left_y), plus_left_y)),
                    np.column_stack(
                        (plus_left_length, plus_left_y + plus_left_length)),
                ),
                axis=1,
            )
            plus_bottom_x = np.arange(1, maximum_x + 1, dtype=np.float32)
            plus_bottom_length = np.minimum(
                maximum_x - plus_bottom_x, maximum_y)
            plus_bottom = np.stack(
                (
                    np.column_stack(
                        (plus_bottom_x, np.zeros_like(plus_bottom_x))),
                    np.column_stack(
                        (plus_bottom_x + plus_bottom_length, plus_bottom_length)),
                ),
                axis=1,
            )

            # Diagonais com inclinação -1: inícios na borda oeste e norte.
            minus_left_y = np.arange(maximum_y + 1, dtype=np.float32)
            minus_left_length = np.minimum(maximum_x, minus_left_y)
            minus_left = np.stack(
                (
                    np.column_stack(
                        (np.zeros_like(minus_left_y), minus_left_y)),
                    np.column_stack(
                        (minus_left_length, minus_left_y - minus_left_length)),
                ),
                axis=1,
            )
            minus_top_x = np.arange(1, maximum_x + 1, dtype=np.float32)
            minus_top_length = np.minimum(maximum_x - minus_top_x, maximum_y)
            minus_top = np.stack(
                (
                    np.column_stack(
                        (minus_top_x, np.full_like(minus_top_x, maximum_y))),
                    np.column_stack(
                        (minus_top_x + minus_top_length, maximum_y - minus_top_length)),
                ),
                axis=1,
            )
            segment_sets.extend(
                (plus_left, plus_bottom, minus_left, minus_top))

        return np.concatenate(segment_sets, axis=0)

    @staticmethod
    def _legend(threshold):
        terrain_legend = [
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color,
                markeredgecolor="lightgray" if color == "white" else "none",
                markersize=8, label=Config.TERRAIN_CODE_TO_LABEL[code],
            )
            for code, color in Config.TERRAIN_CODE_TO_COLOR.items()
        ]
        return [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                       markersize=8, label=f"Queimando (I+R ≥ {threshold:.0%})"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=8, label=f"Recuperado (R ≥ {threshold:.0%})"),
            *terrain_legend,
        ]

    def _rasterize_static_base(self, figure, axis, node_size, draw_edges):
        """Converte arestas e cores de fundo em uma única imagem estática."""
        print("[Animação] Rasterizando nós e arestas estáticos uma única vez...")
        edge_artist, nodes_artist = self._draw_static_graph(
            axis, node_size, draw_edges)
        figure.canvas.draw()
        canvas_width, canvas_height = figure.canvas.get_width_height()
        canvas = np.asarray(figure.canvas.buffer_rgba()).reshape(
            canvas_height, canvas_width, 4
        )
        bbox = axis.get_window_extent(figure.canvas.get_renderer())
        x0, y0, x1, y1 = map(int, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
        # O buffer do canvas começa no topo; o bbox do Matplotlib começa embaixo.
        static_image = canvas[canvas_height -
                              y1:canvas_height - y0, x0:x1].copy()
        if edge_artist is not None:
            edge_artist.remove()
        nodes_artist.remove()
        axis.imshow(
            static_image,
            extent=(-0.5, self.terrain_map.grid_size_x - 0.5,
                    -0.5, self.terrain_map.grid_size_y - 0.5),
            origin="upper",
            interpolation="nearest",
            zorder=0,
        )
        axis.set_aspect("equal")
        axis.set_anchor("W")
        axis.set_xlim(-0.5, self.terrain_map.grid_size_x - 0.5)
        axis.set_ylim(-0.5, self.terrain_map.grid_size_y - 0.5)
        axis.axis("off")

    @staticmethod
    def _print_progress(frame_number, total_frames, started_at, last_printed):
        interval = max(1, total_frames // 20)
        is_final = frame_number + 1 >= total_frames
        if frame_number and frame_number % interval and not is_final:
            return last_printed
        if frame_number == last_printed and not is_final:
            return last_printed
        elapsed = time.perf_counter() - started_at
        completed = frame_number + 1
        mean_time = elapsed / completed
        remaining = max(0.0, mean_time * (total_frames - completed))
        print(
            f"[Animação] quadro {completed}/{total_frames} "
            f"({completed / total_frames:.0%}) | "
            f"decorrido: {elapsed / 60:.1f} min | "
            f"estimativa restante: {remaining / 60:.1f} min"
        )
        return frame_number

    def create_probability_animation(
        self,
        threshold=Config.DEFAULT_PROBABILITY_THRESHOLD,
        save_file=Config.PROBABLE_INFECTION_PATH_FILE,
        frame_step=1,
        fps=10,
        node_size=20,
        draw_edges=True,
    ):
        self._require_probability_matrices()
        if not 0 < threshold <= 1:
            raise ValueError("threshold deve estar entre 0 e 1")
        if frame_step <= 0:
            raise ValueError("frame_step deve ser positivo")

        plt.ioff()
        figure, axis = plt.subplots(
            figsize=(Config.DEFAULT_VIEWSIZE_X, Config.DEFAULT_VIEWSIZE_Y))
        self._rasterize_static_base(figure, axis, node_size, draw_edges)
        title = axis.set_title("")
        axis.legend(handles=self._legend(threshold),
                    loc="center left", bbox_to_anchor=(1.0, 0.5))

        burning_artist = axis.scatter(
            [], [], s=node_size, marker="o", color="black", edgecolors="none", zorder=2)
        burned_artist = axis.scatter(
            [], [], s=node_size, marker="o", color="gray", edgecolors="none", zorder=2)
        state_history = (
            self.infected_counts
            if self.infected_counts is not None
            else self.infected_prob
        )
        max_steps = state_history.shape[0]
        n_simulations = (
            self.count_simulations
            if self.count_simulations is not None
            else len(self.simulation_results)
        )
        frames = list(range(0, max_steps, frame_step))

        def update(frame):
            likely_burning, likely_burned_out = self._state_masks_for_step(
                frame, threshold)
            burning_artist.set_offsets(self.positions[likely_burning])
            burned_artist.set_offsets(self.positions[likely_burned_out])
            title.set_text(
                "Probabilidade de incêndio por limiar\n"
                f"Passo {frame}/{max_steps - 1}  |  tempo: {self._format_time(frame)}\n"
                f"{n_simulations} simulações  |  limiar: {threshold:.0%}"
            )
            return burning_artist, burned_artist, title

        animation_object = animation.FuncAnimation(
            figure, update, frames=frames, interval=1000 / fps,
            blit=False, repeat=False, cache_frame_data=False,
        )
        if save_file:
            # yuv420p é aceito pelo Windows Media Player, Explorador e navegadores.
            # Sem esta opção, o Matplotlib pode gerar yuv444p, que o VS Code abre
            # por possuir decodificador próprio, mas o Windows costuma recusar.
            writer = (
                animation.PillowWriter(fps=fps)
                if save_file.lower().endswith(".gif")
                else animation.FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    extra_args=["-pix_fmt", "yuv420p",
                                "-movflags", "+faststart"],
                )
            )
            started_at = time.perf_counter()
            last_printed = [-1]
            print(
                f"[Animação] Iniciando exportação de {len(frames)} quadros para "
                f"'{os.path.abspath(save_file)}'."
            )

            def progress_callback(frame_number, total_frames):
                last_printed[0] = self._print_progress(
                    frame_number, total_frames, started_at, last_printed[0]
                )

            try:
                animation_object.save(
                    save_file, writer=writer, progress_callback=progress_callback
                )
            finally:
                plt.close(figure)
            print(f"[Animação] Concluída: '{os.path.abspath(save_file)}'")
        else:
            plt.close(figure)
        return animation_object

    def create_burning_probability_animation(
        self,
        save_file=None,
        frame_step=1,
        fps=5,
        node_size=20,
        draw_edges=True,
    ):
        """Anima P(I) em cada instante com uma escala contínua de cores.

        Diferentemente da animação por limiar, esta mostra a probabilidade de
        o nó estar no estado I naquele passo, sem somar os nós recuperados.
        """
        self._require_probability_matrices()
        if frame_step <= 0:
            raise ValueError("frame_step deve ser positivo")
        if save_file is None:
            save_file = (
                f"burning_probability_path_{Config.EXPERIMENT_ID}.mp4"
            )

        plt.ioff()
        figure, axis = plt.subplots(
            figsize=(Config.DEFAULT_VIEWSIZE_X, Config.DEFAULT_VIEWSIZE_Y)
        )
        self._rasterize_static_base(figure, axis, node_size, draw_edges)

        # A base continua visível para orientar a leitura espacial, mas fica
        # suavizada: cores amarelas/alaranjadas do terreno eram confundidas com
        # o antigo mapa YlOrRd de probabilidade.
        axis.imshow(
            np.ones((2, 2, 4), dtype=np.float32),
            extent=(-0.5, self.terrain_map.grid_size_x - 0.5,
                    -0.5, self.terrain_map.grid_size_y - 0.5),
            origin="lower",
            interpolation="nearest",
            alpha=0.10,
            zorder=1,
        )

        # A maior parte das probabilidades P(I) é pequena, sobretudo depois
        # dos primeiros passos. PowerNorm apenas expande visualmente essa faixa
        # baixa; os valores e os rótulos da barra continuam sendo P(I) reais.
        color_map = plt.get_cmap("Reds")
        color_scale = PowerNorm(gamma=0.35, vmin=0.0, vmax=1.0)
        burning_artist = axis.scatter(
            [], [], s=node_size, marker="o", c=np.array([]),
            cmap=color_map, norm=color_scale, edgecolors="none", zorder=2,
        )
        colorbar = figure.colorbar(burning_artist, ax=axis, pad=0.02)
        colorbar.set_label("P(I) -- escala com realce de valores baixos")
        colorbar.set_ticks([0.0, 0.01, 0.05, 0.25, 1.0])
        colorbar.set_ticklabels(["0%", "1%", "5%", "25%", "100%"])
        title = axis.set_title("")

        state_history = (
            self.infected_counts
            if self.infected_counts is not None
            else self.infected_prob
        )
        max_steps = state_history.shape[0]
        n_simulations = (
            self.count_simulations
            if self.count_simulations is not None
            else len(self.simulation_results)
        )
        frames = list(range(0, max_steps, frame_step))

        def update(frame):
            active_indices, probabilities = self._burning_probability_values(
                frame)
            burning_artist.set_offsets(self.positions[active_indices])
            burning_artist.set_array(probabilities)
            title.set_text(
                "Probabilidade de o nó estar queimando (estado I)\n"
                f"Passo {frame}/{max_steps - 1}  |  tempo: {self._format_time(frame)}\n"
                f"{n_simulations} simulações"
            )
            return burning_artist, title

        animation_object = animation.FuncAnimation(
            figure, update, frames=frames, interval=1000 / fps,
            blit=False, repeat=False, cache_frame_data=False,
        )
        if save_file:
            writer = (
                animation.PillowWriter(fps=fps)
                if save_file.lower().endswith(".gif")
                else animation.FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    extra_args=["-pix_fmt", "yuv420p",
                                "-movflags", "+faststart"],
                )
            )
            started_at = time.perf_counter()
            last_printed = [-1]
            print(
                f"[Animação P(I)] Iniciando exportação de {len(frames)} quadros "
                f"para '{os.path.abspath(save_file)}'."
            )

            def progress_callback(frame_number, total_frames):
                last_printed[0] = self._print_progress(
                    frame_number, total_frames, started_at, last_printed[0]
                )

            try:
                animation_object.save(
                    save_file, writer=writer, progress_callback=progress_callback
                )
            finally:
                plt.close(figure)
            print(f"[Animação P(I)] Concluída: '{os.path.abspath(save_file)}'")
        else:
            plt.close(figure)
        return animation_object

    def save_probability_map(
        self,
        step,
        threshold=Config.DEFAULT_PROBABILITY_THRESHOLD,
        filename=None,
        node_size=20,
        draw_edges=True,
    ):
        self._require_probability_matrices()
        state_history = (
            self.infected_counts
            if self.infected_counts is not None
            else self.infected_prob
        )
        if not 0 <= step < state_history.shape[0]:
            raise ValueError("step fora do intervalo disponível")
        figure, axis = plt.subplots(
            figsize=(Config.DEFAULT_VIEWSIZE_X, Config.DEFAULT_VIEWSIZE_Y))
        self._rasterize_static_base(figure, axis, node_size, draw_edges)
        colors = self._colors_for_step(step, threshold)
        likely_burning, likely_burned_out = self._state_masks_for_step(
            step, threshold)
        axis.scatter(self.positions[likely_burning, 0], self.positions[likely_burning, 1],
                     s=node_size, marker="o", color="black", edgecolors="none", zorder=2)
        axis.scatter(self.positions[likely_burned_out, 0], self.positions[likely_burned_out,
                     1], s=node_size, marker="o", color="gray", edgecolors="none", zorder=2)
        axis.set_title(
            f"Probabilidade de incêndio — passo {step} | tempo: {self._format_time(step)}")
        axis.legend(handles=self._legend(threshold),
                    loc="center left", bbox_to_anchor=(1.0, 0.5))
        if filename:
            figure.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
