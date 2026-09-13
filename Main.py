import os
import random
import json
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
import Config
from Config import (
    DEFAULT_INITIAL_INFECTED,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_PROBABILITY_THRESHOLD,
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_VIEWSIZE_X,
    DEFAULT_VIEWSIZE_Y,
    GRID_SIZE_X,
    GRID_SIZE_Y,
    INITIAL_INFECTED_NODES_FILE,
    MULTIPLE_SIMULATIONS_FILE,
    SIMULATION_RESULTS_CSV,
    TERRAIN_CODE_TO_COLOR,
    TERRAIN_CODE_TO_LABEL,
)
from Mapa import Mapa, build_mapa
from SIRmodel import SIRmodel
from Visualizer import Visualizer


class SimulationRunner:
    """Carro chefe"""

    def __init__(
        self,
        num_simulations=DEFAULT_NUM_SIMULATIONS,
        grid_size_x=GRID_SIZE_X,
        grid_size_y=GRID_SIZE_Y,
        existing_map=None,
        changes_for_viz=True,
        random_seed=None,
        confirm_initial_ignition=None,
    ):
        self.num_simulations = num_simulations
        self.results = []
        self.fire_metrics_runs = []
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.random_seed, self.numpy_random_seed = self._configure_random_generators(
            random_seed
        )
        self.confirm_initial_ignition = (
            bool(getattr(Config, "CONFIRM_INITIAL_IGNITION", True))
            if confirm_initial_ignition is None
            else bool(confirm_initial_ignition)
        )
        self.initial_selection_attempts = 0

        if existing_map:
            self.shared_map = existing_map
            print("Mapa de terreno existente será usado em todas as simulações")
        else:
            self.shared_map = build_mapa(grid_size_x, grid_size_y)
            print("Novo mapa de terreno criado e será reutilizado em todas as simulações")

        self.initial_infected_nodes = None
        self.changes_for_viz = changes_for_viz

        self.n_nodes = len(self.shared_map.graph.nodes())
        self.all_nodes = list(self.shared_map.graph.nodes())
        self.node_to_idx = {node: index for index,
                            node in enumerate(self.all_nodes)}

    @staticmethod
    def _configure_random_generators(random_seed):
        """Define e registra sementes para os sorteios Python e NumPy."""
        configured_seed = getattr(Config, "SIMULATION_RANDOM_SEED", None)
        seed = configured_seed if random_seed is None else random_seed
        if seed is None:
            # Seed curta para facilitar leitura, anotação e reprodução manual.
            seed = random.SystemRandom().randrange(1_000_000_000)
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("SIMULATION_RANDOM_SEED deve ser inteiro ou None")
        seed = int(seed)
        numpy_seed = seed % (2 ** 32)
        random.seed(seed)
        np.random.seed(numpy_seed)
        print(f"Semente da execução: {seed} (NumPy: {numpy_seed})")
        return seed, numpy_seed

    def _validate_initial_nodes(self, nodes):
        if len(nodes) != 4:
            raise ValueError(
                "A ignição inicial deve conter exatamente quatro nós (2×2).")
        graph = self.shared_map.graph
        normalized_nodes = [tuple(map(int, node)) for node in nodes]
        non_combustible_colors = {"blue", "white"}
        if any(
            not graph.has_node(node)
            or self.shared_map.node_colors[node] in non_combustible_colors
            for node in normalized_nodes
        ):
            raise ValueError(
                "Os nós iniciais devem existir e ser todos combustíveis.")
        return normalized_nodes

    def _write_run_metadata(self, simulation_steps):
        """Salva o necessário para reproduzir a execução posteriormente."""
        filename = f"simulation_metadata_{Config.EXPERIMENT_ID}.json"
        metadata = {
            "random_seed": self.random_seed,
            "numpy_random_seed": self.numpy_random_seed,
            "initial_infected_nodes": [list(node) for node in self.initial_infected_nodes],
            "initial_selection_attempts": self.initial_selection_attempts,
            "num_simulations": self.num_simulations,
            "simulation_steps": simulation_steps,
            "time_step_s": Config.FIRE_TIME_STEP_S,
            "grid_size": [self.grid_size_x, self.grid_size_y],
            "wind_speed_m_s": Config.FIRE_WIND_SPEED_M_S,
            "wind_direction_from_degrees": Config.FIRE_WIND_DIRECTION_FROM_DEGREES,
            "uses_diagonals": Config.USE_DIAGONAL_CONNECTIONS,
        }
        with open(filename, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        print(
            f"Metadados de reprodução salvos em '{os.path.abspath(filename)}'")

    def select_initial_infected(self, initial_infected_count=DEFAULT_INITIAL_INFECTED):
        """Sorteia um foco inicial compacto: quatro células em um quadrado 2×2."""
        if initial_infected_count != 4:
            raise ValueError(
                "A ignição inicial é um quadrado 2×2 e, portanto, requer "
                "DEFAULT_INITIAL_INFECTED = 4."
            )

        configured_nodes = getattr(
            Config, "INITIAL_INFECTED_NODES_OVERRIDE", None)
        if configured_nodes is not None:
            self.initial_infected_nodes = self._validate_initial_nodes(
                configured_nodes)
            print(
                "Nós iniciais definidos explicitamente em INITIAL_INFECTED_NODES_OVERRIDE:")
            print(self.initial_infected_nodes)
            return self.initial_infected_nodes

        # terrain_code_array está indexada como [y, x]. Esta operação examina
        # toda a grade de uma vez, em vez de percorrer nós do NetworkX em Python.
        non_combustible_codes = [
            code
            for code, color in Config.TERRAIN_CODE_TO_COLOR.items()
            if color in ("blue", "white")
        ]
        combustible = ~np.isin(
            self.shared_map.terrain_code_array, non_combustible_codes
        )
        valid_block_origins = (
            combustible[:-1, :-1]
            & combustible[:-1, 1:]
            & combustible[1:, :-1]
            & combustible[1:, 1:]
        )
        origins_y, origins_x = np.nonzero(valid_block_origins)

        if origins_x.size == 0:
            raise ValueError(
                "Não foi encontrado nenhum bloco 2×2 inteiramente combustível "
                "para iniciar o incêndio."
            )

        selected = random.randrange(origins_x.size)
        x, y = int(origins_x[selected]), int(origins_y[selected])
        self.initial_infected_nodes = [
            (x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1),
        ]
        self.initial_selection_attempts += 1
        print("Bloco 2×2 selecionado aleatoriamente para a infecção inicial:")
        print(self.initial_infected_nodes)
        return self.initial_infected_nodes

    def preview_and_confirm_initial_infected(self, initial_infected_count):
        """Exibe a ignição proposta e permite confirmar, sortear novamente ou parar."""
        configured_nodes = getattr(
            Config, "INITIAL_INFECTED_NODES_OVERRIDE", None)
        while True:
            self._print_initial_node_validation()
            self.visualize_initial_infected(
                show=True,
                save=False,
                title="Prévia da ignição inicial — confirme no terminal",
            )
            try:
                answer = input(
                    "Usar estes quatro nós iniciais? "
                    "[S]im / [N]ovo sorteio / [C]ancelar: "
                ).strip().casefold()
            except EOFError as error:
                raise RuntimeError(
                    "Não foi possível confirmar a ignição no terminal. "
                    "Use confirm_initial_ignition=False para execução não interativa."
                ) from error

            if answer in {"s", "sim", "y", "yes"}:
                return
            if answer in {"n", "não", "nao", "r", "novo"}:
                if configured_nodes is not None:
                    print(
                        "Há nós iniciais fixos no Config; novo sorteio não é permitido.")
                    continue
                self.select_initial_infected(initial_infected_count)
                continue
            if answer in {"c", "cancelar", "q", "sair"}:
                raise RuntimeError("Execução cancelada antes das simulações.")
            print("Resposta inválida. Digite S, N ou C.")

    def _print_initial_node_validation(self):
        """Mostra no terminal a mesma matriz de códigos usada no sorteio."""
        print("Validação das quatro células iniciais:")
        for x, y in self.initial_infected_nodes:
            terrain_code = int(self.shared_map.terrain_code_array[y, x])
            color = Config.TERRAIN_CODE_TO_COLOR[terrain_code]
            label = Config.TERRAIN_CODE_TO_LABEL[terrain_code]
            print(f"  ({x}, {y}): código {terrain_code} — {label} ({color})")

    def run_simulations(
        self,
        initial_infected=DEFAULT_INITIAL_INFECTED,
        simulation_steps=DEFAULT_SIMULATION_STEPS,
    ):
        started_at = time.perf_counter()
        print(
            f"Iniciando {self.num_simulations} simulações com os mesmos parâmetros")
        self.results = []
        self.fire_metrics_runs = []
        if self.changes_for_viz:
            self._initialize_probability_counts(simulation_steps)

        if self.initial_infected_nodes is None:
            self.select_initial_infected(initial_infected)

        if self.confirm_initial_ignition:
            self.preview_and_confirm_initial_infected(initial_infected)
        # Salva uma única vez o mapa da ignição efetivamente confirmada.
        self.visualize_initial_infected(show=False, save=True)
        self._write_run_metadata(simulation_steps)

        print(
            f"Configuração: {initial_infected} infectados iniciais, {simulation_steps} passos")
        print(
            f"Usando sempre os mesmos {len(self.initial_infected_nodes)} "
            "nós infectados inicialmente"
        )

        for simulation_index in range(self.num_simulations):
            simulation_started_at = time.perf_counter()
            print(
                f"\nExecutando simulação {simulation_index + 1}/{self.num_simulations}")
            sir_model = SIRmodel(
                self.shared_map, track_changes=self.changes_for_viz)
            sir_model.initialize_states(
                initial_infected,
                infection_strategy="manual",
                manual_nodes=self.initial_infected_nodes,
            )
            sir_model.run_simulation(simulation_steps)
            self.results.append(sir_model.get_state_counts())
            self.fire_metrics_runs.append(sir_model.get_fire_metrics())

            if self.changes_for_viz:
                self._accumulate_probability_counts(sir_model.state_changes)

            elapsed = time.perf_counter() - simulation_started_at
            print(
                f"Simulação {simulation_index + 1} concluída em {elapsed:.2f}s")
            # O histórico de eventos já foi acumulado; não o mantemos até a
            # renderização da animação.
            del sir_model

        if self.changes_for_viz:
            self._finalize_probability_counts()

        print(
            f"\nTempo total: {(time.perf_counter() - started_at) / 60:.2f} minutos")
        print("\nGerando visualizações...")

        self.visualize_simulation_results()
        self.save_results_to_csv()
        self.visualize_fire_metrics()
        if self.changes_for_viz:
            self.visualize_probability_map()
        return self.results

    def _initialize_probability_counts(self, simulation_steps):
        """Reserva contagens inteiras, não probabilidades em ponto flutuante."""
        if self.num_simulations <= np.iinfo(np.int8).max:
            self._probability_count_dtype = np.int8
        elif self.num_simulations <= np.iinfo(np.int16).max:
            self._probability_count_dtype = np.int16
        else:
            self._probability_count_dtype = np.int32

        # [nó, passo] torna a cumulativa no eixo temporal contígua em memória.
        # Na animação, a transposta é uma visão, sem cópia.
        shape = (self.n_nodes, simulation_steps + 1)
        # Até a cumulativa final, estes vetores são deltas no tempo: +1 na
        # ignição, -1 na recuperação. Isso evita varrer todos os nós em cada
        # passo de cada execução.
        self.infected_count_accumulator = np.zeros(
            shape, dtype=self._probability_count_dtype
        )
        self.recovered_count_accumulator = np.zeros(
            shape, dtype=self._probability_count_dtype
        )
        self._probability_counts_finalized = False

    def _accumulate_probability_counts(self, state_changes):
        """Registra apenas os eventos S→I e I→R de uma execução."""
        if not state_changes:
            return
        changes = np.asarray(state_changes, dtype=np.int32).reshape(-1, 3)
        steps, nodes, states = changes[:, 0], changes[:, 1], changes[:, 2]

        infected = states == SIRmodel.INFECTED
        recovered = states == SIRmodel.RECOVERED

        # I permanece contado a partir da ignição e deixa de ser contado no
        # passo de recuperação. R permanece contado dali em diante.
        np.add.at(
            self.infected_count_accumulator,
            (nodes[infected], steps[infected]),
            1,
        )
        np.add.at(
            self.infected_count_accumulator,
            (nodes[recovered], steps[recovered]),
            -1,
        )
        np.add.at(
            self.recovered_count_accumulator,
            (nodes[recovered], steps[recovered]),
            1,
        )

    def _finalize_probability_counts(self):
        """Converte os deltas temporais em contagens I e R por passo."""
        if self._probability_counts_finalized:
            return
        np.cumsum(
            self.infected_count_accumulator,
            axis=1,
            dtype=self._probability_count_dtype,
            out=self.infected_count_accumulator,
        )
        np.cumsum(
            self.recovered_count_accumulator,
            axis=1,
            dtype=self._probability_count_dtype,
            out=self.recovered_count_accumulator,
        )
        self._probability_counts_finalized = True

    @staticmethod
    def _mean_and_ci(values):
        values = np.asarray(values, dtype=np.float64)
        mean = values.mean(axis=0)
        if values.shape[0] <= 1:
            return mean, np.zeros_like(mean)
        return mean, 1.96 * stats.sem(values, axis=0)

    def _metric_time_minutes(self):
        return np.arange(len(self.fire_metrics_runs[0]["burned_area_m2"])) * (
            Config.FIRE_TIME_STEP_S / 60.0
        )

    def _area_m2_to_hectares(self, area_m2):
        """Converte a área das métricas, preservada internamente em m², para ha."""
        return np.asarray(area_m2, dtype=np.float64) / 10_000.0

    def _add_burned_nodes_axis(self, axis, area_unit="ha"):
        """Adiciona a escala de células, para eixo principal em ha ou m²."""
        if area_unit == "ha":
            area_per_node = (self.shared_map.cell_size_m ** 2) / 10_000.0
        elif area_unit == "m2":
            area_per_node = self.shared_map.cell_size_m ** 2
        else:
            raise ValueError("area_unit deve ser 'ha' ou 'm2'")
        if area_per_node <= 0:
            return None

        nodes_axis = axis.secondary_yaxis(
            "right",
            functions=(
                lambda area: area / area_per_node,
                lambda nodes: nodes * area_per_node,
            ),
        )
        nodes_axis.set_ylabel("Nós queimados (média)")
        nodes_axis.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        return nodes_axis

    def visualize_fire_metrics(self):
        """Cria os três gráficos físicos sem abrir novas janelas."""
        if not self.fire_metrics_runs:
            return
        self._save_fire_front_graph()
        self._save_burned_area_graph()
        self._save_burned_area_by_terrain_graph()
        self._save_stacked_burned_area_by_terrain_graph()

    def _save_fire_front_graph(self):
        time_minutes = self._metric_time_minutes()
        has_wind = self.fire_metrics_runs[0]["has_wind"]
        if not has_wind:
            print("Gráfico da cabeça do fogo não foi gerado: velocidade do vento é zero.")
            return

        head, head_ci = self._mean_and_ci(
            [metrics["front_head_m"] for metrics in self.fire_metrics_runs]
        )
        figure, axis = plt.subplots(
            figsize=(DEFAULT_VIEWSIZE_X, DEFAULT_VIEWSIZE_Y))
        axis.plot(time_minutes, head, color="black", linewidth=1.8,
                  label="Cabeça do fogo (a favor do vento)")
        if self.num_simulations > 1:
            axis.fill_between(time_minutes, np.maximum(0, head - head_ci), head + head_ci,
                              color="black", alpha=0.15, label="IC 95%")
        axis.set_title("Comprimento da cabeça do fogo")
        axis.set_xlabel("Tempo simulado (min)")
        axis.set_ylabel("Comprimento da cabeça (m)")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        filename = f"fire_front_length_{Config.EXPERIMENT_ID}.png"
        figure.savefig(filename, dpi=150)
        plt.close(figure)
        print(f"Gráfico de frente salvo em '{os.path.abspath(filename)}'")

    def _save_burned_area_graph(self):
        time_minutes = self._metric_time_minutes()
        area, area_ci = self._mean_and_ci(
            [metrics["burned_area_m2"] for metrics in self.fire_metrics_runs]
        )
        area_hectares = self._area_m2_to_hectares(area)
        area_ci_hectares = self._area_m2_to_hectares(area_ci)
        figure, axis = plt.subplots(
            figsize=(DEFAULT_VIEWSIZE_X, DEFAULT_VIEWSIZE_Y))
        axis.plot(time_minutes, area_hectares, color="black",
                  label="Área queimada agregada")
        if self.num_simulations > 1:
            axis.fill_between(time_minutes,
                              np.maximum(0, area_hectares - area_ci_hectares),
                              area_hectares + area_ci_hectares,
                              color="black", alpha=0.15, label="IC 95%")
        # Área acumulada não pode ser negativa; iniciar em zero também produz
        # uma escala inteira e legível no eixo secundário de nós.
        axis.set_ylim(bottom=0)
        axis.set_title("Área queimada acumulada")
        axis.set_xlabel("Tempo simulado (min)")
        axis.set_ylabel("Área queimada acumulada (ha)")
        self._add_burned_nodes_axis(axis)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        filename = f"burned_area_{Config.EXPERIMENT_ID}.png"
        figure.savefig(filename, dpi=150)
        plt.close(figure)
        print(
            f"Gráfico de área queimada salvo em '{os.path.abspath(filename)}'")

    def _save_burned_area_by_terrain_graph(self):
        time_minutes = self._metric_time_minutes()
        terrain_codes = self.fire_metrics_runs[0]["terrain_codes"]
        terrain_area, terrain_area_ci = self._mean_and_ci(
            [metrics["burned_area_by_terrain_m2"]
                for metrics in self.fire_metrics_runs]
        )
        total, total_ci = self._mean_and_ci(
            [metrics["burned_area_m2"] for metrics in self.fire_metrics_runs]
        )
        burned_terrain_indices = [
            terrain_index
            for terrain_index in range(len(terrain_codes))
            if np.any(terrain_area[:, terrain_index] > 0.0)
        ]
        minor_indices = []
        if len(burned_terrain_indices) >= 2:
            dominant_index = max(
                burned_terrain_indices,
                key=lambda index: terrain_area[-1, index],
            )
            minor_indices = [
                index for index in burned_terrain_indices
                if index != dominant_index
            ]

        # O zoom ganha um segundo quadro, abaixo do gráfico principal. Assim,
        # ele não encobre as curvas nem o intervalo de confiança do agregado.
        if minor_indices:
            figure, (axis, zoom_axis) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(DEFAULT_VIEWSIZE_X, DEFAULT_VIEWSIZE_Y * 1.8),
                gridspec_kw={"height_ratios": [3.2, 1.15], "hspace": 0.62},
            )
        else:
            figure, axis = plt.subplots(
                figsize=(DEFAULT_VIEWSIZE_X, DEFAULT_VIEWSIZE_Y)
            )
            zoom_axis = None
        axis.plot(time_minutes, total, color="black", linewidth=2.2,
                  label="Área queimada agregada")
        if self.num_simulations > 1:
            axis.fill_between(time_minutes,
                              np.maximum(0, total - total_ci),
                              total + total_ci,
                              color="black", alpha=0.12, label="IC 95% do agregado")

        zero_burn_handles = []
        for terrain_index, terrain_code in enumerate(terrain_codes):
            color = TERRAIN_CODE_TO_COLOR[int(terrain_code)]
            label = TERRAIN_CODE_TO_LABEL[int(terrain_code)]
            values = terrain_area[:, terrain_index]
            if np.any(values > 0.0):
                axis.plot(time_minutes, values, color=color, linewidth=1.3,
                          label=label)
            else:
                # A fitofisionomia continua documentada na legenda, mas uma
                # reta em zero não ocupa visualmente o gráfico.
                zero_burn_handles.append(
                    Line2D(
                        [0], [0], linestyle="none", marker="o", markersize=6,
                        markerfacecolor=color, markeredgecolor="none",
                        label=f"{label} (sem queima)",
                    )
                )

        if zoom_axis is not None:
            for terrain_index in minor_indices:
                terrain_code = int(terrain_codes[terrain_index])
                zoom_axis.plot(
                    time_minutes,
                    terrain_area[:, terrain_index],
                    color=TERRAIN_CODE_TO_COLOR[terrain_code],
                    linewidth=1.2,
                    label=TERRAIN_CODE_TO_LABEL[terrain_code],
                )
            zoom_axis.set_title("Zoom: classes minoritárias", fontsize=9)
            zoom_axis.set_xlabel("Tempo simulado (min)", fontsize=8)
            zoom_axis.set_ylabel(
                r"Área acumulada ($\mathrm{m}^2$)", fontsize=8)
            zoom_axis.tick_params(axis="both", labelsize=8)
            # Em escala logarítmica, o valor zero não é representável. As
            # classes minoritárias começam no primeiro valor positivo e a
            # diferença entre ordens de grandeza passa a ser visível.
            positive_minor_values = terrain_area[:, minor_indices]
            positive_minor_values = positive_minor_values[
                positive_minor_values > 0.0
            ]
            minimum_positive = np.min(positive_minor_values)
            maximum_positive = np.max(positive_minor_values)
            lower_limit = 10 ** np.floor(np.log10(minimum_positive))
            upper_limit = 10 ** np.ceil(np.log10(maximum_positive))
            if lower_limit == upper_limit:
                upper_limit *= 10.0
            zoom_axis.set_yscale("log")
            zoom_axis.set_ylim(lower_limit, upper_limit)
            zoom_axis.grid(True, which="both", alpha=0.25)
            zoom_axis.legend(fontsize=7, loc="upper left")

        # Os rótulos finais são afastados das curvas e recebem fundo opaco:
        # isso evita que uma linha atravesse as letras, preservando a leitura.
        label_offsets = [
            14 if index % 2 == 0 else -14
            for index in range(len(burned_terrain_indices))
        ]
        for label_offset, terrain_index in zip(label_offsets, burned_terrain_indices):
            terrain_code = int(terrain_codes[terrain_index])
            values = terrain_area[:, terrain_index]
            axis.annotate(
                f"{TERRAIN_CODE_TO_LABEL[terrain_code]}: {values[-1]:.0f} m²",
                xy=(time_minutes[-1], values[-1]),
                xytext=(-12, label_offset),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=7,
                color=TERRAIN_CODE_TO_COLOR[terrain_code],
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.90,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": TERRAIN_CODE_TO_COLOR[terrain_code],
                    "linewidth": 0.7,
                    "alpha": 0.8,
                },
            )

        axis.set_ylim(bottom=0)
        axis.set_title("Área queimada agregada e por fitofisionomia")
        axis.set_xlabel("Tempo simulado (min)")
        axis.set_ylabel(r"Área acumulada ($\mathrm{m}^2$)")
        self._add_burned_nodes_axis(axis, area_unit="m2")
        axis.grid(True, alpha=0.3)
        # A legenda também fica fora dos dois eixos, na base da figura.
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles + zero_burn_handles, labels + [
            handle.get_label() for handle in zero_burn_handles
        ], loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, fontsize=8)
        # A margem inferior reserva uma faixa exclusiva para a legenda. Isso
        # desloca o painel de zoom para cima e mantém visíveis seus rótulos X.
        figure.tight_layout(rect=(0, 0.32, 1, 1))
        filename = f"burned_area_by_vegetation_{Config.EXPERIMENT_ID}.png"
        figure.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(
            f"Gráfico por fitofisionomia salvo em '{os.path.abspath(filename)}'")
        self._save_burned_area_by_terrain_csv(
            time_minutes,
            terrain_codes,
            terrain_area,
            terrain_area_ci,
        )
        self._save_final_burned_area_by_terrain_bar_chart(
            terrain_codes,
            terrain_area,
            terrain_area_ci,
        )

    def _save_stacked_burned_area_by_terrain_graph(self):
        """Mostra a contribuição acumulada de cada fitofisionomia à área queimada.

        Cada faixa colorida representa a área média queimada de uma classe. A
        altura acumulada das faixas é a área total queimada; a linha preta é
        desenhada sobre elas para deixar essa igualdade explícita.
        """
        time_minutes = self._metric_time_minutes()
        terrain_codes = self.fire_metrics_runs[0]["terrain_codes"]
        terrain_area, _ = self._mean_and_ci(
            [metrics["burned_area_by_terrain_m2"]
             for metrics in self.fire_metrics_runs]
        )
        total, _ = self._mean_and_ci(
            [metrics["burned_area_m2"] for metrics in self.fire_metrics_runs]
        )

        burned_indices = [
            index
            for index in range(len(terrain_codes))
            if np.any(terrain_area[:, index] > 0.0)
        ]
        if not burned_indices:
            print("Gráfico de área empilhada não foi gerado: não houve área queimada.")
            return

        figure, axis = plt.subplots(
            figsize=(DEFAULT_VIEWSIZE_X, DEFAULT_VIEWSIZE_Y * 1.15)
        )
        series = [terrain_area[:, index] for index in burned_indices]
        colors = [
            TERRAIN_CODE_TO_COLOR[int(terrain_codes[index])]
            for index in burned_indices
        ]
        labels = [
            TERRAIN_CODE_TO_LABEL[int(terrain_codes[index])]
            for index in burned_indices
        ]
        axis.stackplot(
            time_minutes,
            *series,
            labels=labels,
            colors=colors,
            alpha=0.82,
            linewidth=0.7,
            edgecolor="white",
        )
        axis.plot(
            time_minutes,
            total,
            color="black",
            linewidth=1.8,
            label="Área queimada agregada",
            zorder=3,
        )
        axis.set_ylim(bottom=0)
        axis.set_title("Composição da área queimada por fitofisionomia")
        axis.set_xlabel("Tempo simulado (min)")
        axis.set_ylabel(r"Área queimada acumulada ($\mathrm{m}^2$)")
        self._add_burned_nodes_axis(axis, area_unit="m2")
        axis.grid(True, axis="y", alpha=0.3)

        handles, legend_labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(3, len(handles)),
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.16, 1, 1))
        filename = f"burned_area_stacked_by_vegetation_{Config.EXPERIMENT_ID}.png"
        figure.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(
            "Gráfico de área empilhada salvo em "
            f"'{os.path.abspath(filename)}'"
        )

    def _save_burned_area_by_terrain_csv(
        self,
        time_minutes,
        terrain_codes,
        terrain_area_m2,
        terrain_area_ci_m2,
    ):
        """Salva a série temporal média e o IC 95% de cada fitofisionomia."""
        cell_area_m2 = self.shared_map.cell_size_m ** 2
        rows = []
        for terrain_index, terrain_code in enumerate(terrain_codes):
            terrain_code = int(terrain_code)
            for step, time_min in enumerate(time_minutes):
                mean_area_m2 = terrain_area_m2[step, terrain_index]
                ci_area_m2 = terrain_area_ci_m2[step, terrain_index]
                rows.append(
                    {
                        "passo": step,
                        "tempo_simulado_min": time_min,
                        "codigo_terreno": terrain_code,
                        "fitofisionomia": TERRAIN_CODE_TO_LABEL[terrain_code],
                        "area_queimada_media_m2": mean_area_m2,
                        "ic95_area_queimada_m2": ci_area_m2,
                        "nos_queimados_medios": mean_area_m2 / cell_area_m2,
                        "ic95_nos_queimados": ci_area_m2 / cell_area_m2,
                    }
                )

        filename = (
            f"burned_area_by_vegetation_summary_{Config.EXPERIMENT_ID}.csv"
        )
        pd.DataFrame(rows).to_csv(filename, index=False, encoding="utf-8-sig")
        print(
            "Resumo numérico por fitofisionomia salvo em "
            f"'{os.path.abspath(filename)}'"
        )

    def _save_final_burned_area_by_terrain_bar_chart(
        self,
        terrain_codes,
        terrain_area_m2,
        terrain_area_ci_m2,
    ):
        """Salva, em arquivo próprio, a área final média por fitofisionomia."""
        burned_terrain_indices = [
            terrain_index
            for terrain_index in range(len(terrain_codes))
            if np.any(terrain_area_m2[:, terrain_index] > 0.0)
        ]
        if not burned_terrain_indices:
            print("Gráfico final por fitofisionomia não foi gerado: não houve queima.")
            return

        labels = [
            TERRAIN_CODE_TO_LABEL[int(terrain_codes[index])]
            for index in burned_terrain_indices
        ]
        colors = [
            TERRAIN_CODE_TO_COLOR[int(terrain_codes[index])]
            for index in burned_terrain_indices
        ]
        final_area_m2 = terrain_area_m2[-1, burned_terrain_indices]
        final_area_ci_m2 = terrain_area_ci_m2[-1, burned_terrain_indices]

        figure, axis = plt.subplots(
            figsize=(DEFAULT_VIEWSIZE_X, DEFAULT_VIEWSIZE_Y)
        )
        bars = axis.bar(
            labels,
            final_area_m2,
            color=colors,
            edgecolor="none",
            yerr=final_area_ci_m2 if self.num_simulations > 1 else None,
            capsize=4,
            error_kw={"ecolor": "black", "linewidth": 1},
        )
        for bar, value, ci in zip(bars, final_area_m2, final_area_ci_m2):
            axis.annotate(
                f"{value:.0f}",
                # O texto fica à direita do topo do erro-padrão, para que o
                # traço vertical e a barra horizontal não atravessem o valor.
                xy=(bar.get_x() + bar.get_width() / 2, value + ci),
                xytext=(8, 2),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=8,
                clip_on=False,
            )

        axis.set_title("Área final queimada por fitofisionomia")
        axis.set_xlabel("Fitofisionomia")
        axis.set_ylabel(r"Área final média queimada ($\mathrm{m}^2$)")
        axis.set_ylim(bottom=0)
        axis.margins(x=0.12, y=0.10)
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", labelrotation=25)
        figure.tight_layout()
        filename = f"burned_area_final_by_vegetation_{Config.EXPERIMENT_ID}.png"
        figure.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(
            "Gráfico final em barras por fitofisionomia salvo em "
            f"'{os.path.abspath(filename)}'"
        )

    def visualize_simulation_results(self):
        results_array = np.asarray(self.results)
        susceptible, infected, recovered = (
            results_array[:, :, 0], results_array[:,
                                                  :, 1], results_array[:, :, 2]
        )
        susceptible_mean, susceptible_ci = self._mean_and_ci(susceptible)
        infected_mean, infected_ci = self._mean_and_ci(infected)
        recovered_mean, recovered_ci = self._mean_and_ci(recovered)
        # I + R representa toda a área já alcançada pelo incêndio: tanto os
        # nós ainda queimando (I) quanto os que já deixaram de queimar (R).
        # O IC é calculado por simulação antes da média, para preservar a
        # relação entre I e R em cada realização estocástica.
        reached_mean, reached_ci = self._mean_and_ci(infected + recovered)
        time_range = np.arange(len(self.results[0]))
        # Todas as séries usam a escala SIR original: número médio de nós.
        figure, axis = plt.subplots(
            figsize=(DEFAULT_VIEWSIZE_X, DEFAULT_VIEWSIZE_Y)
        )

        axis.plot(
            time_range, susceptible_mean, color="forestgreen", linewidth=1.6,
            label="Suscetíveis (S)",
        )
        axis.fill_between(
            time_range,
            np.maximum(0, susceptible_mean - susceptible_ci),
            susceptible_mean + susceptible_ci,
            color="forestgreen", alpha=0.16, label="IC 95% de S",
        )
        axis.plot(
            time_range, recovered_mean, color="gray", linewidth=1.7,
            label="Recuperados (R)",
        )
        axis.fill_between(
            time_range,
            np.maximum(0, recovered_mean - recovered_ci),
            recovered_mean + recovered_ci,
            color="blue", alpha=0.18,
            label="IC 95% de R",
        )
        axis.plot(
            time_range, infected_mean, color="black", linewidth=1.5,
            label="Infectados (I)",
        )
        axis.fill_between(
            time_range,
            np.maximum(0, infected_mean - infected_ci),
            infected_mean + infected_ci,
            color="darkorange", alpha=0.15, label="IC 95% de I",
        )
        axis.plot(
            time_range, reached_mean, color="red", linewidth=1.8,
            label="Total alcançado (I + R)",
        )
        axis.fill_between(
            time_range,
            np.maximum(0, reached_mean - reached_ci),
            reached_mean + reached_ci,
            color="red", alpha=0.12,
            label="IC 95% de I + R",
        )
        axis.set_title(
            f"Resultados de {self.num_simulations} Simulações (Média com IC 95%)"
        )
        axis.set_xlabel("Passos de Tempo")
        axis.set_ylabel("Número de Nós (escala logarítmica)")
        axis.set_xlim(0, len(time_range) - 1)

        observed_upper = max(
            float(np.max(susceptible_mean + susceptible_ci)),
            float(np.max(infected_mean + infected_ci)),
            float(np.max(recovered_mean + recovered_ci)),
            float(np.max(reached_mean + reached_ci)),
        )
        rounding_unit = 10 ** max(
            0, int(np.floor(np.log10(max(observed_upper, 1.0)))) - 1
        )
        y_upper = max(
            float(rounding_unit),
            float(np.ceil(observed_upper / rounding_unit) * rounding_unit),
        )
        # Escala simétrica-log: preserva o zero de I após a extinção e, ao
        # mesmo tempo, torna visíveis valores pequenos diante de S em grades
        # grandes. Acima de um nó, a escala passa a ser logarítmica.
        axis.set_yscale("symlog", linthresh=1.0, linscale=1.0)
        axis.set_ylim(0, y_upper)
        axis.grid(True, alpha=0.55)
        # A legenda fica fora do quadro do gráfico, à direita. Assim não
        # encobre as curvas de S, I, R ou I+R em nenhuma escala de simulação.
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.78, 0.5),
            fontsize=8,
        )

        # Reserva uma faixa exclusiva para a legenda no lado direito.
        figure.tight_layout(rect=(0, 0, 0.76, 1))
        figure.savefig(MULTIPLE_SIMULATIONS_FILE, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(
            f"Gráfico salvo em '{os.path.abspath(MULTIPLE_SIMULATIONS_FILE)}'")

    def visualize_initial_infected(self, show=True, save=True, title="Mapa com Nós Inicialmente Infectados"):
        figure, axis = plt.subplots(
            figsize=(DEFAULT_VIEWSIZE_X, DEFAULT_VIEWSIZE_Y)
        )
        # Reaproveita a base estática rasterizada: é muito mais leve que pedir
        # ao NetworkX para desenhar milhões de arestas em uma grade 1000×1000.
        map_visualizer = Visualizer(self.shared_map, [])
        map_visualizer._rasterize_static_base(
            figure, axis, node_size=20, draw_edges=True
        )
        initial_positions = np.asarray(
            self.initial_infected_nodes, dtype=np.float32)
        axis.scatter(
            initial_positions[:, 0], initial_positions[:, 1],
            s=36, marker="o", color="black", edgecolors="none", zorder=3,
        )
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                       markersize=10, label="Infectados Iniciais"),
            *[
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                           markeredgecolor="lightgray" if color == "white" else "none",
                           markersize=10, label=TERRAIN_CODE_TO_LABEL[code])
                for code, color in TERRAIN_CODE_TO_COLOR.items()
            ],
        ]
        axis.legend(handles=legend_elements, loc="center left",
                    bbox_to_anchor=(1.0, 0.5))
        axis.set_title(title)
        figure.tight_layout()
        if save:
            figure.savefig(INITIAL_INFECTED_NODES_FILE)
            print(
                f"Mapa salvo em '{os.path.abspath(INITIAL_INFECTED_NODES_FILE)}'")
        if show:
            plt.show()
        plt.close(figure)

    def save_results_to_csv(self):
        data = [
            {"simulation": simulation_index + 1, "step": step_index,
             "susceptible": susceptible, "infected": infected,
             "recovered": recovered, "total": susceptible + infected + recovered}
            for simulation_index, simulation_result in enumerate(self.results)
            for step_index, (susceptible, infected, recovered) in enumerate(simulation_result)
        ]
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(SIMULATION_RESULTS_CSV, index=False)
        print(
            f"Resultados salvos em '{os.path.abspath(SIMULATION_RESULTS_CSV)}'")
        return dataframe

    def visualize_probability_map(self):
        if not self.changes_for_viz:
            print("Visualizações de probabilidade desabilitadas (changes_for_viz=False)")
            return None
        visualizer = Visualizer(self.shared_map, self.results)
        visualizer.set_probability_counts(
            self.infected_count_accumulator.T,
            self.recovered_count_accumulator.T,
            self.num_simulations,
        )
        threshold_animation = visualizer.create_probability_animation(
            threshold=DEFAULT_PROBABILITY_THRESHOLD,
            frame_step=2,
        )
        burning_probability_animation = (
            visualizer.create_burning_probability_animation(frame_step=2)
        )
        return threshold_animation, burning_probability_animation


def main():
    runner = SimulationRunner(changes_for_viz=True)
    runner.run_simulations()
    print("\nProcesso de simulações múltiplas concluído!")

    if runner.changes_for_viz:
        used_mb = (
            runner.infected_count_accumulator.nbytes
            + runner.recovered_count_accumulator.nbytes
        ) / 1e6
        print(
            f"Memória aproximada usada para contagens da animação: {used_mb:.2f} MB")


if __name__ == "__main__":
    main()
