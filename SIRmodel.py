import random
import numpy as np
import Config
from FuelModels import CENTIMETERS_PER_FOOT, FuelModel, RothermelFuelBed


class SIRmodel:
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2

    def __init__(self, terrain_map, gamma=None, track_changes=False):
        self.terrain_map = terrain_map
        self.gamma = getattr(Config, "DEFAULT_GAMMA", 1.0 /
                             10.0) if gamma is None else gamma
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma deve estar entre 0 e 1")
        self.track_changes = track_changes

        raw_fuel_inputs = getattr(Config, "FIRE_FUEL_INPUTS", None)
        if raw_fuel_inputs is None:
            raise ValueError("Defina FIRE_FUEL_INPUTS no Config.py.")
        self.fuel_models, self.rothermel_results = self._build_fuel_models(
            raw_fuel_inputs)

        if not hasattr(terrain_map, "cell_size_m"):
            raise ValueError("Mapa deve informar cell_size_m")
        self.cell_size_m = float(terrain_map.cell_size_m)
        self.time_step_s = float(getattr(Config, "FIRE_TIME_STEP_S", 60.0))
        self.wind_speed_m_s = float(
            getattr(Config, "FIRE_WIND_SPEED_M_S", 0.0))
        self.wind_direction_degrees = (
            float(getattr(Config, "FIRE_WIND_DIRECTION_FROM_DEGREES", 0.0)) + 180.0
        ) % 360.0
        self.apply_wind_limit = bool(
            getattr(Config, "FIRE_APPLY_WIND_LIMIT", False))
        if self.cell_size_m <= 0 or self.time_step_s <= 0 or self.wind_speed_m_s < 0:
            raise ValueError(
                "Tamanho de célula e passo devem ser > 0; vento deve ser >= 0")

        # Com oito vizinhos, cada direção de avanço possui três rotas locais
        # possíveis (cardinal e duas diagonais adjacentes). Repartimos a
        # probabilidade de avanço entre elas como proposto para a grade de 8.
        # Para quatro vizinhos não há essa normalização adicional.
        self.neighbor_probability_normalization = (
            1.0 / 3.0 if bool(getattr(terrain_map,
                              "use_diagonals", False)) else 1.0
        )

        self.all_nodes = list(terrain_map.graph.nodes())
        self.n_nodes = len(self.all_nodes)
        self.node_to_idx = {node: idx for idx,
                            node in enumerate(self.all_nodes)}
        self.idx_to_node = {idx: node for node,
                            idx in self.node_to_idx.items()}
        self.states_array = np.zeros(self.n_nodes, dtype=np.uint8)
        self.history_counts = []
        self.state_changes = []
        self._front_head_history_m = []
        self._burned_area_history_m2 = []
        self._burned_area_by_terrain_history_m2 = []
        self._initialize_caches()

    @staticmethod
    def _build_fuel_models(raw_fuel_inputs):
        """Calcula um FuelModel pronto para cada fitofisionomia combustível."""
        fuel_models = {}
        calculation_results = {}
        defaults = {
            "particle_density_lb_ft3": float(Config.PARTICLE_DENSITY_LB_FT3),
            "total_mineral_content": float(Config.TOTAL_MINERAL_CONTENT),
            "effective_mineral_content": float(Config.EFFECTIVE_MINERAL_CONTENT),
            "sigma_10h_ft_inv": float(Config.SIGMA_10H_FT_INV),
            "sigma_100h_ft_inv": float(Config.SIGMA_100H_FT_INV),
        }
        for code, inputs in raw_fuel_inputs.items():
            code = int(code)
            if isinstance(inputs, FuelModel):
                fuel_models[code] = inputs
                continue

            name = inputs.get("name", f"código {code}")
            if not bool(inputs.get("burns", False)):
                fuel_models[code] = FuelModel(0.0, 1.0, 1.0, burns=False)
                continue

            fuel_data = {
                key: value for key, value in inputs.items()
                if key not in {"name", "burns"}
            }
            for field, value in defaults.items():
                fuel_data.setdefault(field, value)
            try:
                fuel_bed = RothermelFuelBed(**fuel_data)
                result = fuel_bed.calculate_base_spread_rate()
            except TypeError as error:
                raise ValueError(
                    f"Faltam ou sobram campos físicos em FIRE_FUEL_INPUTS para '{name}': {error}"
                ) from error
            except ValueError as error:
                raise ValueError(
                    f"Dados físicos inválidos em FIRE_FUEL_INPUTS para '{name}': {error}"
                ) from error

            fuel_models[code] = fuel_bed.to_fuel_model()
            calculation_results[code] = result
        return fuel_models, calculation_results

    def get_fuel_calculations(self):
        """Retorna v'_F e termos intermediários por código de fitofisionomia."""
        return self.rothermel_results.copy()

    def _initialize_caches(self):
        self.terrain_codes = self.terrain_map.get_flat_terrain_codes()
        missing_codes = set(map(int, self.terrain_codes)) - \
            set(self.fuel_models)
        if missing_codes:
            raise ValueError(
                f"Faltam FuelModels para os códigos: {sorted(missing_codes)}")

        self.combustible_node_mask = np.asarray(
            [self.fuel_models[int(code)].burns for code in self.terrain_codes],
            dtype=bool,
        )
        self.metric_terrain_codes = np.asarray(
            sorted(code for code, fuel in self.fuel_models.items() if fuel.burns),
            dtype=np.int16,
        )
        self._terrain_code_to_metric_index = np.full(
            int(self.metric_terrain_codes.max()) + 1, -1, dtype=np.int16
        )
        self._terrain_code_to_metric_index[self.metric_terrain_codes] = np.arange(
            len(self.metric_terrain_codes), dtype=np.int16
        )
        self._cell_area_m2 = self.cell_size_m ** 2

        elevation = getattr(self.terrain_map, "elevation_array", None)
        if elevation is None:
            self.elevation_m = None
            self.elevation_valid_mask = None
        else:
            self.elevation_m = np.asarray(elevation, dtype=np.float64)
            expected_shape = (self.terrain_map.grid_size_y,
                              self.terrain_map.grid_size_x)
            if self.elevation_m.shape != expected_shape:
                raise ValueError(
                    f"Raster de elevação deve ter forma {expected_shape}; recebeu {self.elevation_m.shape}"
                )
            if not np.isfinite(self.elevation_m).all():
                raise ValueError(
                    "Raster de elevação contém valores ausentes ou inválidos")
            valid_mask = getattr(
                self.terrain_map, "elevation_valid_mask", None)
            if valid_mask is None:
                self.elevation_valid_mask = np.ones(expected_shape, dtype=bool)
            else:
                self.elevation_valid_mask = np.asarray(valid_mask, dtype=bool)
                if self.elevation_valid_mask.shape != expected_shape:
                    raise ValueError(
                        "Máscara de elevação deve ter a mesma forma do raster de elevação"
                    )
        self._build_edge_cache()

    def _build_edge_cache(self):
        graph_edges = list(self.terrain_map.graph.edges())
        sources = np.empty(2 * len(graph_edges), dtype=np.int32)
        targets = np.empty(2 * len(graph_edges), dtype=np.int32)
        for index, (first, second) in enumerate(graph_edges):
            first_idx, second_idx = self.node_to_idx[first], self.node_to_idx[second]
            sources[2 * index], targets[2 * index] = first_idx, second_idx
            sources[2 * index + 1], targets[2 *
                                            index + 1] = second_idx, first_idx

        order = np.argsort(sources, kind="stable")
        sorted_sources = sources[order]
        self._edge_targets = targets[order]
        coordinates = np.asarray(self.all_nodes, dtype=np.float64)
        dx = coordinates[self._edge_targets, 0] - \
            coordinates[sorted_sources, 0]
        dy = coordinates[self._edge_targets, 1] - \
            coordinates[sorted_sources, 1]
        self._edge_is_cardinal = (np.abs(dx) + np.abs(dy)) == 1
        if self.wind_speed_m_s > 0:
            wind_angle = np.deg2rad(self.wind_direction_degrees)
            self._edge_wind_alignment = (
                dx * np.sin(wind_angle) + dy * np.cos(wind_angle)
            ) / np.hypot(dx, dy)
        else:
            self._edge_wind_alignment = np.zeros(
                len(sorted_sources), dtype=np.float32)
        self._edge_probabilities = self._calculate_edge_probabilities(
            sorted_sources, self._edge_targets)
        degree_counts = np.bincount(sorted_sources, minlength=self.n_nodes)
        self._outgoing_offsets = np.empty(self.n_nodes + 1, dtype=np.int64)
        self._outgoing_offsets[0] = 0
        np.cumsum(degree_counts, out=self._outgoing_offsets[1:])

    def _calculate_edge_probabilities(self, sources, targets):
        coordinates = np.asarray(self.all_nodes, dtype=np.float64)
        dx = coordinates[targets, 0] - coordinates[sources, 0]
        dy = coordinates[targets, 1] - coordinates[sources, 1]
        grid_distance = np.hypot(dx, dy)
        distance_m = self.cell_size_m * grid_distance

        if self.elevation_m is None:
            elevation_change = np.zeros(len(sources), dtype=np.float64)
        else:
            x, y = coordinates[:, 0].astype(
                np.intp), coordinates[:, 1].astype(np.intp)
            node_elevation = self.elevation_m[y, x]
            elevation_change = node_elevation[targets] - \
                node_elevation[sources]
            node_has_elevation = self.elevation_valid_mask[y, x]
            edge_has_elevation = (
                node_has_elevation[targets] & node_has_elevation[sources]
            )
            # Falta de MDE significa relevo plano local: phi_s = 0 nesta aresta.
            elevation_change[~edge_has_elevation] = 0.0

        wind_angle = np.deg2rad(self.wind_direction_degrees)
        # A grade interna tem y crescente para o norte, tal como o mapa exibido.
        alignment = (dx * np.sin(wind_angle) + dy *
                     np.cos(wind_angle)) / grid_distance
        uphill = elevation_change >= 0
        with_wind = alignment >= 0
        target_codes = self.terrain_codes[targets]
        slope_squared = (elevation_change / distance_m) ** 2
        probabilities = np.zeros(len(sources), dtype=np.float64)

        for terrain_code, fuel in self.fuel_models.items():
            mask = target_codes == terrain_code
            if not fuel.burns or not np.any(mask):
                continue

            sigma_ft_inv = fuel.sigma_cm_inv * CENTIMETERS_PER_FOOT
            phi_s = 5.275 * \
                fuel.relative_packing_ratio ** (-0.3) * slope_squared[mask]
            c = 7.47 * np.exp(-0.133 * sigma_ft_inv ** 0.55)
            b = 0.02526 * sigma_ft_inv ** 0.54
            e = 0.715 * np.exp(-3.59e-4 * sigma_ft_inv)
            aligned_wind_m_s = self.wind_speed_m_s * np.abs(alignment[mask])
            if self.apply_wind_limit and fuel.reaction_intensity_btu_ft2_min is not None:
                # Limite de vento de Rothermel: U_lim = 96.8 * I_R^(1/3),
                # em ft/min quando I_R está em Btu/(ft² min). A comparação é
                # feita em m/s porque a velocidade de entrada usa essa unidade.
                wind_limit_ft_min = 96.8 * (
                    fuel.reaction_intensity_btu_ft2_min ** (1.0 / 3.0)
                )
                wind_limit_m_s = wind_limit_ft_min / 196.8503937
                aligned_wind_m_s = np.minimum(aligned_wind_m_s, wind_limit_m_s)
            wind_ft_min = aligned_wind_m_s * 196.8503937
            phi_w = np.zeros(len(wind_ft_min), dtype=np.float64)
            nonzero_wind = wind_ft_min > 0
            phi_w[nonzero_wind] = (
                c * wind_ft_min[nonzero_wind] ** b *
                fuel.relative_packing_ratio ** (-e)
            )

            local_uphill, local_with_wind = uphill[mask], with_wind[mask]
            correction = np.ones(mask.sum(), dtype=np.float64)
            uphill_with_wind = local_uphill & local_with_wind
            correction[uphill_with_wind] += phi_s[uphill_with_wind] + \
                phi_w[uphill_with_wind]
            downhill_with_wind = ~local_uphill & local_with_wind
            correction[downhill_with_wind] += np.maximum(
                0.0, phi_w[downhill_with_wind] - phi_s[downhill_with_wind])
            uphill_against_wind = local_uphill & ~local_with_wind
            correction[uphill_against_wind] += np.maximum(
                0.0, phi_s[uphill_against_wind] - phi_w[uphill_against_wind])

            raw_probability = (
                fuel.base_spread_rate_m_s
                * correction
                * self.time_step_s
                / distance_m[mask]
            )
            probabilities[mask] = np.minimum(
                1.0,
                self.neighbor_probability_normalization * raw_probability,
            )
        return probabilities.astype(np.float32)

    def _reset_fire_metrics(self, initial_infected_indices):
        """Inicia as séries de frente e de área no instante t=0."""
        self._burned_area_total_m2 = 0.0
        self._burned_area_by_terrain_m2 = np.zeros(
            len(self.metric_terrain_codes), dtype=np.float64
        )
        self._front_head_history_m = []
        self._burned_area_history_m2 = []
        self._burned_area_by_terrain_history_m2 = []
        self._add_newly_burned_area(initial_infected_indices)
        self._record_fire_metrics(initial_infected_indices)

    def _add_newly_burned_area(self, newly_infected_indices):
        """Acrescenta área somente quando um nó entra em I pela primeira vez."""
        if not len(newly_infected_indices):
            return
        codes = self.terrain_codes[newly_infected_indices]
        metric_indices = self._terrain_code_to_metric_index[codes]
        metric_indices = metric_indices[metric_indices >= 0]
        if not len(metric_indices):
            return
        self._burned_area_total_m2 += len(metric_indices) * self._cell_area_m2
        np.add.at(
            self._burned_area_by_terrain_m2,
            metric_indices,
            self._cell_area_m2,
        )

    def _record_fire_metrics(self, infected_indices):
        """Registra área acumulada e o comprimento da cabeça I→S do fogo."""
        outgoing_edges = self._active_edge_indices(infected_indices)
        if len(outgoing_edges):
            targets = self._edge_targets[outgoing_edges]
            active_interfaces = (
                self._edge_is_cardinal[outgoing_edges]
                & self.combustible_node_mask[targets]
                & (self.states_array[targets] == self.SUSCEPTIBLE)
            )
            if self.wind_speed_m_s > 0 and np.any(active_interfaces):
                alignment = self._edge_wind_alignment[outgoing_edges][active_interfaces]
                head_edges = int((alignment >= np.cos(np.pi / 4)).sum())
            else:
                head_edges = 0
        else:
            head_edges = 0

        self._front_head_history_m.append(head_edges * self.cell_size_m)
        self._burned_area_history_m2.append(self._burned_area_total_m2)
        self._burned_area_by_terrain_history_m2.append(
            self._burned_area_by_terrain_m2.copy()
        )

    def _pad_fire_metrics_after_extinction(self, remaining_steps):
        """Mantém área final e frente zero até completar o horizonte solicitado."""
        for _ in range(remaining_steps):
            self._front_head_history_m.append(0.0)
            self._burned_area_history_m2.append(self._burned_area_total_m2)
            self._burned_area_by_terrain_history_m2.append(
                self._burned_area_by_terrain_m2.copy()
            )

    def get_fire_metrics(self):
        """Retorna métricas físicas de uma execução, alinhadas aos passos SIR."""
        return {
            "terrain_codes": self.metric_terrain_codes.copy(),
            "has_wind": self.wind_speed_m_s > 0,
            "front_head_m": np.asarray(self._front_head_history_m, dtype=np.float64),
            "burned_area_m2": np.asarray(self._burned_area_history_m2, dtype=np.float64),
            "burned_area_by_terrain_m2": np.asarray(
                self._burned_area_by_terrain_history_m2, dtype=np.float64
            ),
        }

    def initialize_states(self, initial_infected_count=10, infection_strategy="random", manual_nodes=None):
        self.states_array[:] = self.SUSCEPTIBLE
        if infection_strategy == "manual":
            if not manual_nodes:
                raise ValueError(
                    "É necessário fornecer nós para infecção manual.")
            initial_infected = list(manual_nodes)
        else:
            initial_infected = random.sample(
                self.all_nodes, initial_infected_count)
        initial_indices = np.fromiter(
            (self.node_to_idx[node] for node in initial_infected),
            dtype=np.intp, count=len(initial_infected),
        )
        self.states_array[initial_indices] = self.INFECTED
        self.history_counts = [self._count_states()]
        self.state_changes = [(0, int(index), self.INFECTED)
                              for index in initial_indices] if self.track_changes else []
        self._reset_fire_metrics(initial_indices)
        return self.states_array

    def _count_states(self):
        counts = np.bincount(self.states_array, minlength=3)
        return int(counts[0]), int(counts[1]), int(counts[2])

    def _active_edge_indices(self, infected_indices):
        counts = self._outgoing_offsets[infected_indices +
                                        1] - self._outgoing_offsets[infected_indices]
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.intp)
        starts = self._outgoing_offsets[infected_indices]
        block_offsets = np.cumsum(counts) - counts
        local_offsets = np.arange(
            total, dtype=np.int64) - np.repeat(block_offsets, counts)
        return np.repeat(starts, counts) + local_offsets

    def run_simulation(self, steps=200):
        susceptible_count, infected_count, recovered_count = self._count_states()
        infected_indices = np.flatnonzero(self.states_array == self.INFECTED)
        for step in range(steps):
            outgoing_edges = self._active_edge_indices(infected_indices)
            targets, probabilities = self._edge_targets[outgoing_edges], self._edge_probabilities[outgoing_edges]
            eligible = (self.states_array[targets] == self.SUSCEPTIBLE) & (
                probabilities > 0)
            targets, probabilities = targets[eligible], probabilities[eligible]
            if len(targets):
                candidate_targets, inverse = np.unique(
                    targets, return_inverse=True)
                log_not_infected = np.zeros(
                    len(candidate_targets), dtype=np.float64)
                with np.errstate(divide="ignore"):
                    np.add.at(log_not_infected, inverse,
                              np.log1p(-probabilities))
                infection_probability = -np.expm1(log_not_infected)
                new_infected = candidate_targets[np.random.random(
                    len(candidate_targets)) < infection_probability]
            else:
                new_infected = np.empty(0, dtype=np.intp)
            recovery_draw = np.random.random(
                len(infected_indices)) < self.gamma
            new_recovered, surviving_infected = infected_indices[
                recovery_draw], infected_indices[~recovery_draw]
            self.states_array[new_infected] = self.INFECTED
            self.states_array[new_recovered] = self.RECOVERED
            infected_indices = np.concatenate(
                (surviving_infected, new_infected))
            self._add_newly_burned_area(new_infected)
            susceptible_count -= len(new_infected)
            infected_count += len(new_infected) - len(new_recovered)
            recovered_count += len(new_recovered)
            self.history_counts.append(
                (susceptible_count, infected_count, recovered_count))
            if self.track_changes:
                self.state_changes.extend(
                    (step + 1, int(index), self.INFECTED) for index in new_infected)
                self.state_changes.extend(
                    (step + 1, int(index), self.RECOVERED) for index in new_recovered)
            self._record_fire_metrics(infected_indices)
            if not len(infected_indices):
                self.history_counts.extend(
                    [self.history_counts[-1]] * (steps - step - 1))
                self._pad_fire_metrics_after_extinction(steps - step - 1)
                break
        return self.history_counts

    def get_state_counts(self):
        return self.history_counts

    def reconstruct_history(self):
        if not self.track_changes:
            raise ValueError(
                "track_changes deve ser True para reconstruir histórico")
        history, current_state, change_idx = [], np.zeros(
            self.n_nodes, dtype=np.uint8), 0
        for step in range(len(self.history_counts)):
            while change_idx < len(self.state_changes) and self.state_changes[change_idx][0] == step:
                _, node_idx, new_state = self.state_changes[change_idx]
                current_state[node_idx] = new_state
                change_idx += 1
            history.append({self.idx_to_node[index]: int(
                current_state[index]) for index in range(self.n_nodes)})
        return history

    def get_current_state_dict(self):
        return {self.idx_to_node[index]: int(self.states_array[index]) for index in range(self.n_nodes)}
