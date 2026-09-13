from dataclasses import dataclass
from math import exp, sqrt

CENTIMETERS_PER_FOOT = 30.48
FEET_PER_METER = 3.28083989501
POUNDS_PER_METRIC_TONNE = 2204.62262185
SQUARE_FEET_PER_HECTARE = 107639.104167
TONNES_PER_HECTARE_TO_LB_PER_FT2 = (
    POUNDS_PER_METRIC_TONNE / SQUARE_FEET_PER_HECTARE
)
KJ_PER_KG_TO_BTU_PER_LB = 0.429922614


@dataclass
class FuelModel:
    """Parâmetros prontos para o simulador de propagação."""

    base_spread_rate_m_s: float
    sigma_cm_inv: float
    relative_packing_ratio: float
    reaction_intensity_btu_ft2_min: float | None = None
    burns: bool = True

    def __post_init__(self) -> None:
        if self.base_spread_rate_m_s < 0 or self.sigma_cm_inv <= 0:
            raise ValueError("A velocidade deve ser >= 0 e sigma deve ser > 0")
        if self.relative_packing_ratio <= 0:
            raise ValueError("relative_packing_ratio deve ser > 0")


@dataclass
class RothermelBaseSpreadResult:
    """Resultado de v'_F e termos de conferência."""

    base_spread_rate_m_s: float
    base_spread_rate_ft_min: float
    characteristic_sigma_cm_inv: float
    bulk_density_lb_ft3: float
    relative_packing_ratio: float
    reaction_intensity_btu_ft2_min: float
    live_moisture_of_extinction_percent: float


@dataclass
class RothermelFuelBed:
    """Dados físicos de uma fitofisionomia, nas unidades das Tabelas 3 e 4.

    As cargas são massas secas. Os quatro últimos parâmetros reproduzem os
    valores constantes usados para modelos customizados do BEHAVE/FARSITE.
    """

    fuel_load_1h_t_ha: float
    fuel_load_10h_t_ha: float
    fuel_load_100h_t_ha: float
    live_herbaceous_fuel_t_ha: float
    live_woody_fuel_t_ha: float
    fuel_bed_depth_m: float
    sigma_1h_cm_inv: float
    sigma_live_herbaceous_cm_inv: float
    sigma_live_woody_cm_inv: float
    dead_moisture_of_extinction_percent: float
    dead_heat_content_kj_kg: float
    live_heat_content_kj_kg: float
    moisture_1h_percent: float
    moisture_10h_percent: float
    moisture_100h_percent: float
    live_herbaceous_moisture_percent: float
    live_woody_moisture_percent: float
    particle_density_lb_ft3: float = 32.0
    total_mineral_content: float = 0.0555
    effective_mineral_content: float = 0.01
    sigma_10h_ft_inv: float = 109.0
    sigma_100h_ft_inv: float = 30.0

    def calculate_base_spread_rate(self) -> RothermelBaseSpreadResult:
        """Calcula v'_F: propagação em terreno plano e sem vento, em m/s."""
        positive = (
            self.fuel_bed_depth_m,
            self.sigma_1h_cm_inv,
            self.sigma_live_herbaceous_cm_inv,
            self.sigma_live_woody_cm_inv,
            self.dead_moisture_of_extinction_percent,
            self.dead_heat_content_kj_kg,
            self.live_heat_content_kj_kg,
            self.particle_density_lb_ft3,
            self.total_mineral_content,
            self.effective_mineral_content,
            self.sigma_10h_ft_inv,
            self.sigma_100h_ft_inv,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(
                "Profundidade, SAV, densidades e parâmetros físicos devem ser positivos")

        loads_t_ha = (
            self.fuel_load_1h_t_ha,
            self.fuel_load_10h_t_ha,
            self.fuel_load_100h_t_ha,
            self.live_herbaceous_fuel_t_ha,
            self.live_woody_fuel_t_ha,
        )
        moistures_percent = (
            self.moisture_1h_percent,
            self.moisture_10h_percent,
            self.moisture_100h_percent,
            self.live_herbaceous_moisture_percent,
            self.live_woody_moisture_percent,
        )
        if any(value < 0 for value in loads_t_ha + moistures_percent):
            raise ValueError(
                "Cargas e teores de umidade não podem ser negativos")

        loads = [value * TONNES_PER_HECTARE_TO_LB_PER_FT2 for value in loads_t_ha]
        sigma = [
            self.sigma_1h_cm_inv * CENTIMETERS_PER_FOOT,
            self.sigma_10h_ft_inv,
            self.sigma_100h_ft_inv,
            self.sigma_live_herbaceous_cm_inv * CENTIMETERS_PER_FOOT,
            self.sigma_live_woody_cm_inv * CENTIMETERS_PER_FOOT,
        ]
        moisture = [value / 100.0 for value in moistures_percent]
        heat_content = [
            self.dead_heat_content_kj_kg * KJ_PER_KG_TO_BTU_PER_LB,
            self.dead_heat_content_kj_kg * KJ_PER_KG_TO_BTU_PER_LB,
            self.dead_heat_content_kj_kg * KJ_PER_KG_TO_BTU_PER_LB,
            self.live_heat_content_kj_kg * KJ_PER_KG_TO_BTU_PER_LB,
            self.live_heat_content_kj_kg * KJ_PER_KG_TO_BTU_PER_LB,
        ]
        areas = [
            sigma_value * load / self.particle_density_lb_ft3
            for sigma_value, load in zip(sigma, loads)
        ]
        dead_area, live_area = sum(areas[:3]), sum(areas[3:])
        total_area = dead_area + live_area
        if dead_area <= 0 or total_area <= 0:
            raise ValueError(
                "É necessária ao menos uma carga de combustível morto")

        dead_weights = [value / dead_area for value in areas[:3]]
        live_weights = [value / live_area for value in areas[3:]
                        ] if live_area else [0.0, 0.0]
        dead_area_fraction = dead_area / total_area
        live_area_fraction = live_area / total_area

        def weighted(values, weights):
            return sum(weight * value for weight, value in zip(weights, values))

        moisture_dead = weighted(moisture[:3], dead_weights)
        moisture_live = weighted(moisture[3:], live_weights)
        sigma_dead = weighted(sigma[:3], dead_weights)
        sigma_live = weighted(sigma[3:], live_weights)
        sigma_characteristic = (
            dead_area_fraction * sigma_dead + live_area_fraction * sigma_live
        )
        heat_dead = weighted(heat_content[:3], dead_weights)
        heat_live = weighted(heat_content[3:], live_weights)

        depth_ft = self.fuel_bed_depth_m * FEET_PER_METER
        bulk_density = sum(loads) / depth_ft
        packing_ratio = bulk_density / self.particle_density_lb_ft3
        packing_optimum = 3.348 * sigma_characteristic ** (-0.8189)
        relative_packing = packing_ratio / packing_optimum

        dead_mx = self.dead_moisture_of_extinction_percent / 100.0
        dead_heating_number = sum(
            load * exp(-138.0 / sigma_value)
            for load, sigma_value in zip(loads[:3], sigma[:3])
        )
        live_heating_number = sum(
            load * exp(-500.0 / sigma_value)
            for load, sigma_value in zip(loads[3:], sigma[3:])
        )
        if live_heating_number:
            fine_dead_moisture = sum(
                load * moisture_value * exp(-138.0 / sigma_value)
                for load, moisture_value, sigma_value in zip(
                    loads[:3], moisture[:3], sigma[:3]
                )
            ) / dead_heating_number
            live_mx = max(
                dead_mx,
                2.9 * (dead_heating_number / live_heating_number)
                * (1.0 - fine_dead_moisture / dead_mx) - 0.226,
            )
        else:
            live_mx = dead_mx

        def moisture_damping(value, extinction):
            if value >= extinction:
                return 0.0
            ratio = value / extinction
            return max(0.0, 1.0 - 2.59 * ratio + 5.11 * ratio**2 - 3.52 * ratio**3)

        mineral_damping = 0.174 * self.effective_mineral_content ** (-0.19)
        damping_dead = moisture_damping(moisture_dead, dead_mx)
        damping_live = moisture_damping(
            moisture_live, live_mx) if live_area else 0.0
        net_loads = [value * (1.0 - self.total_mineral_content)
                     for value in loads]
        net_dead_load = weighted(net_loads[:3], dead_weights)
        net_live_load = sum(net_loads[3:])

        gamma_max = sigma_characteristic ** 1.5 / (
            495.0 + 0.0594 * sigma_characteristic ** 1.5
        )
        reaction_exponent = 133.0 * sigma_characteristic ** (-0.7913)
        reaction_velocity = gamma_max * (
            relative_packing * exp(1.0 - relative_packing)
        ) ** reaction_exponent
        reaction_intensity = reaction_velocity * mineral_damping * (
            net_dead_load * heat_dead * damping_dead
            + net_live_load * heat_live * damping_live
        )
        propagating_flux = exp(
            (0.792 + 0.681 * sqrt(sigma_characteristic))
            * (packing_ratio + 0.1)
        ) / (192.0 + 0.2595 * sigma_characteristic)

        ignition_heat = [250.0 + 1116.0 * value for value in moisture]
        heating_sink_factor = (
            dead_area_fraction * weighted(
                [
                    heat * exp(-138.0 / sigma_value)
                    for heat, sigma_value in zip(ignition_heat[:3], sigma[:3])
                ],
                dead_weights,
            )
            + live_area_fraction * weighted(
                [
                    heat * exp(-138.0 / sigma_value)
                    for heat, sigma_value in zip(ignition_heat[3:], sigma[3:])
                ],
                live_weights,
            )
        )
        heat_sink = bulk_density * heating_sink_factor
        rate_ft_min = reaction_intensity * \
            propagating_flux / heat_sink if heat_sink else 0.0

        return RothermelBaseSpreadResult(
            base_spread_rate_m_s=rate_ft_min * 0.3048 / 60.0,
            base_spread_rate_ft_min=rate_ft_min,
            characteristic_sigma_cm_inv=sigma_characteristic / CENTIMETERS_PER_FOOT,
            bulk_density_lb_ft3=bulk_density,
            relative_packing_ratio=relative_packing,
            reaction_intensity_btu_ft2_min=reaction_intensity,
            live_moisture_of_extinction_percent=live_mx * 100.0,
        )

    def to_fuel_model(self) -> FuelModel:
        """Converte o resultado experimental para a estrutura usada pelo SIR."""
        result = self.calculate_base_spread_rate()
        return FuelModel(
            base_spread_rate_m_s=result.base_spread_rate_m_s,
            sigma_cm_inv=result.characteristic_sigma_cm_inv,
            relative_packing_ratio=result.relative_packing_ratio,
            reaction_intensity_btu_ft2_min=result.reaction_intensity_btu_ft2_min,
            burns=True,
        )
