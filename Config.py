import os

# Grade e simulação
GRID_SIZE_X = 500
GRID_SIZE_Y = 500
USE_DIAGONAL_CONNECTIONS = True
DEFAULT_SIMULATION_STEPS = 300
DEFAULT_INITIAL_INFECTED = 4
DEFAULT_NUM_SIMULATIONS = 200
DEFAULT_GAMMA = 1.0 / 10.0

# Dados espaciais
USE_SHAPEFILE = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEGETATION_SHAPEFILE = r"c:\Users\polya\Documents\Files QGIS\Lencois_Maranhenses 02.shp"
WORKING_CRS = "EPSG:31982"
TARGET_CELL_SIZE = 100
USE_ELEVATION = True
ELEVATION_RASTER_FILE = r"c:\Users\polya\Documents\Files QGIS\Relevo_MDE_Lencois.tif"
VEGETATION_CLASS_COLUMN = "LEGENDA"

VEGETATION_CLASS_TO_COLOR = {
    "área antropizada": "blue",
    "curso d'água": "blue",
    "reservatório": "blue",
    "depósito fluvial": "blue",
    "nuvem": "blue",
    "duna": "blue",
    "campo limpo": "gold",
    "campo sujo": "orange",
    "campo rupestre": "orange",
    "vereda": "turquoise",
    "parque de cerrado": "orange",
    "savana-estépica gramíneo-lenhosa": "orange",
    "cerrado ralo": "darkorange",
    "cerrado típico": "peru",
    "cerrado denso": "peru",
    "mata de galeria": "teal",
    "cerrado rupestre": "peru",
    "savana-estépica arborizada": "peru",
    "savana-estépica florestada": "forestgreen",
    "formação florestal": "forestgreen",
    "formação pioneira arbórea com influência fluviomarinha": "teal",
    "babaçual": "forestgreen",
    "cicatriz de queimada": "blue",
}

# Visualização e arquivos de saída
DEFAULT_PROBABILITY_THRESHOLD = 0.5
DEFAULT_VIEWSIZE_X = 10
DEFAULT_VIEWSIZE_Y = 6
EXPERIMENT_ID = 1
while os.path.exists(f"simulation_results_{EXPERIMENT_ID}.csv"):
    EXPERIMENT_ID += 1
TERRAIN_MAP_FILE = f"network_terrain_{EXPERIMENT_ID}.png"
FINAL_STATE_FILE = f"sir_final_state_{EXPERIMENT_ID}.png"
MULTIPLE_SIMULATIONS_FILE = f"multiple_simulations_results_{EXPERIMENT_ID}.png"
INITIAL_INFECTED_NODES_FILE = f"initial_infected_nodes_{EXPERIMENT_ID}.png"
SIMULATION_RESULTS_CSV = f"simulation_results_{EXPERIMENT_ID}.csv"
PROBABLE_INFECTION_PATH_FILE = f"probable_infection_path_{EXPERIMENT_ID}.mp4"

# Condições externas da simulação
FIRE_TIME_STEP_S = 60.0
FIRE_WIND_SPEED_M_S = 5
# Convenção meteorológica: direção DE ONDE o vento vem (0=N, 90=E).
FIRE_WIND_DIRECTION_FROM_DEGREES = 225
# Limite opcional de Rothermel corrigido por P.Andrews U_lim = 96.8 * I_R^(1/3), em ft/min, quando
# I_R está em Btu/(ft² min). Com False, o vento informado não é limitado.
FIRE_APPLY_WIND_LIMIT = True

# Constantes do Rothermel/BEHAVE. Podem ser alteradas, mas não precisam variar
# entre fitofisionomias enquanto não houver medições específicas.
PARTICLE_DENSITY_LB_FT3 = 32.0
TOTAL_MINERAL_CONTENT = 0.0555
EFFECTIVE_MINERAL_CONTENT = 0.01
SIGMA_10H_FT_INV = 109.0
SIGMA_100H_FT_INV = 30.0

TERRAIN_CODE_TO_COLOR = {
    0: "white", 1: "blue", 2: "gold", 3: "orange", 4: "darkorange",
    5: "peru", 6: "forestgreen", 7: "teal", 8: "olivedrab",
    9: "mediumseagreen", 10: "turquoise",
}
COLOR_TO_TERRAIN_CODE = {color: code for code,
                         color in TERRAIN_CODE_TO_COLOR.items()}
TERRAIN_CODE_TO_LABEL = {
    0: "Sem dado", 1: "Barreira", 2: "Campo limpo", 3: "Campo sujo",
    4: "Campo cerrado", 5: "Cerrado sensu stricto", 6: "Cerradão",
    7: "Mata ciliar", 8: "Murundu", 9: "Brejo-murundu",
    10: "Brejo-veredas",
}

# Dados por fitofisionomia. Cargas: t/ha de massa seca; profundidade: m;
# SAV: cm^-1; umidade: percentual da massa seca; calor: kJ/kg.
# Os valores abaixo reproduzem as Tabelas 3 e 4 de Mistry e Berardi (2005).
# Para um novo cenário de curta duração, altere as cinco umidades da respectiva
# classe antes de iniciar a simulação. O v'_F será recalculado uma única vez.
FIRE_FUEL_INPUTS = {
    0: {"name": "Sem dado", "burns": False},
    1: {"name": "Barreira", "burns": False},
    2: {
        "name": "Campo limpo", "burns": True,
        "fuel_load_1h_t_ha": 5.5, "fuel_load_10h_t_ha": 0.0,
        "fuel_load_100h_t_ha": 0.0, "live_herbaceous_fuel_t_ha": 0.4,
        "live_woody_fuel_t_ha": 0.0, "fuel_bed_depth_m": 0.5,
        "sigma_1h_cm_inv": 90.0, "sigma_live_herbaceous_cm_inv": 39.0,
        "sigma_live_woody_cm_inv": 39.0,
        "dead_moisture_of_extinction_percent": 25.0,
        "dead_heat_content_kj_kg": 17200.0, "live_heat_content_kj_kg": 16300.0,
        "moisture_1h_percent": 16.0, "moisture_10h_percent": 18.0,
        "moisture_100h_percent": 19.0, "live_herbaceous_moisture_percent": 107.0,
        "live_woody_moisture_percent": 79.0,
    },
    3: {
        "name": "Campo sujo", "burns": True,
        "fuel_load_1h_t_ha": 3.5, "fuel_load_10h_t_ha": 0.0,
        "fuel_load_100h_t_ha": 0.0, "live_herbaceous_fuel_t_ha": 0.2,
        "live_woody_fuel_t_ha": 0.2, "fuel_bed_depth_m": 0.5,
        "sigma_1h_cm_inv": 90.0, "sigma_live_herbaceous_cm_inv": 39.0,
        "sigma_live_woody_cm_inv": 39.0,
        "dead_moisture_of_extinction_percent": 25.0,
        "dead_heat_content_kj_kg": 17200.0, "live_heat_content_kj_kg": 16300.0,
        "moisture_1h_percent": 16.0, "moisture_10h_percent": 18.0,
        "moisture_100h_percent": 19.0, "live_herbaceous_moisture_percent": 107.0,
        "live_woody_moisture_percent": 79.0,
    },
    4: {
        "name": "Campo cerrado", "burns": True,
        "fuel_load_1h_t_ha": 2.9, "fuel_load_10h_t_ha": 0.7,
        "fuel_load_100h_t_ha": 0.4, "live_herbaceous_fuel_t_ha": 0.1,
        "live_woody_fuel_t_ha": 2.9, "fuel_bed_depth_m": 0.5,
        "sigma_1h_cm_inv": 90.0, "sigma_live_herbaceous_cm_inv": 39.0,
        "sigma_live_woody_cm_inv": 39.0,
        "dead_moisture_of_extinction_percent": 25.0,
        "dead_heat_content_kj_kg": 17200.0, "live_heat_content_kj_kg": 16300.0,
        "moisture_1h_percent": 16.0, "moisture_10h_percent": 18.0,
        "moisture_100h_percent": 19.0, "live_herbaceous_moisture_percent": 107.0,
        "live_woody_moisture_percent": 79.0,
    },
    5: {
        "name": "Cerrado sensu stricto", "burns": True,
        "fuel_load_1h_t_ha": 2.9, "fuel_load_10h_t_ha": 0.8,
        "fuel_load_100h_t_ha": 0.9, "live_herbaceous_fuel_t_ha": 0.1,
        "live_woody_fuel_t_ha": 2.8, "fuel_bed_depth_m": 0.5,
        "sigma_1h_cm_inv": 90.0, "sigma_live_herbaceous_cm_inv": 39.0,
        "sigma_live_woody_cm_inv": 39.0,
        "dead_moisture_of_extinction_percent": 25.0,
        "dead_heat_content_kj_kg": 17200.0, "live_heat_content_kj_kg": 16300.0,
        "moisture_1h_percent": 16.0, "moisture_10h_percent": 18.0,
        "moisture_100h_percent": 19.0, "live_herbaceous_moisture_percent": 107.0,
        "live_woody_moisture_percent": 79.0,
    },
    6: {
        "name": "Cerradão", "burns": True,
        "fuel_load_1h_t_ha": 2.9, "fuel_load_10h_t_ha": 2.1,
        "fuel_load_100h_t_ha": 3.4, "live_herbaceous_fuel_t_ha": 0.1,
        "live_woody_fuel_t_ha": 3.1, "fuel_bed_depth_m": 0.3,
        "sigma_1h_cm_inv": 90.0, "sigma_live_herbaceous_cm_inv": 39.0,
        "sigma_live_woody_cm_inv": 39.0,
        "dead_moisture_of_extinction_percent": 25.0,
        "dead_heat_content_kj_kg": 17200.0, "live_heat_content_kj_kg": 16300.0,
        "moisture_1h_percent": 16.0, "moisture_10h_percent": 18.0,
        "moisture_100h_percent": 19.0, "live_herbaceous_moisture_percent": 107.0,
        "live_woody_moisture_percent": 79.0,
    },
    7: {
        "name": "Mata ciliar", "burns": True,
        "fuel_load_1h_t_ha": 3.8, "fuel_load_10h_t_ha": 3.1,
        "fuel_load_100h_t_ha": 5.5, "live_herbaceous_fuel_t_ha": 0.6,
        "live_woody_fuel_t_ha": 3.9, "fuel_bed_depth_m": 0.3,
        "sigma_1h_cm_inv": 90.0, "sigma_live_herbaceous_cm_inv": 39.0,
        "sigma_live_woody_cm_inv": 39.0,
        "dead_moisture_of_extinction_percent": 25.0,
        "dead_heat_content_kj_kg": 17200.0, "live_heat_content_kj_kg": 16300.0,
        "moisture_1h_percent": 16.0, "moisture_10h_percent": 18.0,
        "moisture_100h_percent": 19.0, "live_herbaceous_moisture_percent": 150.0,
        "live_woody_moisture_percent": 100.0,
    },
    8: {
        "name": "Murundu", "burns": True,
        "fuel_load_1h_t_ha": 1.5, "fuel_load_10h_t_ha": 0.1,
        "fuel_load_100h_t_ha": 0.1, "live_herbaceous_fuel_t_ha": 1.4,
        "live_woody_fuel_t_ha": 0.7, "fuel_bed_depth_m": 0.3,
        "sigma_1h_cm_inv": 90.0, "sigma_live_herbaceous_cm_inv": 39.0,
        "sigma_live_woody_cm_inv": 39.0,
        "dead_moisture_of_extinction_percent": 25.0,
        "dead_heat_content_kj_kg": 17200.0, "live_heat_content_kj_kg": 16300.0,
        "moisture_1h_percent": 16.0, "moisture_10h_percent": 18.0,
        "moisture_100h_percent": 19.0, "live_herbaceous_moisture_percent": 200.0,
        "live_woody_moisture_percent": 150.0,
    },
    9: {
        "name": "Brejo-murundu", "burns": True,
        "fuel_load_1h_t_ha": 0.7, "fuel_load_10h_t_ha": 0.0,
        "fuel_load_100h_t_ha": 0.0, "live_herbaceous_fuel_t_ha": 2.3,
        "live_woody_fuel_t_ha": 0.1, "fuel_bed_depth_m": 0.4,
        "sigma_1h_cm_inv": 90.0, "sigma_live_herbaceous_cm_inv": 40.0,
        "sigma_live_woody_cm_inv": 40.0,
        "dead_moisture_of_extinction_percent": 20.0,
        "dead_heat_content_kj_kg": 17200.0, "live_heat_content_kj_kg": 16300.0,
        "moisture_1h_percent": 16.0, "moisture_10h_percent": 18.0,
        "moisture_100h_percent": 19.0, "live_herbaceous_moisture_percent": 200.0,
        "live_woody_moisture_percent": 150.0,
    },
    10: {
        "name": "Brejo-veredas", "burns": True,
        "fuel_load_1h_t_ha": 0.7, "fuel_load_10h_t_ha": 0.0,
        "fuel_load_100h_t_ha": 0.0, "live_herbaceous_fuel_t_ha": 3.0,
        "live_woody_fuel_t_ha": 0.0, "fuel_bed_depth_m": 0.4,
        "sigma_1h_cm_inv": 90.0, "sigma_live_herbaceous_cm_inv": 41.0,
        "sigma_live_woody_cm_inv": 41.0,
        "dead_moisture_of_extinction_percent": 20.0,
        "dead_heat_content_kj_kg": 17200.0, "live_heat_content_kj_kg": 16300.0,
        "moisture_1h_percent": 16.0, "moisture_10h_percent": 18.0,
        "moisture_100h_percent": 19.0, "live_herbaceous_moisture_percent": 200.0,
        "live_woody_moisture_percent": 150.0,
    },

}

