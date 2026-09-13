from Mapa import Mapa
from SIRmodel import SIRmodel
from Config import (
    GRID_SIZE_X,
    GRID_SIZE_Y,
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_INITIAL_INFECTED,
    USE_DIAGONAL_CONNECTIONS,
    VEGETATION_SHAPEFILE,
    VEGETATION_CLASS_COLUMN
)


class SimulationApp:

    def __init__(
        self,
        grid_size_x=GRID_SIZE_X,
        grid_size_y=GRID_SIZE_Y,
        use_diagonals=USE_DIAGONAL_CONNECTIONS,
        csv_file=None,
        shapefile_info=None
    ):
        self.terrain_map = Mapa(
            grid_size_x=grid_size_x,
            grid_size_y=grid_size_y,
            csv_file=csv_file,
            shapefile_info=shapefile_info,
            use_diagonals=use_diagonals
        )
        self.sir_model = SIRmodel(self.terrain_map)

    def run(
        self,
        initial_infected=DEFAULT_INITIAL_INFECTED,
        infection_strategy="random",
        simulation_steps=DEFAULT_SIMULATION_STEPS
    ):

        print(
            f"Iniciando simulação com grid "
            f"{self.terrain_map.grid_size_x}x{self.terrain_map.grid_size_y}"
        )
        print(f"Número de nós: {len(self.terrain_map.graph)}")
        print(
            f"Conexões diagonais: {'Sim' if self.terrain_map.use_diagonals else 'Não'}")

        self.sir_model.initialize_states(initial_infected, infection_strategy)
        self.sir_model.run_simulation(simulation_steps)
