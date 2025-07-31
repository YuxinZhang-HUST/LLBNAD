from .mvtec import MVTecSolver
from .visa import VisASolver
from .real_case import RealCaseSolver

GEN_DATA_SOLVER = {
    'mvtec': MVTecSolver,
    'visa': VisASolver,
    'real_case': RealCaseSolver
}
