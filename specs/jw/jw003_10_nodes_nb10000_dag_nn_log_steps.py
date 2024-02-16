from mrunner.helpers.specification_helper import create_experiments_helper

from main import parse_args
from utils import combine_with_defaults

name = globals()["script"][:-3]

base_config = {
    "train": True,
    "data_path": "/home/asia/dcdi2/data/data_p10_e10.0_n10000_benchmark_nn_con",
    "num_vars": 10,
    "exp_path": "./out",
    "model": "DCDI-DSF",
    "lr": 0.005,
    "no_w_adjs_log": True,
    "reg_coeff": 1,
    "intervention": True,
    "intervention_type": "perfect",
    "intervention_knowledge": "known",
    "gpu": True,
    "neptune": True,
}
base_config = combine_with_defaults(base_config, defaults=vars(parse_args([])))

params_grid = {"i_dataset": [i for i in range(1, 11)]}

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="reformer-tts/dcdi2",
    script="python3 -u mrun.py",
    python_path="",
    exclude=[
        "apptainer",
        "cam",
        "gies",
        "jci",
        "igsp",
        "data",
        "venv",
    ],
    tags=[name],
    base_config=base_config,
    params_grid=params_grid,
)