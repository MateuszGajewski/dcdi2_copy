from mrunner.helpers.specification_helper import create_experiments_helper

from main import parse_args
from utils import combine_with_defaults

name = globals()["script"][:-3]

base_config = {
    "train": True,
    "data_path": "/data/data_p3_e3.0_n5000_nn",
    # "data_path": "/net/pr2/projects/plgrid/plggsubgoal/causal/data/data_p3_e3.0_n5000_nn",
    "num_vars": 3,
    "i_dataset": 1,
    "exp_path": "./out",
    "model": "DCDI-DSF",
    "lr": 0.005,
    "no_w_adjs_log": True,
    "gpu": True,
    "neptune": True,
}
base_config = combine_with_defaults(base_config, defaults=vars(parse_args([])))

params_grid = {}

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
