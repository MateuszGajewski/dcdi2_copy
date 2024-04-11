from mrunner.helpers.specification_helper import create_experiments_helper

from main import parse_args
from utils import combine_with_defaults

name = globals()["script"][:-3]

base_config = {
    "train": True,
    "data_path": "/net/pr2/projects/plgrid/plggcausal/causal/data/data_p10_e20.0_n10000_identifiable_obs_bench",
    "num_vars": 10,
    "i_dataset": 1,
    "exp_path": "./out",
    "model": "DCDI-G",
    # "lr": 0.005,
    "no_w_adjs_log": True,
    "gpu": True,
    "reg_coeff": 0.1,
    "intervention": False,
    "intervention_type": "perfect",
    "intervention_knowledge": "known",
    "sampling": True,
    "dags_per_sample": 4,
    "cpdag": True,
    "neptune": True,
    "train_batch_size": 10,
    "weights_mode": "no_weights",
}
base_config = combine_with_defaults(base_config, defaults=vars(parse_args([])))

params_grid = {"i_dataset": [i for i in range(1, 11)], "reg_coeff": [0.05, 0.1, 0.3]}

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
