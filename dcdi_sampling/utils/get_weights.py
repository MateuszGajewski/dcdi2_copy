import torch


"""
1. Sampling + no weights. only propability
2. Sampling + weights with continous gradient
3. Sampling + arbitrary set weights

"""


def get_one_weights(
    sampled_immoralities, dag_immoralities, sampled_cycles, dag_cycles, eps=0.0001
):
    return torch.tensor(1.0)


def get_soft_weights(
    sampled_immoralities, dag_immoralities, sampled_cycles, dag_cycles, eps=0.0001
):
    w = torch.exp(
        eps * (sampled_immoralities - dag_immoralities)
        + eps * (sampled_cycles - dag_cycles)
    )
    return w


def get_hard_weights(
    sampled_immoralities, dag_immoralities, sampled_cycles, dag_cycles, eps=0.0001
):
    w = torch.exp(
        -3 * (sampled_immoralities - dag_immoralities)
        - 3 * (sampled_cycles - dag_cycles)
    )
    if w > 0.99:
        return torch.tensor(1.0)
    return torch.tensor(eps)
