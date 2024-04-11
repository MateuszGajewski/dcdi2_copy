import torch
import dcdi_sampling.utils.get_weights as weights

# Old way of calculating it - the gradient behaves badly
"""
def count_cycles(g):
    d = g.shape[0]
    B = torch.clone(g)
    T = torch.clone(g)
    for _ in range(0, d):
        B = B @ g
        T += B

    tr = torch.trace(T)
    return tr
"""


class DagSampler(torch.nn.Module):
    def __init__(self, patience, weights_mode="no_weights", epsilon=1.0) -> None:
        super(DagSampler, self).__init__()
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.patience = patience
        self.epsilon = epsilon
        if weights_mode == "no_weights":
            self.weight_function = weights.get_one_weights
        elif weights_mode == "soft_weights":
            self.weight_function = weights.get_soft_weights
        elif weights_mode == "hard_weights":
            self.weight_function = weights.get_hard_weights
        else:
            raise NotImplementedError

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def forward(self, cpdag: torch.Tensor, samples_number: int, drawhard=False):
        """
        Forward pass of the model.

        Parameters:
        -----------
        cpdag: torch.Tensor
            Input completed partially directed acyclic graph (CPDAG) adjacency matrix.
        samples_number: int
            Number of samples to generate.
        drawhard: bool, optional
            If True, performs hard Gumbel-Softmax transformation during sampling.

        Returns:
        --------
        Tuple containing:
        - g: torch.Tensor
            Generated graph adjacency matrix.
        - weights: torch.Tensor
            Generated weights matrix.
        - avg_correct: torch.Tensor
            Average of correct graphs.
        - dag_size
            Size of the DAG part of a graph.
        - udg_size

            Size of the undirected part of a graph.
        """
        dag = cpdag * (1 - cpdag.T)
        udg = cpdag * cpdag.T

        dag_size = torch.sum(dag)
        udg_size = torch.sum(udg) / 2

        g, weights, avg_correct = self.generate_amoral_acyclic_orientation(
            udg, dag, samples_number, self.uniform, drawhard
        )

        return g, weights, avg_correct, dag_size, udg_size

    def generate_amoral_acyclic_orientation(
        self, udg, dag, sample_number, uniform, drawhard
    ):
        """
        Generates amoral acyclic orientations if possible. If not, returns random orientation

        Parameters:
        -----------
        udg: torch.Tensor
            UDG adjacency matrix.
        dag: torch.Tensor
            DAG adjacency matrix.
        sample_number: int
            Number of samples to generate.
        uniform: torch.distributions.Uniform
            Uniform distribution object used for sampling.
        drawhard: bool
            If True, performs hard Gumbel-Sigmoid transformation during sampling.

        Returns:
        --------
        results: list
            List of generated amoral acyclic orientations.
        weights: torch.Tensor
            Weights associated with each generated orientation.
        avg_correct: float
            Average of correct orientations.
        """
        results = []
        weights = []
        correct = []
        dag_cycles = self.count_cycles(dag)
        dag_immoralities = self.count_immoralities(dag)
        iterations = 0
        p = 0.001
        while len(results) < sample_number:
            g = self.gumbel_sigmoid(udg, uniform, 1, drawhard)
            sampled_cycles = self.count_cycles(g + dag)
            sampled_immoralities = self.count_immoralities(g + dag)

            while (
                sampled_cycles != dag_cycles or sampled_immoralities != dag_immoralities
            ) and torch.rand(1) < p:
                if iterations > self.patience:
                    p = 0.999
                iterations += 1
                g = self.gumbel_sigmoid(udg, uniform, 1, drawhard)
                sampled_cycles = self.count_cycles(g + dag)
                sampled_immoralities = self.count_immoralities(g + dag)

            iterations = 0
            w = self.weight_function(
                sampled_immoralities,
                dag_immoralities,
                sampled_cycles,
                dag_immoralities,
                self.epsilon,
            )
            weights.append(w)
            correct.append(
                sampled_cycles == dag_cycles
                and sampled_immoralities == dag_immoralities
            )
            results.append(g + dag)
        weights = torch.stack(weights)
        normalizing_constant = (1 / (1.0 + torch.sum(weights))) * (
            (sample_number + 1) / sample_number
        )

        weights = weights * normalizing_constant

        return results, weights, sum(correct) / len(correct)

    @staticmethod
    def sample_logistic(shape: tuple, uniform: torch.distributions.Uniform):
        """
        Generates samples from a logistic distribution.

        Parameters:
        -----------
        shape: tuple
            Shape of the output tensor.
        uniform: torch.distributions.Uniform
            Uniform distribution object used for sampling.

        Returns:
        --------
        torch.Tensor
            Samples from the logistic distribution.
        """
        u = uniform.sample(shape)
        return torch.log(u) - torch.log(1 - u)

    @staticmethod
    def count_immoralities(g: torch.Tensor):
        """
        Counts the number of immoralities in a DAG.

        Parameters:
        -----------
        g: torch.Tensor
            Input directed acyclic graph (DAG) adjacency matrix.

        Returns:
        --------
        result: torch.Tensor
            Number of immoralities in the DAG.
        """
        n = g @ g.T
        ud = g + g.T
        a = ud - n
        result = torch.sum(a.fill_diagonal_(0) < 0)
        return result / 2

    @staticmethod
    def count_cycles(g: torch.Tensor):
        """
        Counts the number of cycles in a graph.

        Parameters:
        -----------
        g: torch.Tensor
            Input graph adjacency matrix.

        Returns:
        --------
        tr: torch.Tensor
            Number of cycles in the graph.
        """
        exp_g = torch.exp(g)
        tr = torch.diagonal(exp_g).sum()
        return tr

    def gumbel_sigmoid(
        self,
        udg: torch.Tensor,
        uniform: torch.Tensor,
        tau: float = 1,
        hard: bool = False,
    ):
        """
        Applies Gumbel-sigmoid transformation to a given matrix.

        Parameters:
        -----------
        udg: torch.Tensor
            Input matrix.
        uniform: torch.Tensor
            Tensor sampled from a uniform distribution.
        tau: float, optional
            Temperature parameter for the Gumbel-Softmax transformation.
        hard: bool, optional
            If True, performs hard Gumbel-Softmax transformation.

        Returns:
        --------
        y: torch.Tensor
            Transformed matrix.
        """
        shape = tuple(list(udg.size()))
        logistic_noise = self.sample_logistic(shape, uniform)
        udg = torch.tril(udg)

        y_soft = torch.sigmoid((udg - 0.5 + logistic_noise) / tau)

        if hard:
            y = (y_soft > 0.5).type(torch.Tensor)
        else:
            y = y_soft

        y = y * udg
        y = udg.T - y.T + y

        return y

    def generate_all_proper_dags(self, cpdag):
        dag = cpdag * (1 - cpdag.T)
        udg = cpdag * cpdag.T

        udg_size = int(torch.sum(udg) / 2)

        dag_cycles = self.count_cycles(dag)
        dag_immoralities = self.count_immoralities(dag)

        proper_dags = []
        triangular = torch.tril(cpdag)
        edges = (triangular == 1).nonzero(as_tuple=True)
        edges_t = [edges[1], edges[0]]

        for number in range(0, 2 ** (udg_size)):
            picked_edges = [[], []]
            picked_matrix = torch.zeros_like(cpdag)
            binary_number = str(bin(number))[2:]
            binary_number = binary_number.zfill(udg_size)
            mask = [int(i) for i in binary_number]
            for i, n in enumerate(mask):
                if n:
                    picked_edges[0].append(edges[0][i])
                    picked_edges[1].append(edges[1][i])
                else:
                    picked_edges[0].append(edges_t[0][i])
                    picked_edges[1].append(edges_t[1][i])
            picked_matrix[picked_edges] = 1
            picked_matrix += dag

            new_cycles = self.count_cycles(picked_matrix)
            new_immoralities = self.count_immoralities(picked_matrix)
            if (new_cycles == dag_cycles) and (new_immoralities == dag_immoralities):
                proper_dags.append(picked_matrix)
        return proper_dags


"""
ds = DagSampler()
udg = torch.tensor([[0, 1, 1, 1, 0], 
                    [1, 0, 1, 0, 0], 
                    [1, 1, 0, 1, 0], 
                    [1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0.0]])

dag = torch.tensor([[0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0]])

g, w = ds(udg +  dag, 20, True)
print(f'Weights: {w}, graph: {g}, size of adjacency matrix s: {g.shape}')


dag = torch.tensor([[0, 1, 0, 0], 
                    [0, 0, 1, 0], 
                    [0, 0, 0, 1], 
                    [1, 0, 0, 0]])

print(count_cycles(dag))
"""
