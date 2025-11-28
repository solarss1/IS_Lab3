import random
from typing import Optional, List, Union, Type
from collections.abc import Sequence

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def func(x: np.ndarray) -> np.ndarray:
    return x * np.sin(x)

def plot(x: Union[np.ndarray, Sequence[np.ndarray]],
         y: Union[np.ndarray, Sequence[np.ndarray]]) -> None:
    xs, ys = x, y
    if isinstance(xs, np.ndarray):
        xs = [xs]
    if isinstance(ys, np.ndarray):
        ys = [ys]
    assert len(xs) == len(ys), "Number of series must match"
    for xi, yi in zip(xs, ys):
        assert len(xi) == len(yi), "Series lengths must match"
        plt.plot(xi, yi)
    plt.show()

class Model:

    def __init__(self, num_parameters: int) -> None:
        self._parameters: np.ndarray = np.random.random(num_parameters)

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters
    
    @parameters.setter
    def parameters(self, value: np.ndarray) -> None:
        self._parameters = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def fitness_score(self, y_expected: np.ndarray, x: np.ndarray) -> float:
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"Model parameters: {self._parameters}"
    
class PolynomialModel(Model):

    def __init__(self, num_parameters: int) -> None:
        super().__init__(num_parameters)

    def predict(self, x: np.ndarray) -> np.ndarray:
        y_predicted: np.ndarray = np.zeros_like(x)
        for i in range(len(self._parameters)):
            y_predicted += self.parameters[i] * np.pow(x, i)
        return y_predicted
    
    def fitness_score(self, y_expected: np.ndarray, x: np.ndarray) -> float:
        y_predicted: np.ndarray = self.predict(x)
        return float(np.mean(np.pow(y_predicted - y_expected, 2)))
    
    def __str__(self) -> str:
        str_pepresentation: str = super().__str__()
        str_pepresentation = "Model type: y=p0+p1*x+p2*x^2+...+p(N-1)*x^(N-1)\n" + str_pepresentation
        return str_pepresentation
    
class CrossOver:

    def __init__(self) -> None:
        pass

    def cross(self, parent_1_params: np.ndarray, parent_2_params: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
class BLXAlphaCrossOver(CrossOver):
    """
    Blend crossover (BLX-α).
    alpha >= 0 controls exploration outside the parents' range.
    Optional [lower_bound, upper_bound] lets you constrain the parameters.
    """
    def __init__(self, alpha: float = 0.5, 
                 lower_bound: Optional[float] = None, 
                 upper_bound: Optional[float] = None) -> None:
        super().__init__()
        if alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        if (lower_bound is not None) and (upper_bound is not None) and (lower_bound > upper_bound):
            raise ValueError("lower_bound must be <= upper_bound")
        self._alpha = alpha
        self._lower = lower_bound
        self._upper = upper_bound

    def cross(self, parent_1_params: np.ndarray, parent_2_params: np.ndarray) -> np.ndarray:
        if parent_1_params.shape != parent_2_params.shape:
            raise ValueError("Parents must have the same shape")

        c_min = np.minimum(parent_1_params, parent_2_params)
        c_max = np.maximum(parent_1_params, parent_2_params)
        interval = c_max - c_min

        low = c_min - self._alpha * interval
        high = c_max + self._alpha * interval

        child = np.random.uniform(low, high)

        # Clip to global bounds if provided
        if self._lower is not None or self._upper is not None:
            lo = self._lower if self._lower is not None else -np.inf
            hi = self._upper if self._upper is not None else np.inf
            child = np.clip(child, lo, hi)

        return child

    
class Population:

    def __init__(self, individual_type: Type[Model], crossover: Optional[CrossOver] = None) -> None:
        self._population: List[Model] = []
        self._fitness_scores: Optional[List[float]] = None
        self._crossover: Optional[CrossOver] = crossover
        self._individual_type: Type[Model] = individual_type

    def __len__(self) -> int:
        return len(self._population)
    
    def add_individual(self, individual: Model) -> None:
        self._population.append(individual)

    @property
    def fitness_scores(self) -> Optional[List[float]]:
        return self._fitness_scores

    def init_fitness_scores(self, y_expected: np.ndarray, x: np.ndarray) -> None:
        if self._fitness_scores is None:
            self._fitness_scores = [m.fitness_score(y_expected, x) for m in self._population]

    def best_fitting_model(self) -> Optional[Model]:
        if self._fitness_scores is None:
            return None
        return self._population[np.argmin(self._fitness_scores)]
    
    def evolve(self) -> None:
        if self._fitness_scores is not None:
            # Select top 50% (at least 2 to allow sampling two parents)
            k = max(2, len(self._fitness_scores) // 2)
            elite_idxs = np.argsort(self._fitness_scores)[:k]
            new_population: List[Model] = [self._population[idx] for idx in elite_idxs]

            if self._crossover is not None:
                while len(new_population) < len(self._population):
                    parent_1, parent_2 = random.sample(new_population, 2)
                    assert len(parent_1.parameters) == len(parent_2.parameters)
                    child: Model = self._individual_type(len(parent_1.parameters))
                    child.parameters = self._crossover.cross(parent_1.parameters.copy(),
                                                            parent_2.parameters.copy())
                    new_population.append(child)
            self._population = new_population
            self._fitness_scores = None



POPULATION_SIZE: int = 1000
NUM_MODEL_PARAMETERS: int = 8
NUM_ITERATIONS: int = 10000
LOG_ITERATIONS: int = 1000

def main():

    np.random.seed(47)
    random.seed(47)

    x: np.ndarray = np.linspace(-np.pi, np.pi, 100)
    y_expected: np.ndarray = func(x)
    plot(x, y_expected)

    population: Population = Population(PolynomialModel, BLXAlphaCrossOver(alpha=0.5, lower_bound=-5.0, upper_bound=5.0))
    for _ in range(POPULATION_SIZE):
        population.add_individual(PolynomialModel(NUM_MODEL_PARAMETERS))
    print(f"Size of population: {len(population)}")

    for i in tqdm(range(NUM_ITERATIONS)):
        population.init_fitness_scores(y_expected, x)
        if i % LOG_ITERATIONS == 0:
            best_model: Model = population.best_fitting_model()
            # print(population.fitness_scores)
            y_predicted: np.ndarray = best_model.predict(x)
            plot([x, x], [y_expected, y_predicted])
        population.evolve()



if __name__ == "__main__":
    main()
