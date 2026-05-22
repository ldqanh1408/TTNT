# ga_ai.py  –  Genetic Algorithm cho Chrome Dino
# Toán tử: Tournament + Uniform Crossover + Adaptive Gaussian Mutation + Age-Penalized Elitism

import os
import random
import numpy as np
import pickle
from typing import Optional

from ga.neural_network import DinoNet, NNConfig


class GAConfig:
    POPULATION_SIZE  = 80
    ELITISM_COUNT    = 8        # 10% của pop=80
    TOURNAMENT_SIZE  = 5
    CROSSOVER_RATE   = 0.80
    MUTATION_RATE   = 0.08    # giảm từ 0.12
    MUTATION_STRENGTH = 0.10  # giảm từ 0.15
    FITNESS_EVALS   = 5
    RANDOM_SEED     = None
    AGE_PENALTY     = 0.005   # penalty mỗi thế hệ cho age > 20

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class GenomeIndividual:
    __slots__ = ("network", "fitness", "age", "id", "best_gen")

    _counter = 0

    def __init__(self, network: Optional[DinoNet] = None):
        self.network  = network or DinoNet()
        self.fitness  = 0.0
        self.age      = 0
        self.best_gen = 0
        GenomeIndividual._counter += 1
        self.id = GenomeIndividual._counter

    def __repr__(self) -> str:
        return (f"<GenomeIndividual #{self.id} "
                f"fitness={self.fitness:.2f} age={self.age}>")

    @classmethod
    def _reset_counter(cls):
        cls._counter = 0

    def copy(self) -> "GenomeIndividual":
        net = self.network.copy()
        new = GenomeIndividual(network=net)
        new.fitness  = self.fitness
        new.age      = self.age
        new.best_gen = self.best_gen
        return new


class GeneticAlgorithm:
    def __init__(self, ga_cfg: Optional[GAConfig] = None,
                 nn_cfg: Optional[NNConfig] = None):
        self.ga_cfg = ga_cfg or GAConfig()
        self.nn_cfg = nn_cfg or NNConfig()
        self.population: list[GenomeIndividual] = []
        self.generation = 0
        self.best_ever: Optional[GenomeIndividual] = None
        self.history: list[dict] = []

        if self.ga_cfg.RANDOM_SEED is not None:
            random.seed(self.ga_cfg.RANDOM_SEED)
            np.random.seed(self.ga_cfg.RANDOM_SEED)

    def initialize_population(self) -> None:
        GenomeIndividual._reset_counter()
        self.population = []
        for _ in range(self.ga_cfg.POPULATION_SIZE):
            net = DinoNet(self.nn_cfg)
            self.population.append(GenomeIndividual(network=net))
        self.generation = 0
        self.best_ever  = None
        self.history    = []

        print(f"\n  [GA] Đã khởi tạo quần thể: "
              f"{self.ga_cfg.POPULATION_SIZE} cá thể")
        print(f"  [GA] Network: {self.nn_cfg.INPUT_SIZE}"
              f" → {self.nn_cfg.HIDDEN_SIZES}"
              f" → {self.nn_cfg.OUTPUT_SIZE}"
              f"  ({DinoNet(self.nn_cfg).num_params} params)")

    def inject_best(self, net: DinoNet, fitness: float, gen: int) -> None:
        ind = GenomeIndividual(network=net.copy())
        ind.fitness  = fitness
        ind.best_gen = gen
        self.best_ever = ind

    def evaluate_fitness(self, individual: GenomeIndividual,
                        fitness_fn) -> float:
        scores = []
        for _ in range(self.ga_cfg.FITNESS_EVALS):
            scores.append(fitness_fn(individual))
        return float(np.mean(scores))

    def _adjusted_fitness(self, ind: GenomeIndividual) -> float:
        """Fitness có age penalty — khuyến khích cá thể trẻ, giảm overfitting."""
        age = max(0, ind.age - 20)
        return ind.fitness - age * self.ga_cfg.AGE_PENALTY * ind.fitness

    def evaluate_all(self, fitness_fn) -> None:
        for ind in self.population:
            ind.fitness = self.evaluate_fitness(ind, fitness_fn)
            ind.age    = self.generation
        current_best = max(self.population, key=lambda x: x.fitness)
        if (self.best_ever is None
                or current_best.fitness > self.best_ever.fitness):
            self.best_ever     = current_best
            self.best_ever.best_gen = self.generation

    def _tournament_selection(self) -> list[GenomeIndividual]:
        selected = []
        pop = self.population
        k   = self.ga_cfg.TOURNAMENT_SIZE
        n   = self.ga_cfg.POPULATION_SIZE
        for _ in range(n):
            candidates = random.sample(pop, min(k, len(pop)))
            winner = max(candidates, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    def _tournament_selection_adaptive(self) -> list[GenomeIndividual]:
        """Tournament với fitness đã điều chỉnh — cân bằng giữa exploitation và exploration."""
        selected = []
        pop = self.population
        k   = self.ga_cfg.TOURNAMENT_SIZE
        n   = self.ga_cfg.POPULATION_SIZE
        for _ in range(n):
            candidates = random.sample(pop, min(k, len(pop)))
            adj_scores = [(c, self._adjusted_fitness(c)) for c in candidates]
            winner = max(adj_scores, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1: GenomeIndividual,
                 parent2: GenomeIndividual) -> GenomeIndividual:
        w1 = parent1.network.get_flat_weights()
        w2 = parent2.network.get_flat_weights()
        mask       = np.random.randint(0, 2, size=w1.size).astype(bool)
        child_weights = np.where(mask, w1, w2)
        child_net = DinoNet(self.nn_cfg)
        child_net.set_flat_weights(child_weights)
        return GenomeIndividual(network=child_net)

    def mutate(self, individual: GenomeIndividual,
               sigma_override: Optional[float] = None) -> None:
        """Gaussian Mutation với adaptive sigma."""
        weights = individual.network.get_flat_weights()
        size    = weights.size

        # Adaptive sigma: giảm theo thế hệ để exploitation dần
        base_rate = self.ga_cfg.MUTATION_RATE
        base_sigma = self.ga_cfg.MUTATION_STRENGTH
        gen_ratio = self.generation / max(1, 300)
        adaptive_sigma = sigma_override or (base_sigma * (1.0 - 0.5 * gen_ratio))
        adaptive_rate = base_rate * (1.0 - 0.3 * gen_ratio)

        mask  = np.random.rand(size) < adaptive_rate
        noise = np.random.randn(size) * adaptive_sigma
        weights[mask] += noise[mask]
        individual.network.set_flat_weights(weights)

    def create_next_generation(self) -> None:
        pop_size = self.ga_cfg.POPULATION_SIZE
        elite    = self.ga_cfg.ELITISM_COUNT

        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # Bước 1: Giữ elite (cá thể tốt nhất của thế hệ hiện tại)
        new_pop = [ind.copy() for ind in sorted_pop[:elite]]

        # Bước 2: BẮT BUỘC đưa best_ever vào quần thể mới
        # Đây là cơ chế chống Catastrophic Forgetting
        if self.best_ever is not None:
            be_copy = self.best_ever.copy()
            be_copy.age = self.generation
            # Đưa best_ever vào vị trí ngẫu nhiên, không phải cuối
            insert_pos = random.randint(0, min(len(new_pop), pop_size - 1))
            new_pop.insert(insert_pos, be_copy)

        parents = self._tournament_selection_adaptive()

        while len(new_pop) < pop_size:
            p1, p2 = random.sample(parents, 2)
            if random.random() < self.ga_cfg.CROSSOVER_RATE:
                child = self.crossover(p1, p2)
            else:
                child = GenomeIndividual(network=p1.network.copy())
            self.mutate(child)
            new_pop.append(child)

        self.population = new_pop[:pop_size]
        self.generation += 1

    def copy_best(self) -> GenomeIndividual:
        if self.best_ever is None:
            raise RuntimeError("Chưa có cá thể nào.")
        ind  = self.best_ever
        copy = GenomeIndividual(network=ind.network.copy())
        copy.fitness  = ind.fitness
        copy.age      = ind.age
        copy.best_gen = ind.best_gen
        return copy

    def evolve(self, n_generations: int, fitness_fn,
               verbose_every: int = 10,
               save_path: str = "model/best_dino_ga.pkl",
               checkpoint_every: int = 50,
               callback_each_gen: Optional[callable] = None) -> GenomeIndividual:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  🧬 BẮT ĐẦU TIẾN HOÁ – {n_generations} thế hệ")
        print(f"  Pop={self.ga_cfg.POPULATION_SIZE}  "
              f"Elite={self.ga_cfg.ELITISM_COUNT}  "
              f"Mut={self.ga_cfg.MUTATION_RATE*100:.0f}% σ={self.ga_cfg.MUTATION_STRENGTH}  "
              f"FitEvals={self.ga_cfg.FITNESS_EVALS}")
        print(f"{'='*60}")
        self.evaluate_all(fitness_fn)
        self.best_ever.best_gen = self.generation
        self._log_generation()
        self._save_best(save_path)

        prev_best = self.best_ever.fitness if self.best_ever else 0
        for gen in range(1, n_generations + 1):
            self.create_next_generation()
            self.evaluate_all(fitness_fn)
            self._log_generation(verbose=(gen % verbose_every == 0))
            if callback_each_gen:
                callback_each_gen(self, gen)

            improved = self.best_ever.fitness > prev_best
            if improved:
                self._save_best(save_path)
                prev_best = self.best_ever.fitness

            if checkpoint_every > 0 and gen % checkpoint_every == 0:
                ckpt_path = save_path.replace(".pkl", f"_gen{gen}.pkl")
                self._save_best(ckpt_path)

        print(f"\n{'='*60}")
        print(f"  ✅ TIẾN HOÁ HOÀN TẤT!")
        print(f"  Best ever: fitness={self.best_ever.fitness:.2f}  "
              f"at gen={self.best_ever.best_gen}")
        print(f"  Đã lưu vào: {save_path}")
        print(f"{'='*60}\n")
        return self.best_ever

    def _log_generation(self, verbose: bool = True) -> None:
        fitnesses = [ind.fitness for ind in self.population]
        best_fit  = max(fitnesses)
        avg_fit   = float(np.mean(fitnesses))
        worst_fit = min(fitnesses)
        std_fit   = float(np.std(fitnesses))

        self.history.append({
            "generation":    self.generation,
            "best_fitness":  best_fit,
            "avg_fitness":   avg_fit,
            "worst_fitness": worst_fit,
            "std_fitness":   std_fit,
        })

        if verbose:
            delta = ""
            if self.generation > 0 and len(self.history) > 1:
                prev_avg = self.history[-2]["avg_fitness"]
                delta = f"  (Δ avg={avg_fit - prev_avg:+.2f})"
            print(f"  Gen {self.generation:>4} | "
                  f"Best={best_fit:>7.2f}  "
                  f"Avg={avg_fit:>7.2f}{delta}  "
                  f"Worst={worst_fit:>7.2f}  "
                  f"Std={std_fit:>5.2f}")

    def _save_best(self, path: str) -> None:
        if self.best_ever is None:
            return
        best = self.copy_best()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "weights":    best.network.get_flat_weights(),
                "fitness":    best.fitness,
                "best_gen":   best.best_gen,
                "nn_config": {
                    "input_size":   self.nn_cfg.INPUT_SIZE,
                    "hidden_sizes": self.nn_cfg.HIDDEN_SIZES,
                    "output_size":  self.nn_cfg.OUTPUT_SIZE,
                },
            }, f)

    @classmethod
    def load_best(cls, path: str) -> tuple[DinoNet, dict]:
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = NNConfig()
        cfg.INPUT_SIZE   = data["nn_config"]["input_size"]
        cfg.HIDDEN_SIZES = data["nn_config"]["hidden_sizes"]
        cfg.OUTPUT_SIZE  = data["nn_config"]["output_size"]
        net = DinoNet(cfg)
        net.set_flat_weights(data["weights"])
        fitness  = data["fitness"]
        best_gen = data.get("best_gen", data.get("generation", 0))
        meta = {"fitness": fitness, "best_gen": best_gen}
        print(f"  [GA] Đã tải best dino từ {path}  "
              f"(fitness={fitness:.2f}, gen={best_gen})")
        return net, meta

    def print_history(self) -> None:
        if not self.history:
            print("  Chưa có lịch sử.")
            return
        print(f"\n{'─'*60}")
        print(f"{'Gen':>4}  {'Best':>8}  {'Avg':>8}  {'Worst':>8}  {'Std':>7}")
        print(f"{'─'*60}")
        for h in self.history:
            print(f"{h['generation']:>4}  "
                  f"{h['best_fitness']:>8.2f}  "
                  f"{h['avg_fitness']:>8.2f}  "
                  f"{h['worst_fitness']:>8.2f}  "
                  f"{h['std_fitness']:>7.2f}")
        print(f"{'─'*60}\n")
