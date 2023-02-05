import functools
import random
import sys
import time
import typing as t


def override_line(text: str) -> None:
    sys.stdout.write(f"\r{' ' * 100}\r{text}")
    sys.stdout.flush()


class NaturalSelectionExperiment:
    
    def __init__(
        self,
        population_size: int,
        genes_alphabet: str,
        chromosome_length: int,
        fitness_function: t.Callable[[str], float],
        gene_chance_of_mutation: int,
        max_stale_generations: int,
        verbose: bool = False
    ) -> None:
        self.population_size = population_size
        self.genes_alphabet = genes_alphabet
        self.chromosome_length = chromosome_length
        self.gene_chance_of_mutation = gene_chance_of_mutation
        self.max_stale_generations = max_stale_generations
        self.verbose = verbose

        self.fitness_function: t.Callable[[str], float] = (functools.lru_cache(maxsize=131072))(fitness_function)


    def run(self) -> str:
        
        start_time = time.time()

        population = self.gen_initial_population()
        generation_number = 1
        if self.verbose and self.population_size <= 50:
            print(f"Initial population: {population}")

        best_individual = self.get_fittest_individual(population)

        best_score = self.fitness_function(best_individual)

        best_generation = generation_number

        generations_since_best = 0
        
        while generations_since_best < self.max_stale_generations:
            population = self.gen_new_generation(population)
            generation_number = generation_number + 1

            generation_fittest = self.get_fittest_individual(population)
            generation_fittest_score = self.fitness_function(generation_fittest)

            if generation_fittest_score > best_score:
                best_individual = generation_fittest
                best_score = generation_fittest_score
                best_generation = generation_number
                generations_since_best = 0
                if self.verbose:
                    override_line(
                        f"[Generation {generation_number:4}] "
                        f"Fittest chromosome: {generation_fittest} "
                        f"(score {generation_fittest_score:10})\n"
                    )
            else:
                generations_since_best = generations_since_best + 1

                if self.verbose:
                    override_line(
                        f"Generation {generation_number}: "
                        f"Fittest: {generation_fittest} "
                        f"(score {generation_fittest_score} "
                        f"elapsed time {(time.time() - start_time):2.2f}s)"
                    )

        if self.verbose:
            total_time = time.time() - start_time
            override_line(
                f"Fittest genome: {best_individual} "
                f"(generation {best_generation}, "
                f"score: {best_score})\n"
            )
            print(f"Generations: {generation_number}")
            print(
                f"Elapsed time: {total_time:.2f}s "
                f"(avg {total_time / generation_number:.2}s)"
            )

        return best_individual


    def gen_initial_population(self) -> t.List[str]:
        
        return [self.gen_random_chromosome() for _ in range(self.population_size)]


    def gen_random_chromosome(self) -> str:
        
        return "".join(random.choices(self.genes_alphabet, k=self.chromosome_length))


    def gen_new_generation(self, old_generation: t.List[str]) -> t.List[str]:
        
        population_fitness = self.compute_population_fitness(old_generation)

        fit_individuals_iter = iter(
            self.sample_individuals(
                population=old_generation,
                weights=population_fitness,
                sample_size=2 * self.population_size,
            )
        )

        new_generation = [self.mate(fit_individual, next(fit_individuals_iter))
                          for fit_individual in fit_individuals_iter]
        
        return new_generation


    def compute_population_fitness(self, population: t.List[str]) -> t.List[float]:
        return [self.fitness_function(individual) for individual in population]


    def get_fittest_individual(self, population: t.List[str]) -> str:
        return max(population, key=self.fitness_function)


    def sample_individuals(
        self, population: t.List[str], weights: t.List[float], sample_size: int
    ) -> t.List[str]:
        return random.choices(population, weights, k=sample_size)


    def mate(self, parent_a: str, parent_b: str) -> str:
        
        new_chromosome = self.crossover(parent_a, parent_b)
        return self.mutate(new_chromosome)

    
    def crossover(self, parent_a: str, parent_b: str) -> str:
        
        crossover_point = self.gen_crossover_point()

        if crossover_point >= self.chromosome_length:
            crossover_point -= self.chromosome_length
            parent_a, parent_b = parent_b, parent_a

        return parent_a[0:crossover_point] + parent_b[crossover_point:]


    def gen_crossover_point(self) -> int:
        
        return random.randrange(2 * self.chromosome_length)


    def mutate(self, chromosome: str) -> str:
       
        return "".join([self.gen_random_gene()
                        if random.randint(1, self.gene_chance_of_mutation) == 1
                        else gene
                        for gene in chromosome])


    def gen_random_gene(self) -> str:
        return random.choice(self.genes_alphabet)


def make_chromosome_fitness_function(target_chromosome: str) -> t.Callable[[str], float]:
    
    def fitness_function(chromosome: str) -> float:
        score = len(target_chromosome) - hamming_distance(chromosome, target_chromosome)

        return score ** 10

    return fitness_function


def hamming_distance(a: str, b: str) -> int:
    
    assert len(a) == len(b), "Strings must have the same length."
    return sum(1 if a[i] != b[i] else 0 for i in range(len(a)))


TARGET_STRING = "my name is ali ahmad and registration number is fa19-bcs-027"

experiment = NaturalSelectionExperiment(
    population_size=100,
    genes_alphabet="abcdefghijklmnopqrstuvwxyz 0123456789 -",
    chromosome_length=len(TARGET_STRING),
    fitness_function=make_chromosome_fitness_function(TARGET_STRING),
    gene_chance_of_mutation=60,
    max_stale_generations=1000,
    verbose=True,
)

solution = experiment.run()
print(solution)