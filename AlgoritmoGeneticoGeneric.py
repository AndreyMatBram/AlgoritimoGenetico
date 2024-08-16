from deap import base, creator, tools
import numpy as np
import random



# Define o tipo de problema (maximização) e o tipo de indivíduo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Atributo gerador
toolbox.register("attr_bool", random.randint, 0, 1)

# Inicializadores de estruturas
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 50)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    return sum(individual),

# Operadores genéticos
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=2)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 800

    # Avalia a população inteira
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("População inicial: ", pop)
    print("\n")

    for g in range(NGEN):
        # Seleciona a próxima geração de indivíduos
        offspring = toolbox.select(pop, len(pop))
        # Clona os indivíduos selecionados
        offspring = list(map(toolbox.clone, offspring))

        # Aplica crossover e mutação nos descendentes
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Avalia os indivíduos com uma aptidão inválida
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # A população é inteiramente substituída pelos descendentes
        pop[:] = offspring

    print("População final: ", pop)
    print("\n")

if __name__ == "__main__":
    main()
