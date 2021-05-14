""" Solves a Sudoku puzzle using a evolutionary algorithm."""

from utils import *
from operators import *
import numpy as np
import copy
from random import random, sample, randint, uniform, sample, shuffle, gauss, randrange, shuffle
from operator import itemgetter
import sys

''' Run algorithm '''
def run(numb_runs, numb_generations, size_pop, size_cromo, prob_mut,  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol):
    statistics = []
    for i in range(numb_runs):
        best, stat_best, stat_aver = sea(numb_generations, size_pop, size_cromo, prob_mut,  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol)
        statistics.append(stat_best)
    stat_gener = list(zip(*statistics))
    boa = [max(g_i) for g_i in stat_gener]  # maximization
    aver_gener = [sum(g_i)/len(g_i) for g_i in stat_gener]
    print(boa)
    return boa, aver_gener

''' Save runs in file '''
def run_for_file(filename, numb_runs, numb_generations, size_pop, size_cromo, prob_mut,  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol):
    with open(filename, 'w') as f_out:
        for i in range(numb_runs):
            best, stat_best, stat_aver, gen = sea(numb_generations, size_pop, size_cromo, prob_mut,
                                                  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol)
            f_out.write(str(best[1]) + ',' + str(gen) + '\n')


''' Evolutionary Algorithm '''
def sea(n_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quiz, sol):

    #
    stale_threshold = 40
    stale = stale_threshold
    
    # generate a random population
    population = random_solution_gen(size_pop, size_cromo, quiz)

    # evaluate fitness
    population = [(individual, fitness_func(individual))
                  for individual in population]

    # for statistics
    stat = [best_pop(population)[1]]
    stat_aver = [average_pop(population)]

    prev_best = best_pop(population)[1]

    for gen in range(n_generations):
        
        # select parents
        mate_pool = sel_parents(population)
        #print(mate_pool)
    # ------ Variation Operators -----------------------------------------------------
    # ------ Crossover ---------------------------------------------------------------
        parents = []
        for i in range(0, size_pop - 1, 2):
            indiv_1 = mate_pool[i]
            indiv_2 = mate_pool[i + 1]
            children = recombination(indiv_1, indiv_2, prob_cross)
            parents.extend(children)
        # ------ Mutation ----------------------------------------------------------------
        descendents = []
        #print(parents)
        for cromo, fit in parents:
            new_indiv = mutation(cromo, prob_mut, quiz)
            descendents.append((new_indiv, fitness_func(new_indiv)))

        # ------ New population ----------------------------------------------------------
        population = sel_survivors(population, descendents)

        # ------ Update Fitness --------------
        population = [(individual[0], fitness_func(individual[0])) for individual in population]

        #print('Generation: ', gen, 'Best: ', best_pop(population)[1])

        if prev_best != best_pop(population)[1]:
            prev_best = best_pop(population)[1]
            stale = stale_threshold
        else:
            stale -= 1

        if stale == 0:
            # generate a random population
            population = random_solution_gen(size_pop, size_cromo, quiz)
            population = [(individual, fitness_func(individual)) for individual in population]
            stale = stale_threshold

        # ------ Statistics --------------------------------------------------------------
        stat.append(best_pop(population)[1])
        stat_aver.append(average_pop(population))

        # ------ Found Solution ----------------------------------------------------------
        if best_pop(population)[1] == 162:
            break

    print_board(best_pop(population)[0])
    print('')
    print_board(sol)

    #display(stat, stat_aver)

    return best_pop(population), stat, stat_aver, gen

''' A candidate solutions to the Sudoku puzzle. Generates a random solution for each block by applying a random permutation from 1 to 9 to each block. Avoiding duplicates in each block '''
def random_solution_gen(size_pop, size_cromo, quizz):

    population = []

    for i in range(size_pop):
        individual = copy.deepcopy(quizz)
        # for each block
        for row in range(0, 9, 3):
            for column in range(0, 9, 3):
                available = list(range(1, 10))
                
                if quizz[row][column] != 0:
                    available.remove(quizz[row][column])

                if quizz[row][column + 1] != 0:
                    available.remove(quizz[row][column + 1])

                if quizz[row][column + 2] != 0:
                    available.remove(quizz[row][column + 2])

                if quizz[row + 1][column] != 0:
                    available.remove(quizz[row + 1][column])

                if quizz[row + 1][column + 1] != 0:
                    available.remove(quizz[row + 1][column + 1])

                if quizz[row + 1][column + 2] != 0:
                    available.remove(quizz[row + 1][column + 2])

                if quizz[row + 2][column] != 0:
                    available.remove(quizz[row + 2][column])

                if quizz[row + 2][column + 1] != 0:
                    available.remove(quizz[row + 2][column + 1])

                if quizz[row + 2][column + 2] != 0:
                    available.remove(quizz[row + 2][column + 2])

                # shuffle the resulting values
                shuffle(available)

                if quizz[row][column] == 0:
                    individual[row][column] = available[0]
                    available.pop(0)
                if quizz[row][column + 1] == 0:
                    individual[row][column + 1] = available[0]
                    available.pop(0)
                if quizz[row][column + 2] == 0:
                    individual[row][column + 2] = available[0]
                    available.pop(0)

                if quizz[row + 1][column] == 0:
                    individual[row + 1][column] = available[0]
                    available.pop(0)
                if quizz[row + 1][column + 1] == 0:
                    individual[row + 1][column + 1] = available[0]
                    available.pop(0)
                if quizz[row + 1][column + 2] == 0:
                    individual[row + 1][column + 2] = available[0]
                    available.pop(0)

                if quizz[row + 2][column] == 0:
                    individual[row + 2][column] = available[0]
                    available.pop(0)
                if quizz[row + 2][column + 1] == 0:
                    individual[row + 2][column + 1] = available[0]
                    available.pop(0)
                if quizz[row + 2][column + 2] == 0:
                    individual[row + 2][column + 2] = available[0]
                    available.pop(0)

        population.append(individual)
        
    compute_score(individual)

    return population

''' The fitness of a candidate solution is determined by how close it is to being the actual solution to the puzzle. '''
def evaluate(individual):

    row_sum = 0
    column_sum = 0

    dim = 9

    # for each row
    for row in range(dim):
        row_sum +=  len(set(individual[row]))

    # for each column
    for column in range(dim):
        new_column = []
        for row in range(dim):
            new_column.append(individual[row][column])

        column_sum += len(set(new_column))

    fitness = column_sum + row_sum
    return fitness

if __name__ == '__main__':

    path = 'Dataset\\sudoku.csv'
    quizzes, solutions = read_file(path)
    quizz = quizzes[1]
    sol = solutions[1]

    n_runs = 30
    n_generations = 2000
    size_pop = 1000
    # Always constant (81)
    size_cromo = 81
    prob_mut = float(sys.argv[1])
    prob_cross = float(sys.argv[2])
    sel_parents = tour_sel(3)
    recombination = score_based_crossover
    mutation = scramble_in_block
    sel_survivors = sel_survivors_elite(0.05)
    fitness_func = evaluate

    filename = 'out\\var2_gen_' + \
        str(n_generations) + '_pop_' + str(size_pop) + '_mut_' + \
        str(prob_mut) + '_cross_' + str(prob_cross) + '.csv'

    #run(n_runs, n_generations, size_pop, size_cromo, prob_mut,  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol)
    run_for_file(filename, n_runs, n_generations, size_pop, size_cromo, prob_mut,  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol)

    
