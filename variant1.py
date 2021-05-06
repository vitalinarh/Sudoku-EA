""" Solves a Sudoku puzzle using a evolutionary algorithm."""

from utils import *
from operators import *
import numpy as np
import copy
from random import random, sample, randint, uniform, sample, shuffle, gauss, randrange, shuffle
from operator import itemgetter

''' Run algorithm '''

def run(numb_runs, numb_generations, size_pop, size_cromo, prob_mut,  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol):
    statistics = []
    for i in range(numb_runs):
        best, stat_best, stat_aver = sea(numb_generations, size_pop, size_cromo, prob_mut,  prob_cross,
                                         sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol)
        statistics.append(stat_best)
    stat_gener = list(zip(*statistics))
    boa = [max(g_i) for g_i in stat_gener]  # maximization
    aver_gener = [sum(g_i)/len(g_i) for g_i in stat_gener]
    return boa, aver_gener


''' Save runs in file '''


def run_for_file(filename, numb_runs, numb_generations, size_pop, size_cromo, prob_mut,  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol):
    with open(filename, 'w') as f_out:
        for i in range(numb_runs):
            best, stat_best, stat_aver = sea(numb_generations, size_pop, size_cromo, prob_mut,  prob_cross,
                                             sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol)
            f_out.write(str(best[1]) + '\n')

''' Evolutionary Algorithm '''
def sea(n_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quiz, sol):

    # restart population after a number of generations where there is no progress
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

        # Update Fitness of population
        population = [(individual[0], fitness_func(individual[0]))
                      for individual in population]

        if prev_best != best_pop(population)[1]:
            prev_best = best_pop(population)[1]
            stale = stale_threshold
        else:
            stale -= 1

        if stale == 0:
            # generate a random population if there hasn't been any progress beyond the stale threshold
            population = random_solution_gen(size_pop, size_cromo, quiz)
            population = [(individual, fitness_func(individual)) for individual in population]
            stale = stale_threshold

        print('Generation: ', gen, 'Best: ', best_pop(population)[1])
        #print_board(best_pop(population)[0])
        #print(evaluate(best_pop(population)[0]))

        # ------ Statistics --------------------------------------------------------------
        stat.append(best_pop(population)[1])
        stat_aver.append(average_pop(population))

        # ----- Found Solution -------------------------------------------------
        if best_pop(population)[1] == 162:
            break

    print_board(best_pop(population)[0])
    print('')
    print_board(sol)

    display(stat, stat_aver)

    return best_pop(population), stat, stat_aver


def print_board(board):
    for row in range(0, len(board)):
        for column in range(0, len(board[row])):
            print(board[row][column], end=' ')
        print('')


''' A candidate solutions to the Sudoku puzzle. Generates a random solution for each block by applying a random permutation from 1 to 9 to each row. Avoiding duplicates in each row'''
def random_solution_gen(size_pop, size_cromo, quizz):

    population = []

    for i in range(size_pop):
        # copy original grid
        individual = copy.deepcopy(quizz)

        for row in range(len(individual)):
            # create all legal values that can be places in the row
            available = list(range(1, 10))

            # remove values that are already inserted in the original grid
            for column in range(len(individual[row])):
                if individual[row][column] != 0:
                    available.remove(individual[row][column])

            # shuffle the resulting values
            shuffle(available)

            # apply the values to the current row
            for column in range(len(individual[row])):
                if individual[row][column] == 0:
                    individual[row][column] = available[0]
                    available.pop(0)

        population.append(individual)

    return population


''' The fitness of a candidate solution is determined by how close it is to being the actual solution to the puzzle. '''


def evaluate(individual):

    column_sum = 0
    block_sum = 0
    row_sum = 0

    dim = 9

    # for each roq
    for row in range(dim):
        row_sum += len(set(individual[row]))

    # for each column
    for column in range(dim):
        new_column = []
        for row in range(dim):
            new_column.append(individual[row][column])

        column_sum += len(set(new_column))

    # for each block
    for row in range(0, 9, 3):
        block = []
        for column in range(0, 9, 3):
            block.append(individual[row][column])
            block.append(individual[row][column + 1])
            block.append(individual[row][column + 2])

            block.append(individual[row + 1][column])
            block.append(individual[row + 1][column + 1])
            block.append(individual[row + 1][column + 2])

            block.append(individual[row + 2][column])
            block.append(individual[row + 2][column + 1])
            block.append(individual[row + 2][column + 2])

            block_sum += len(set(block))

    
    fitness = column_sum + block_sum

    return fitness


if __name__ == '__main__':

    path = 'Dataset\\sudoku.csv'
    quizzes, solutions = read_file(path)
    quizz = quizzes[1]
    sol = solutions[1]

    n_runs = 1
    n_generations = 1000
    size_pop = 800
    # Always constant (81)
    size_cromo = 81
    prob_mut = 0.65
    prob_cross = 0.9
    sel_parents = tour_sel(3)
    recombination = two_points_cross
    mutation = swap_mutation
    sel_survivors = sel_survivors_elite(0.05)
    fitness_func = evaluate

    filename = 'Sudoku\\out\\var1_gen_' + \
        str(n_generations) + '_pop_' + str(size_pop) + '_mut_' + \
        str(prob_mut) + '_cross_' + str(prob_cross) + '.csv'

    run(n_runs, n_generations, size_pop, size_cromo, prob_mut,  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol)
    #run_for_file(filename, n_runs, n_generations, size_pop, size_cromo, prob_mut,  prob_cross, sel_parents, recombination, mutation, sel_survivors, fitness_func, quizz, sol)
