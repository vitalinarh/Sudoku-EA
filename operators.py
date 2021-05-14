from operator import itemgetter
import numpy as np
import copy
from random import random, sample, randint, uniform, sample, shuffle, gauss, randrange, shuffle

# =================== Parents Selection =========================================
# Tournament Selection
def tour_sel(t_size):
    def tournament(pop):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = one_tour(pop, t_size)
            mate_pool.append(winner)
        return mate_pool
    return tournament

def one_tour(population, size):
    """Maximization Problem. Deterministic"""
    pool = sample(list(range(len(population))), size)
    aux = []
    for i in pool:
        aux.append(population[i])
    aux.sort(key=itemgetter(1), reverse=True)
    return aux[0]

# =================== Survivals Selection =========================================
# Elitism
def sel_survivors_elite(elite):
    def elitism(parents, offspring):
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    return elitism

# ==================== VARIATION OPERATORS =========================================
""" Crossover relates to the analogy of genes within each parent candidate mixing together in the hopes of 
    creating a fitter child candidate. Cycle crossover is used here (see e.g. A. E. Eiben, J. E. Smith. 
    Introduction to Evolutionary Computing. Springer, 2007). """ 
# score based crossover
def score_based_crossover(indiv_1, indiv_2, prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = copy.deepcopy(indiv_1[0])
        cromo_2 = copy.deepcopy(indiv_2[0])

        row_scores1, column_scores1 = compute_score(cromo_1)
        row_scores2, column_scores2 = compute_score(cromo_2)

        # create child 1
        for i in range(3):
            if row_scores1[i] > row_scores2[i]:
                for j in range(3):
                    cromo_1[i * 3 + j] = cromo_1[i * 3 + j]
            else:
                for j in range(3):
                    cromo_1[i * 3 + j] = cromo_2[i * 3 + j]
            
        # create child 2
        for i in range(3):
            if column_scores1[i] > column_scores2[i]:
                for column in range(3):
                    for row in range(9):
                        cromo_2[row][column * i] = cromo_1[row][column * i]
            else:
                for column in range(3):
                    for row in range(9):
                        cromo_2[row][column * i] = cromo_2[row][column * i]

        return ((cromo_1, 0), (cromo_2, 0))
    else:
        return (indiv_1, indiv_2)

# one block crossover
""" (!) Use only with variant2. Select a random block and swap. """
def one_block_crossover(indiv_1, indiv_2, prob_cross):
    value = random()
    if value < prob_cross:

        cromo_1 = copy.deepcopy(indiv_1[0])
        cromo_2 = copy.deepcopy(indiv_2[0])

        idx = [0, 3, 6]
        row = randint(0, 2)
        column = randint(0, 2)
        row = idx[row]
        column = idx[column]

        # swap values
        cromo_1[row][column], cromo_2[row][column] = cromo_2[row][column], cromo_1[row][column]
        cromo_1[row][column + 1], cromo_2[row][column + 1] = cromo_2[row][column + 1], cromo_1[row][column + 1]
        cromo_1[row][column + 2], cromo_2[row][column + 2] = cromo_2[row][column + 2], cromo_1[row][column + 2]

        cromo_1[row + 1][column], cromo_2[row + 1][column] = cromo_2[row + 1][column], cromo_1[row + 1][column]
        cromo_1[row + 1][column + 1], cromo_2[row + 1][column + 1] = cromo_2[row + 1][column + 1], cromo_1[row + 1][column + 1]
        cromo_1[row + 1][column + 2], cromo_2[row + 1][column + 2] = cromo_2[row + 1][column + 2], cromo_1[row + 1][column + 2]

        cromo_1[row + 2][column], cromo_2[row + 2][column] = cromo_2[row + 2][column], cromo_1[row + 2][column]
        cromo_1[row + 2][column + 1], cromo_2[row + 2][column + 1] = cromo_2[row + 2][column + 1], cromo_1[row + 2][column + 1]
        cromo_1[row + 2][column + 2], cromo_2[row + 2][column + 2] = cromo_2[row + 2][column + 2], cromo_1[row + 2][column + 2]

        return ((cromo_1, 0), (cromo_2, 0))
    else:
        return (indiv_1, indiv_2)

# one row crossover
""" (!) Use only with variant1. Take one of the rows, and swap them. """
def one_row_crossover(indiv_1, indiv_2, prob_cross):
    value = random()
    if value < prob_cross:

        cromo_1 = copy.deepcopy(indiv_1[0])
        cromo_2 = copy.deepcopy(indiv_2[0])

        row = randint(0, 8)

        cromo_1[row], cromo_2[row] = cromo_2[row], cromo_1[row]

        return ((cromo_1, 0), (cromo_2, 0))
    else:
        return (indiv_1, indiv_2)

# One Point Crossover
""" (!) Use only with variant1. Create two new child candidates by crossing over parent genes where a random crossover point is selected and the tails of its two parents are swapped to get new off-springs."""
def one_point_cross(indiv_1, indiv_2, prob_cross):
    value = random()
    if value < prob_cross:

        cromo_1 = copy.deepcopy(indiv_1[0])
        cromo_2 = copy.deepcopy(indiv_2[0])

        pos = randint(0, len(cromo_1))
        f1 = cromo_1[0 : pos] + cromo_2[pos:]
        f2 = cromo_2[0 : pos] + cromo_1[pos:]

        return ((f1, 0), (f2, 0))
    else:
	    return (indiv_1, indiv_2)

# Two Points Crossover
""" (!) Use only with variant1. Take two points of the chromossome and swap the rows up until that point. """
def two_points_cross(indiv_1, indiv_2, prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = copy.deepcopy(indiv_1[0])
        cromo_2 = copy.deepcopy(indiv_2[0])
        pc = sample(range(len(cromo_1)), 2)
        pc.sort()
        pc1, pc2 = pc
        f1 = cromo_1[:pc1] + cromo_2[pc1 : pc2] + cromo_1[pc2:]
        f2 = cromo_2[:pc1] + cromo_1[pc1 : pc2] + cromo_2[pc2:]
        return ((f1, 0), (f2, 0))
    else:
        return (indiv_1, indiv_2)

# Uniform Crossover
"""  (!) Use only with variant1. Each row of the chromossome is selected randomly from one of the corresponding genes of the parent chromosome """
def uniform_cross(indiv_1, indiv_2, prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = copy.deepcopy(indiv_1[0])
        cromo_2 = copy.deepcopy(indiv_2[0])
        f1 = []
        f2 = []
        for i in range(0, len(chromo_1)):
            if random() < 0.5:
                f1.append(chromo_1[i])
                f2.append(chromo_2[i])
            else:
                f1.append(chromo_2[i])
                f2.append(chromo_1[i])

        return ((f1, 0), (f2, 0))
    else:
        return (indiv_1, indiv_2)
    
# Mutation
""" (!) Use only with variant2. Swaps two values in a random block. """
def swap_in_block(cromo, prob_muta, quiz):

    done = False

    if random() < prob_muta:
        cromo_temp = copy.deepcopy(cromo)

        while (not done):
            # choose random block
            idx = [0, 3, 6]
            row = randint(0, 2)
            column = randint(0, 2)
            row_block = idx[row]
            column_block = idx[column]

            # choose two numbers to switch
            row_value1 = randint(0, 2)
            column_value1 = randint(0, 2)

            row_value2 = randint(0, 2)
            column_value2 = randint(0, 2)

            # check if can alter values
            if (quiz[row_block + row_value1][column_block + column_value1] == 0 and quiz[row_block + row_value2][column_block + column_value2] == 0):
                # check if position 1 is different from position 2
                if ( row_value1 != row_value2 and column_value1 != column_value2):
                    cromo_temp[row_block + row_value1][column_block + column_value1], cromo_temp[row_block + row_value2][column_block +
                                                                                                                    column_value2] = cromo_temp[row_block + row_value2][column_block + column_value2], cromo_temp[row_block + row_value1][column_block + column_value1]
                    done = True
        
        return cromo_temp
    return cromo


""" (!) Use only with variant1. Mutate a candidate by picking a row, and then picking two values within that row to swap. """
def swap_mutation(cromo, prob_muta, quiz):
    done = False
    if random() < prob_muta:

        cromo_temp = copy.deepcopy(cromo)

        while (not done):
            
            row = randint(0, 8)
            columns = sample(range(0, 8), 2)

            # check if allowed to switch places
            if (quiz[row][columns[0]] == 0 and quiz[row][columns[1]] == 0):
                # swap values
                value1 = cromo_temp[row][columns[0]]
                value2 = cromo_temp[row][columns[1]]
                cromo_temp[row][columns[0]] = value2
                cromo_temp[row][columns[1]] = value1
                done = True

        return cromo_temp

    else:
        return cromo

""" (!) Use only with variant1. Mutate a candidate by picking a row, and then shuffling its values except the given values. """
def scramble_muta(cromo, prob_muta, quiz):
    if random() < prob_muta:
        # select a row
        selected_row = randint(0, 8)
        row_values = []

        # go through each value in row and add the value if it is not given (=0)
        for i in range(len(cromo[selected_row])):
            if quiz[selected_row][i] == 0:
                row_values.append(cromo[selected_row][i])

        shuffle(row_values)

        for i in range(len(cromo[selected_row])):
            if quiz[selected_row][i] == 0:
                cromo[selected_row][i] = row_values[0]
                row_values.pop(0)

    return cromo

""" (!) Use only with variant2. Scramble values in a random block. """
def scramble_in_block(cromo, prob_muta, quiz):
    if random() < prob_muta:
        cromo_temp = copy.deepcopy(cromo)
        # choose random block
        idx = [0, 3, 6]
        row = randint(0, 2)
        column = randint(0, 2)
        row_block = idx[row]
        column_block = idx[column]

        values = []

        if quiz[row_block][column_block] == 0:
            values.append(cromo[row_block][column_block])

        if quiz[row_block][column_block + 1] == 0:
            values.append(cromo[row_block][column_block + 1])

        if quiz[row_block][column_block + 2] == 0:
            values.append(cromo[row_block][column_block + 2])

        if quiz[row_block + 1][column_block] == 0:
            values.append(cromo[row_block + 1][column_block])

        if quiz[row_block + 1][column_block + 1] == 0:
            values.append(cromo[row_block + 1][column_block + 1])

        if quiz[row_block + 1][column_block + 2] == 0:
            values.append(cromo[row_block + 1][column_block + 2])

        if quiz[row_block + 2][column_block] == 0:
            values.append(cromo[row_block + 2][column_block])

        if quiz[row_block + 2][column_block + 1] == 0:
            values.append(cromo[row_block + 2][column_block + 1])

        if quiz[row_block + 2][column_block + 2] == 0:
            values.append(cromo[row_block + 2][column_block + 2])

        shuffle(values)

        if quiz[row_block][column_block] == 0:
            cromo_temp[row_block][column_block] = values[0]
            values.pop(0)

        if quiz[row_block][column_block + 1] == 0:
            cromo_temp[row_block][column_block + 1] = values[0]
            values.pop(0)

        if quiz[row_block][column_block + 2] == 0:
            cromo_temp[row_block][column_block + 2] = values[0]
            values.pop(0)

        if quiz[row_block + 1][column_block] == 0:
            cromo_temp[row_block + 1][column_block] = values[0]
            values.pop(0)

        if quiz[row_block + 1][column_block + 1] == 0:
            cromo_temp[row_block + 1][column_block + 1] = values[0]
            values.pop(0)

        if quiz[row_block + 1][column_block + 2] == 0:
            cromo_temp[row_block + 1][column_block + 2] = values[0]
            values.pop(0)

        if quiz[row_block + 2][column_block] == 0:
            cromo_temp[row_block + 2][column_block] = values[0]
            values.pop(0)

        if quiz[row_block + 2][column_block + 1] == 0:
            cromo_temp[row_block + 2][column_block + 1] = values[0]
            values.pop(0)

        if quiz[row_block + 2][column_block + 2] == 0:
            cromo_temp[row_block + 2][column_block + 2] = values[0]
            values.pop(0)
        
        return cromo_temp

    return cromo

# ==================== AUXILIARY =====================================================
def best_pop(populacao):
    populacao.sort(key=itemgetter(1), reverse=True)
    return populacao[0]

def average_pop(populacao):
    return sum([fit for cromo, fit in populacao])/len(populacao)

''' Check if there's a duplicate in row. '''
def is_row_duplicate(row, value):
    return value in row

''' Check if there's a duplicate in column. '''
def is_column_duplicate(indiv, column, value):
    for i in range(9):
        if indiv[i][column] == value:
            return True
    return False

def print_board(board):
    for row in range(0, len(board)):
        for column in range(0, len(board[row])):
            print(board[row][column], end=' ')
        print('')

    print('---------')

def compute_score(cromo):
    row_scores = []
    column_scores = []

    for row in range(0, len(cromo), 3):
        score = 0
        for i in range(3):
            score += len(set(cromo[row + i]))
        row_scores.append(score)
    
    column_arrays = []
    for column in range(len(cromo)):
        column_array = []
        for row in range(len(cromo)):
            column_array.append(cromo[row][column])
        column_arrays.append(column_array)

        if column == 2 or column == 5 or column == 8:
            score = 0
            for i in range(3):
                score += len(set(column_arrays[i]))
            column_scores.append(score)
            column_arrays = []

    return row_scores, column_scores
