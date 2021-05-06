import matplotlib.pyplot as plt

def read_file(path):
    # total quizzes to read from file
    total_quizzes = 9

    quizzes = []
    solutions = []

    file = open(path, 'r').read()

    for i, line in enumerate(file.splitlines()[1: total_quizzes + 1]):

        new_quiz = []
        new_solution = []
        row_q = []
        row_s = []

        columns_counter = 0

        quiz, solution = line.split(",")

        for j, q_s in enumerate(zip(quiz, solution)):

            q, s = q_s
            row_q.append(int(q))
            row_s.append(int(s))
            columns_counter += 1

            if columns_counter%9 == 0:
                new_quiz.append(row_q)
                new_solution.append(row_s)
                row_q = []
                row_s = []
                columns_counter = 0

        quizzes.append(new_quiz)
        solutions.append(new_solution)

    return quizzes, solutions

def display(data_best, data_average):
    plt.title('EA: Sudoku')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    x = range(1, len(data_best) + 1)
    plt.plot(x, data_best, label='Best')
    plt.plot(x, data_average, label='Average')
    plt.legend(loc='best')
    plt.show()
