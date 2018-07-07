import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
#from itertools import product
from functools import partial
import time
import sys

class Player(object):
    def __init__(self, strategy='A', left=None, right=None, payoff={}, beta=1):
        self.strategy = strategy
        self.left = left
        self.right = right
        self.payoff = payoff
        self.beta = beta

    def fitness(self):
        l = self.strategy + ':' + self.left.strategy
        r = self.strategy + ':' + self.right.strategy
        l, r = self.payoff[l], self.payoff[r]
        return 1. -self.beta + self.beta * (l + r)
        #return np.exp(self.beta * (l + r))

    def death_and_replace(self):
        l, r = self.left.fitness(), self.right.fitness()
        p_l = l / (l + r)
        ran = random.random()
        if ran < p_l:
            new_strategy = self.left.strategy
        else:
            new_strategy = self.right.strategy

        result = 0
        if self.strategy == new_strategy:
            result = 0
        elif self.strategy == 'A':
            result = -1
        else:
            result = 1

        self.strategy = new_strategy
        return result

    def replace_with(self, new_strategy):
        if self.strategy == new_strategy:
            result = 0
        elif self.strategy == 'A':
            result = -1
        else:
            result = 1

        self.strategy = new_strategy
        return result

class CirclePlayers(object):
    def __init__(self, payoff={}, N=10, beta=0.1, initial=[0]):
        self.payoff = payoff
        self.initial = initial
        self.N = N
        self.i_A = len(self.initial)
        self.beta = beta

        self.players = [Player('B', payoff=self.payoff, beta=self.beta) for _ in range(N)]
        for i in range(N):
            self.players[i].left = self.players[(i-1) % self.N]
            self.players[i].right = self.players[(i+1) % self.N]

        self.reset()
    
    def reset(self):
        for p in self.players:
            p.strategy = 'B'
        for i in self.initial:
            self.players[i].strategy = 'A'
        self.i_A = len(self.initial)

    def is_interconnected(self):
        count_A = 0
        pos = 0
        while self.players[pos].strategy <> 'A':
            pos += 1
        while self.players[(pos-1)%self.N].strategy == 'A':
            pos = (pos - 1) % self.N
        
        while self.players[pos].strategy == 'A':
            count_A += 1
            pos = (pos + 1) % self.N
        
        #print([self.players[i].strategy for i in range(self.N)])
        #print(count_A == self.i_A)
        return (count_A == self.i_A)

    def raw_play_db(self):
        re = random.randint(0, self.N - 1)
        self.i_A += self.players[re].death_and_replace()
    
    def raw_play_bd_overall_fitness(self):
        s = [self.players[i].strategy for i in range(self.N)]

        def fitness(N, i, a, b, w):
            return 1. - w + w * (a * (i - 1) + b * (N - i)) / (N - 1)

        def gitness(N, i, c, d, w):
            return 1. - w + w * (c * i + d * (N - i - 1)) / (N - 1)
        
        # get the overall fitness of each player
        f = []
        for i in range(self.N):
            if s[i] == 'A':
                f.append(fitness(self.N, self.i_A, self.payoff['A:A'], self.payoff['A:B'], self.beta))
            else:
                f.append(gitness(self.N, self.i_A, self.payoff['B:A'], self.payoff['B:B'], self.beta))
        
        # let f be the list of proportion of each player's fitness 
        # and choose one propotional to its fitness
        r = random.random()
        s_f = sum(f)
        for i in range(self.N):
            if i == 0:
                f[i] = f[i] / s_f
            else:
                f[i] = f[i] / s_f + f[i-1]
            if r < f[i]:
                r = i
                break
        
        # choose either r - 1 or r + 1 to be replaced by r's reproduction
        result = 0
        rr = random.random()
        if rr < 0.5:
            result = self.players[(r-1)%self.N].replace_with(self.players[r].strategy)
        else:
            result = self.players[(r+1)%self.N].replace_with(self.players[r].strategy)
        self.i_A += result

    def raw_play_bd_local_fitness(self):
        f = [self.players[i].fitness() for i in range(self.N)]
        
        # let f be the list of proportion of each player's fitness #
        # and choose one propotional to its fitness
        r = random.random()
        s_f = sum(f)
        for i in range(self.N):
            if i == 0:
                f[i] = f[i] / s_f
            else:
                f[i] = f[i] / s_f + f[i-1]
            if r < f[i]:
                r = i
                break
        
        # choose either r - 1 or r + 1 to be replaced by r's reproduction
        result = 0
        rr = random.random()
        if rr < 0.5:
            result = self.players[(r-1)%self.N].replace_with(self.players[r].strategy)
        else:
            result = self.players[(r+1)%self.N].replace_with(self.players[r].strategy)
        self.i_A += result

    def play(self, num_iterate=10000, func='db'):
        success, step_uncon, step_con = 0, 0, 0
        raw_play = None
        if func == 'db':
            raw_play = self.raw_play_db
        elif func == 'bd_overall':
            raw_play = self.raw_play_bd_overall_fitness
        elif func == 'bd_local':
            raw_play = self.raw_play_bd_local_fitness

        start = time.time()
        for _ in range(num_iterate):
            self.reset()
            step = 0
            while self.i_A <> 0 and self.i_A <> self.N:
                raw_play()
                step += 1
        
            step_uncon += step
            if self.i_A == self.N:
                success += 1
                step_con += step
            elif self.i_A == 0:
                pass
            else:
                print('Warning!')

            #sys.stdin.readline()
        end = time.time()
        time_cost = end - start

        if success == 0:
            step_con = 0
        else:
            step_con /= float(success)
        success /= float(num_iterate)
        step_uncon /= float(num_iterate)
        print(self.N, success, step_con, step_uncon)
        return time_cost, success, step_con, step_uncon
        #return time_cost, self.N*success, step_con, step_uncon

    def play_part(self, num_iterate=10000, func='db'):
        interconnect_time = [0 for _ in range(self.N - 1)]

        raw_play = None
        if func == 'db':
            raw_play = self.raw_play_db
        elif func == 'bd_overall':
            raw_play = self.raw_play_bd_overall_fitness
        elif func == 'bd_local':
            raw_play = self.raw_play_bd_local_fitness

        step = 0
        start = time.time()
        for _ in range(num_iterate):
            self.reset()
            while not self.is_interconnected():
                raw_play()
                step += 1
                #sys.stdin.readline()
            interconnect_time[self.i_A - 1] += 1

        end = time.time()
        time_cost = end - start

        step /= float(num_iterate)
        for i in range(self.N - 1):
            interconnect_time[i] /= float(num_iterate)
        print(self.N, self.initial, interconnect_time, step, time_cost)
        return interconnect_time, step, time_cost


if __name__ == '__main__':
    output = open('simu25.txt', 'w+')
    N = 25
    beta = 1.

    for strategy in ['db']:
        for r in [0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]:
            payoff = payoff = {'A:A': r, 'A:B': r, 'B:A':1, 'B:B':1}

            for i in [1, 2, 3, 12]:
                game = CirclePlayers(payoff, N, beta, [0, i])
                time_cost, success, step_con, step_uncon = game.play(10000, func=strategy)

                output.write('model: ' + strategy + '\n')
                output.write('N = %d, r = %.1f, d = %d, time cost = ' % (N, r, i-1) + str(time_cost))
                output.write(' / repeated %d times\n' % (10000, ))
                output.write('success = ' + str(success) + '\n')
                output.write('step_con = %d, step_uncon = %d' % (step_con, step_uncon) )
                output.write('\n\n')
