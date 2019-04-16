from simulation import CirclePlayers
import numpy as np

if __name__ == '__main__':
    output = open('simulation-result/simu6_error_bar.txt', 'w+')
    N = 6
    beta = 1.
    num_iter, num_sample = 100000, 100

    for strategy in ['db']:
        for r in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]:
            payoff = payoff = {'A:A': r, 'A:B': r, 'B:A':1, 'B:B':1}

            for i in [1, 2]:
                time_cost, success, step_con, step_uncon = 0, [], [], []
                for it in range(num_sample):
                    game = CirclePlayers(payoff, N, beta, [0, i])
                    temp_tc, temp_success, temp_sc, temp_su = game.play(num_iter, func=strategy)
                    time_cost += temp_tc
                    success.append(temp_success)
                    step_con.append(temp_sc)
                    step_uncon.append(temp_su)

                print(time_cost, success, step_con, step_uncon)
                success = np.array(success)
                step_con = np.array(step_con)
                step_uncon = np.array(step_uncon)

                output.write('model: ' + strategy + '\n')
                output.write('N = %d, r = %.1f, d = %d, time cost = ' % (N, r, i-1) + str(time_cost))
                output.write(' / repeated %d times\n' % (num_iter, ))
                output.write('%d samples are taken\n' % (num_sample, ))

                output.write('success: mean=%f, max=%f, min=%f, diff=%f, var=%f\n' % (success.mean(), success.max(), success.min(), success.max()-success.min(), success.var()))
                output.write('conditional step: mean=%f, max=%f, min=%f, diff=%f, var=%f\n' % (step_con.mean(), step_con.max(), step_con.min(), step_con.max()-step_con.min(), step_con.var()))
                output.write('unconditional step: mean=%f, max=%f, min=%f, diff=%f, var=%f\n' % (step_uncon.mean(), step_uncon.max(), step_uncon.min(), step_uncon.max()-step_uncon.min(), step_uncon.var()))
                output.write('\n\n')
