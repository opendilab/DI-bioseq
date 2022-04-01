import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


def get_result_data(data_dir):
    result = {'score': [], 'cost': [], 'round': []}
    for fname in glob.glob(f'{data_dir}/res*.csv'):
        with open(fname) as f:
            next(f)
            data = pd.read_csv(f)

        rounds = data['round'].max() + 1
        max_score_per_round = [
            data['real_score'][data['round'] == r].max()
            for r in range(rounds)
        ]
        max_score_per_round = np.maximum.accumulate(max_score_per_round)
        score_cost = sorted(set(data['score_times']))
        result['score'].extend(max_score_per_round)
        result['cost'].extend(score_cost)
        result['round'].append(rounds)

    return result


def draw_results(data_dir_dict, fig_name=None):
    for name, data_dir in data_dir_dict.items():
        data = get_result_data(data_dir)
        auc = []
        start_idx = 0
        all_rounds = []
        for r in data['round']:
            end_idx = start_idx + r
            round = list(range(r))
            all_rounds.extend(round)
            score = data['score'][start_idx:end_idx]
            auc.append(metrics.auc(round, score / (r - 1)))
        auc = np.mean(auc)
        sns.lineplot(x=all_rounds, y=data['score'], label="{}(AUC:{:.2f})".format(name, auc))
    plt.grid()
    plt.legend()
    if fig_name is not None:
        plt.savefig(fig_name)
    plt.show()


if __name__ == "__main__":
    fig_name = "compare_results.png"
    names_and_dirs = {
        "Random": "results/gb1/random-linear-onehot",
        "PPO": "results/gb1/ppo_offpolicy-linear-onehot"
    }
    draw_results(names_and_dirs, fig_name)
