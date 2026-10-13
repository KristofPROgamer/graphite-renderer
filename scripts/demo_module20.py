from tune.optuna_search import run_optuna_search


def demo():
    summary = run_optuna_search({}, [], n_trials=4, out_dir="out/tune_debug")
    print("Demo finished:", summary)


if __name__ == '__main__':
    demo()
