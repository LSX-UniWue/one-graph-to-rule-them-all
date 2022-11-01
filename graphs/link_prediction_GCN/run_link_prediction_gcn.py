from experiment_link_prediction_gcn import *


if __name__ == '__main__':
    results = run_experiment_link_prediction_GCN(
        d = 20,
        n_epochs = 15000,
        nfolds = 10
        )
    results = {k: '{0} ± {1}'.format(round(np.mean(v)*100, 2), round(np.std(v)*100, 2)) for k,v in results.items()}

    with open('linkpredGCN.json','w') as f:
        json.dump(results, f)