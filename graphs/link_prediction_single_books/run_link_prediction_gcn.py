from experiment_link_prediction import *

network_names = ["hobbit","lotr", "silmarillion"]
if __name__ == '__main__':
    for network_name in network_names:
        results = run_experiment_link_prediction_GCN(
            d = 20,
            network_name= network_name,
            n_epochs = 15000,
            nfolds = 10
            )
        results = {k: '{0} ± {1}'.format(round(np.mean(v)*100, 2), round(np.std(v)*100, 2)) for k,v in results.items()}

        with open(f'linkpredGCN_{network_name}.json','w') as f:
            json.dump(results, f)