import numpy as np
import pickle
import datetime
import warnings


import sys
import os

# Add the path to the cloned NPEET repository
script_dir = os.path.dirname(os.path.abspath(__file__))
npeet_path = os.path.join(script_dir, 'NPEET')
if os.path.isdir(npeet_path):
    sys.path.append(npeet_path)

# Add the project's source directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from dask.delayed import delayed
from dask.base import compute
from dask.diagnostics.progress import ProgressBar

from symbolic_regression.methods.gp import GP
from symbolic_regression.methods.gpshap import GPSHAP
from symbolic_regression.methods.gpcmi import GPCMI
from symbolic_regression.methods.rfgp import RFGP

from symbolic_regression.utils.pysr_utils import nrmse_loss, train_val_test_split, process_task
from symbolic_regression.datasets import load_datasets

warnings.filterwarnings("ignore", category=UserWarning, module="pysr")
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    n_runs = 4
    test_size = 0.2
    val_size = 0.25
    n_top_features = None
    ns = 100
    ci = 0.99
    k = 5
    record_interval = 5
    n_submodels = 2

    pysr_params = {
        "populations": 1,
        "population_size": 20,
        "niterations": 20,
        "binary_operators": ["+", "-", "*"],
        "unary_operators": ["sqrt", "inv(x) = 1/x"],
        "extra_sympy_mappings": {"inv": lambda x: 1/x},
        "verbosity": 0,
    }

    dataset_names = [
        "F1",
        # "F2",
        # ("4544_GeographicalOriginalofMusic", "4544_GOM"),
        # "505_tecator",
        # ("Communities and Crime", "CCN"),
        # ("Communities and Crime Unnormalized", "CCUN"),
    ]
    datasets = load_datasets(dataset_names)

    gp_params = {
        "loss_function": nrmse_loss,
        "record_interval": record_interval,
        **pysr_params,
    }

    gpshap_params = {
        **gp_params
    }

    gpcmi_params = {
        "ns": ns,
        "ci": ci,
        "k": k,
        **gp_params
    }

    shap_params = {
        "test_size": test_size,
        "val_size": val_size,
        "n_runs": n_runs,
        "n_top_features": n_top_features,
        **gp_params,
    }

    rfgpcmi_params = {
        "n_submodels": n_submodels,
        "method_class": GPCMI,
        "method_params": gpcmi_params
    }

    gp = GP(**gp_params)
    gpshap = GPSHAP(**gpshap_params)
    gpcmi = GPCMI(**gpcmi_params)
    rfgpcmi = RFGP(**rfgpcmi_params)

    methods = {
        # "GP": gp,
        # "GPSHAP": gpshap,
        "GPCMI": gpcmi,
        "RFGPCMI": rfgpcmi,
    }

    width_method = max([round(len(name), 0) for name in methods.keys()])
    width_dataset = max([round(len(name), 0) for name in datasets.keys()])
    n_records = methods[list(methods.keys())[0]].n_records

    # NOTE: The following code is for parallel processing of tasks across datasets and methods.

    delayed_tasks = {}
    if "GPSHAP" in methods:
        methods["GPSHAP"].clear_cache()  # Clear cache before starting new task

    for dataset_name, dataset in datasets.items():
        delayed_tasks[dataset_name] = {}

        X = dataset["X"]
        y = dataset["y"]

        delayed_splits = [delayed(train_val_test_split)(X, y) for _ in range(n_runs)]
        
        for method_name, method in methods.items():
            delayed_tasks[dataset_name][method_name] = []

            if method_name == "GPSHAP": continue  # Skip GPSHAP for now, as it requires precomputed features

            # Create a delayed task for each method and dataset
            for run in range(n_runs):
                delayed_tasks[dataset_name][method_name].append(
                    delayed(process_task)(dataset_name, method_name, run, delayed_splits[run], method)
                )
        
        if ("GPSHAP" in methods):
            if ('GP' in methods):
                delayed_X_trains = tuple([delayed_split[0] for delayed_split in delayed_splits])
                delayed_gp_equations = [delayed_task['equations'][-1] for delayed_task in delayed_tasks[dataset_name]["GP"]]
                
                # Use GP's equations for GPSHAP
                delayed_precomputed_features_task = delayed(methods["GPSHAP"].precompute_features_from_pretrained_models)(
                    delayed_X_trains, delayed_gp_equations, n_top_features
                )

            else:
                delayed_precomputed_features_task = delayed(methods["GPSHAP"].precompute_features)(
                    X, y, **shap_params
                )

            for run in range(n_runs):
                delayed_tasks[dataset_name]["GPSHAP"].append(
                    delayed(process_task)(dataset_name, 'GPSHAP', run, delayed_splits[run], methods["GPSHAP"], delayed_precomputed_features_task)
                )

    tasks_to_run = [
        task 
        for methods_dict in delayed_tasks.values() 
        for task_list in methods_dict.values() 
        for task in task_list
    ]

    with ProgressBar():
        computed_results = compute(*tasks_to_run, scheduler='processes')

    results = {}
    equations = {}
    features = {}

    for dataset_name in datasets.keys():
        results[dataset_name] = {}
        equations[dataset_name] = {}
        features[dataset_name] = {}

        for method_name in methods.keys():
            results[dataset_name][method_name] = {
                "training_losses": np.empty((n_runs, n_records)),
                "validation_losses": np.empty((n_runs, n_records)),
                "test_losses": np.empty((n_runs, n_records)),
            }
            equations[dataset_name][method_name] = []
            features[dataset_name][method_name] = []

    for result in computed_results:
        dataset_name = result['dataset_name']
        method_name = result['method_name']
        run = result['run']
        
        results[dataset_name][method_name]["training_losses"][run, :] = result['losses'][0]
        results[dataset_name][method_name]["validation_losses"][run, :] = result['losses'][1]   
        results[dataset_name][method_name]["test_losses"][run, :] = result['losses'][2]
        
        equations[dataset_name][method_name].append(result['equations'])
        features[dataset_name][method_name].append(result['features'])

    data = {
        "results": results,
        "equations": equations,
        "features": features
    }
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    filename = f"data_{timestamp}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"\nResults saved to {filename}")

if __name__ == '__main__':
    import time
    start_time = time.time()

    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    