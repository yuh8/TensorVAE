import deepchem as dc
import pickle
import numpy as np
from copy import deepcopy
from src.data_process_utils import mol_to_tensor
from src.misc_utils import create_folder
from src.running_stats import RunningStats


def get_and_save_data_batch(dataset, dest_data_path):
    tasks = dataset.tasks.tolist()
    task_idx = [tasks.index('homo'), tasks.index('lumo'), tasks.index('gap')]
    rs_homo_lumo_gap = RunningStats(len(task_idx))
    batch = 0
    for idx, mol in enumerate(dataset.X):
        try:
            g, _ = mol_to_tensor(deepcopy(mol))
        except Exception as e:
            # draw_mol_with_idx(mol)
            print(e)
            continue
        y = dataset.y[idx][task_idx]
        rs_homo_lumo_gap.push(y)

        np.savez_compressed(dest_data_path + f'GDR_{batch}', G=g, Y=y)
        batch += 1
        if batch % 100 == 0:
            mean_homo_lumo_gap = rs_homo_lumo_gap.mean()
            stdev_homo_lumo_gap = rs_homo_lumo_gap.standard_deviation()
            breakpoint()
            with open(dest_data_path + 'stats.pkl', 'wb') as f:
                pickle.dump(np.vstack([mean_homo_lumo_gap,
                                       stdev_homo_lumo_gap]), f)
    mean_homo_lumo_gap = rs_homo_lumo_gap.mean()
    stdev_homo_lumo_gap = rs_homo_lumo_gap.standard_deviation()
    with open(dest_data_path + 'stats.pkl', 'wb') as f:
        pickle.dump(np.vstack([mean_homo_lumo_gap,
                               stdev_homo_lumo_gap]), f)


if __name__ == "__main__":
    tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='Raw', splitter='Scaffold')
    create_folder('D:/transvae_qm9_prop/train_data/train_batch/')
    create_folder('D:/transvae_qm9_prop/test_data/test_batch/')
    create_folder('D:/transvae_qm9_prop/test_data/val_batch/')
    get_and_save_data_batch(datasets[0], 'D:/transvae_qm9_prop/train_data/train_batch/')
    get_and_save_data_batch(datasets[1], 'D:/transvae_qm9_prop/test_data/val_batch/')
    get_and_save_data_batch(datasets[2], 'D:/transvae_qm9_prop/test_data/test_batch/')
