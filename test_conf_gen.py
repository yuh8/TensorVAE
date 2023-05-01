from src.CONSTS import HIDDEN_SIZE, MAX_NUM_ATOMS, TF_EPS
from src.misc_utils import pickle_load, pickle_save, create_folder
from src.graph_utils import draw_smiles
from src.data_process_utils import mol_to_tensor
from rdkit.Chem.rdMolAlign import GetBestRMS
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Geometry import Point3D
from multiprocessing import freeze_support
from copy import deepcopy
import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import numpy as np
from random import shuffle
import pickle
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import matplotlib.pyplot as plt

tfd = tfp.distributions


def load_models():
    g_net = tf.keras.models.load_model('g_net/GNet/')
    gr_net = tf.keras.models.load_model('gr_net/GDRNet/')
    decoder_net = tf.keras.models.load_model('dec_net/DecNet/')
    return g_net, decoder_net, gr_net


def loss_func(y_true, y_pred):
    comp_weight, mean, log_std = tf.split(y_pred, 3, axis=-1)
    comp_weight = tf.nn.softmax(comp_weight, axis=-1)
    log_y_true = tf.math.log(y_true + TF_EPS)
    dist = tfd.Normal(loc=mean, scale=tf.math.exp(log_std))
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, NUM_COMPS]
    _loss = comp_weight * dist.prob(log_y_true)
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS]
    _loss = tf.reduce_sum(_loss, axis=-1)
    _loss = tf.math.log(_loss + TF_EPS)
    mask = tf.squeeze(tf.cast(y_true > 0, tf.float32))
    _loss *= mask
    loss = -tf.reduce_sum(_loss, axis=[1, 2])
    return loss


def get_best_RMSD(probe, ref, prbid=-1, refid=-1):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = GetBestRMS(probe, ref, prbid, refid)
    return rmsd


def unshuffle(r_pred, sequence):
    new_r_pred = np.zeros_like(r_pred)
    for idx, seq in enumerate(sequence):
        new_r_pred[seq] = r_pred[idx]
    return new_r_pred


def get_prediction(mol, sample_size, g_net, decoder_net):
    mol_origin = deepcopy(mol)
    gr, sequence = mol_to_tensor(mol_origin, infer=True)
    g = np.expand_dims(gr, axis=0)[..., :-4]
    mask = np.sum(np.abs(g), axis=-1)
    mask = np.sum(mask, axis=1, keepdims=True) <= 0
    mask = np.expand_dims(mask, axis=1).astype(np.float32)
    with tf.device('/cpu:0'):
        h = g_net.predict(g)
    h = np.tile(h, [sample_size, 1, 1])
    mask = np.tile(mask, [sample_size, 1, 1, 1])
    z = np.random.normal(0, 1, size=(sample_size, MAX_NUM_ATOMS, HIDDEN_SIZE))
    with tf.device('/cpu:0'):
        r_pred = decoder_net.predict(
            [h, mask, z]) * (1 - np.squeeze(mask)[..., np.newaxis])

    new_r_preds = []
    for _r_pred in r_pred:
        new_r_preds.append(unshuffle(_r_pred, sequence))
    return new_r_preds


def get_mol_probs(mol_pred, r_pred, num_gens, FF=True):
    mol_probs = []
    for j in range(num_gens):
        mol_prob = deepcopy(mol_pred)
        for i in range(mol_prob.GetNumAtoms()):
            mol_prob.GetConformer(0).SetAtomPosition(i, r_pred[j][i].tolist())
        if FF:
            MMFFOptimizeMolecule(mol_prob)
        mol_probs.append(mol_prob)
    return mol_probs


def get_conformation_samples(smiles_path, g_net, decoder_net):
    drugs_file = "/mnt/raw_data/rdkit_folder/summary_drugs.json"
    create_folder('./gen_samples/')
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    smiles = pickle_load(smiles_path)
    # shuffle(smiles)

    for idx, smi in enumerate(smiles[:1000]):
        try:
            mol_path = "/mnt/raw_data/rdkit_folder/" + \
                drugs_summ[smi]['pickle_path']
            with open(mol_path, "rb") as f:
                mol_dict = pickle.load(f)
        except:
            print('smiles missing')
            continue

        conf_df = pd.DataFrame(mol_dict['conformers'])
        conf_df.sort_values(by=['boltzmannweight'],
                            ascending=False, inplace=True)

        num_refs = conf_df.shape[0]

        if num_refs < 50:
            continue

        if num_refs > 100:
            continue

        num_gens = num_refs * 2

        mol_pred = deepcopy(conf_df.iloc[0].rd_mol)

        try:
            r_pred = get_prediction(mol_pred, num_gens, g_net, decoder_net)
        except:
            continue

        cov_mat = np.zeros((conf_df.shape[0], num_gens))

        cnt = 0
        gen_confs = []
        ref_confs = []
        try:
            mol_probs = get_mol_probs(mol_pred, r_pred, num_gens, FF=True)
            for _, mol_row in conf_df.iterrows():
                mol_ref = deepcopy(mol_row.rd_mol)
                for j in range(num_gens):
                    rmsd = get_best_RMSD(mol_probs[j], mol_ref)
                    cov_mat[cnt, j] = rmsd
                best_idx = np.argmin(cov_mat[cnt])
                gen_confs.append(mol_probs[best_idx])
                ref_confs.append(mol_ref)

                if len(gen_confs) == 5:
                    break
                cnt += 1
            if cov_mat[:5].min(-1).mean() < 0.7:
                draw_smiles(smi, f'./gen_samples/smi_{idx}')
                pickle_save(gen_confs, f'./gen_samples/gen_conf_{idx}.pkl')
                pickle_save(ref_confs, f'./gen_samples/ref_conf_{idx}.pkl')
        except:
            continue


def compute_cov_mat(smiles_path, g_net, decoder_net):
    drugs_file = "/mnt/raw_data/rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    smiles = pickle_load(smiles_path)

    cov_means = []
    cov_meds = []
    mat_means = []
    mat_meds = []

    covs = []
    mats = []
    for idx, smi in enumerate(smiles):
        try:
            mol_path = "/mnt/raw_data/rdkit_folder/" + \
                drugs_summ[smi]['pickle_path']
            with open(mol_path, "rb") as f:
                mol_dict = pickle.load(f)
        except:
            print('smiles missing')
            continue

        conf_df = pd.DataFrame(mol_dict['conformers'])
        conf_df.sort_values(by=['boltzmannweight'],
                            ascending=False, inplace=True)

        num_refs = conf_df.shape[0]

        if num_refs < 50:
            continue

        if num_refs > 100:
            continue

        num_gens = num_refs * 2

        mol_pred = deepcopy(conf_df.iloc[0].rd_mol)

        try:
            r_pred = get_prediction(mol_pred, num_gens, g_net, decoder_net)
        except:
            continue

        cov_mat = np.zeros((conf_df.shape[0], num_gens))

        cnt = 0
        try:
            mol_probs = get_mol_probs(mol_pred, r_pred, num_gens, FF=False)
            for _, mol_row in conf_df.iterrows():
                mol_ref = deepcopy(mol_row.rd_mol)
                for j in range(num_gens):
                    rmsd = get_best_RMSD(mol_probs[j], mol_ref)
                    cov_mat[cnt, j] = rmsd
                cnt += 1
        except:
            continue
        cov_score = (np.mean(cov_mat.min(-1) < 1.25),
                     np.mean(cov_mat.min(0) < 1.25))
        mat_score = (np.mean(cov_mat.min(-1)),
                     np.mean(cov_mat.min(0)))
        covs.append(cov_score)
        mats.append(mat_score)
        cov_mean = np.round(np.mean(covs, axis=0), 4)
        cov_med = np.round(np.median(covs, axis=0), 4)
        mat_mean = np.round(np.mean(mats, axis=0), 4)
        mat_med = np.round(np.median(mats, axis=0), 4)
        print(
            f'cov_mean_RP = {cov_mean}, cov_med_RP = {cov_med}, mat_mean_RP = {mat_mean}, mat_med_RP = {mat_med} for {idx} th mol')
        if len(covs) == 200:
            cov_means.append(cov_mean)
            cov_meds.append(cov_med)
            mat_means.append(mat_mean)
            mat_meds.append(mat_med)
            covs = []
            mats = []

        if len(cov_means) == 10:
            break

    cov_means_mean = np.round(np.mean(cov_means, axis=0), 4)
    cov_means_std = np.round(np.std(cov_means, axis=0), 4)
    cov_meds_mean = np.round(np.mean(cov_meds, axis=0), 4)
    cov_meds_std = np.round(np.std(cov_meds, axis=0), 4)

    mat_means_mean = np.round(np.mean(mat_means, axis=0), 4)
    mat_means_std = np.round(np.std(mat_means, axis=0), 4)
    mat_meds_mean = np.round(np.mean(mat_meds, axis=0), 4)
    mat_meds_std = np.round(np.std(mat_meds, axis=0), 4)
    print(f'cov_means_RP_mean = {cov_means_mean} with std {cov_means_std}')
    print(f'cov_meds_RP_mean = {cov_meds_mean} with std {cov_meds_std}')
    print(f'mat_means_RP_mean = {mat_means_mean} with std {mat_means_std}')
    print(f'mat_meds_RP_mean = {mat_meds_mean} with std {mat_meds_std}')


if __name__ == "__main__":
    freeze_support()
    g_net, decoder_net, _ = load_models()
    test_path = '/mnt/raw_data/transvae/test_data/test_batch/'
    compute_cov_mat(test_path + 'smiles.pkl', g_net, decoder_net)
    get_conformation_samples(test_path, g_net, decoder_net)
