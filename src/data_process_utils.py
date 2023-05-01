import numpy as np
from .graph_utils import mol_to_extended_graph
from .CONSTS import (BOND_RINGTYPE_SIZE,
                     MAX_NUM_ATOMS, FEATURE_DEPTH,
                     CHARGES, ATOM_LIST, ATOM_CHIR_NAMES,
                     BOND_NAMES, BOND_STEREO_NAMES, RING_SIZES,
                     ATOM_TYPE_SIZE, CHARGE_TYPE_SIZE, CHIR_TYPE_SIZE,
                     BOND_TYPE_SIZE, BOND_STEREOTYPE_SIZE)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1000)


def get_atom_channel_feature(element, charge, chir):
    # element
    atom_idx = ATOM_LIST.index(element)
    atom_type_channel = np.zeros(ATOM_TYPE_SIZE)
    atom_type_channel[atom_idx] = 1

    # charge
    charge_idx = CHARGES.index(charge)
    charge_type_channel = np.zeros(CHARGE_TYPE_SIZE)
    charge_type_channel[charge_idx] = 1

    # chiral
    chir_idx = ATOM_CHIR_NAMES.index(chir)
    chir_type_channel = np.zeros(CHIR_TYPE_SIZE)
    chir_type_channel[chir_idx] = 1

    return np.hstack([atom_type_channel,
                      charge_type_channel,
                      chir_type_channel])


def get_bond_channel_feature(bond_idx=None,
                             stereo_idx=None,
                             ring_indices=[],
                             kind=1, neighbor_len=1):

    bond_type_channel = np.zeros(BOND_TYPE_SIZE)
    bond_stereo_channel = np.zeros(BOND_STEREOTYPE_SIZE)
    bond_ring_channel = np.zeros(BOND_RINGTYPE_SIZE)
    if bond_idx is not None:
        bond_type_channel[bond_idx] = 1
        bond_stereo_channel[stereo_idx] = 1

        for ring_idx in ring_indices:
            bond_ring_channel[ring_idx] = 1

        bond_type_channel[-1] = kind / neighbor_len

    return np.hstack([bond_type_channel,
                      bond_stereo_channel,
                      bond_ring_channel])


def get_node_feature(mol, node_idx):
    node_feat = mol.GetAtomWithIdx(node_idx)
    element = node_feat.GetSymbol()
    charge = node_feat.GetFormalCharge()
    chir = node_feat.GetChiralTag()
    return get_atom_channel_feature(element, charge, chir)


def get_edge_feature(mol, source_idx, sink_idx, kind, max_neighbor_len):
    if kind == 1:
        bond = mol.GetBondBetweenAtoms(source_idx, sink_idx)
        bond_name = bond.GetBondType()
        bond_idx = BOND_NAMES.index(bond_name)

        # stereo type
        stereo_name = bond.GetStereo()
        stereo_idx = BOND_STEREO_NAMES.index(stereo_name)

        # ring size
        ring_indices = [idx + 1 for idx,
                        size in enumerate(RING_SIZES) if bond.IsInRingSize(size)]
        if len(ring_indices) == 0:
            ring_indices = [0]

        bond_channel_feature = get_bond_channel_feature(bond_idx, stereo_idx,
                                                        ring_indices,
                                                        neighbor_len=max_neighbor_len)

    else:
        bond_channel_feature = get_bond_channel_feature(bond_idx=-2, stereo_idx=0,
                                                        ring_indices=[0],
                                                        kind=kind, neighbor_len=max_neighbor_len)

    conf = mol.GetConformer(0)
    pos_ii = conf.GetAtomPosition(source_idx)
    pos_jj = conf.GetAtomPosition(sink_idx)
    dist = np.linalg.norm(pos_ii - pos_jj)

    return bond_channel_feature, dist


def mol_to_tensor(mol, infer=False):
    smi_graph = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4))
    R = np.zeros((MAX_NUM_ATOMS, 3))
    conf = mol.GetConformer(0)
    graph, max_neighbor_len = mol_to_extended_graph(mol)
    dfs_nodes = list(graph.nodes)

    for new_idx, node in enumerate(graph.nodes):
        node_feature = get_node_feature(mol, node)
        smi_graph[new_idx, new_idx, :len(node_feature)] = node_feature
        smi_graph[new_idx, new_idx, -3:] = conf.GetAtomPosition(node)
        R[new_idx, :] = conf.GetAtomPosition(node)

    for (source_idx, sink_idx) in graph.edges:
        kind = graph.edges[(source_idx, sink_idx)]['kind']
        edge_feature, dist = get_edge_feature(
            mol, source_idx, sink_idx, kind, max_neighbor_len)
        node_feature = get_node_feature(
            mol, source_idx) + get_node_feature(mol, sink_idx)
        new_source_idx = dfs_nodes.index(source_idx)
        new_sink_idx = dfs_nodes.index(sink_idx)
        smi_graph[new_source_idx, new_sink_idx,
                  :len(node_feature)] = node_feature
        smi_graph[new_sink_idx, new_source_idx,
                  :len(node_feature)] = node_feature

        smi_graph[new_source_idx, new_sink_idx,
                  len(node_feature):-4] = edge_feature
        smi_graph[new_sink_idx, new_source_idx,
                  len(node_feature):-4] = edge_feature
        smi_graph[new_source_idx, new_sink_idx, -4] = dist
        smi_graph[new_sink_idx, new_source_idx, -4] = dist

    if infer:
        return smi_graph, list(graph.nodes)

    return smi_graph, R
