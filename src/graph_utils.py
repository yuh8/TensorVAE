import numpy as np
from networkx import Graph
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Draw import rdMolDraw2D
from typing import List


def draw_smiles(smi, file_name):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Draw.MolToFile(mol, '{}.png'.format(file_name))


def draw_mol_with_idx(mol):
    d = rdMolDraw2D.MolDraw2DCairo(960, 960)
    d.drawOptions().addAtomIndices = True
    d.DrawMolecule(mol)
    d.FinishDrawing()
    with open('atom_annotation_1.png', 'wb') as f:
        f.write(d.GetDrawingText())


def mol_to_extended_graph(molecule: Mol, seed: int = 0) -> Graph:
    mol_len = molecule.GetNumAtoms()
    rng = np.random.default_rng(seed=seed)
    start = rng.integers(low=0, high=mol_len, size=1).item()
    bond_graph = build_bond_graph(molecule)
    visited = set()
    dfs_traversal = list()
    sequence = get_random_df_sequence(graph=bond_graph, start=start,
                                      visited=visited,
                                      dfs_traversal=dfs_traversal)

    assert mol_len == len(sequence), 'disconnected graph'
    assert np.max(sequence) == mol_len - 1, 'disconnected graph'

    graph = Graph()
    max_neighbor_len = -np.inf
    for new_node in sequence:
        neighbor_len = embed_node_in_graph(graph, new_node=new_node,
                                           bond_graph=bond_graph,
                                           max_dist=mol_len, rng=rng)
        if neighbor_len > max_neighbor_len:
            max_neighbor_len = neighbor_len

    return graph, max_neighbor_len


def build_bond_graph(molecule: Mol) -> Graph:
    graph = Graph()
    for bond in molecule.GetBonds():
        source_index, sink_index = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph.add_edge(source_index, sink_index)
    return graph


def embed_node_in_graph(graph, new_node: int, bond_graph: Graph, max_dist: int, rng: np.random.Generator) -> None:
    graph.add_node(new_node)
    bonded_neighborhood_list = get_neighborhoods(
        bond_graph, new_node, max_distance=max_dist)
    dfs_atoms = np.unique(np.hstack(bonded_neighborhood_list))

    # assure dfs traverse the entire graph
    assert max_dist == dfs_atoms.shape[0], 'disconnected graph'
    assert max_dist == np.max(dfs_atoms) + 1, 'disconnected graph'
    neighbor_len = len(bonded_neighborhood_list)

    for k in range(1, neighbor_len):
        neighborhood = bonded_neighborhood_list[k]
        rng.shuffle(neighborhood)

        for neighbor in neighborhood:
            if neighbor in graph.nodes and not graph.has_edge(new_node, neighbor):
                graph.add_edge(new_node, neighbor, kind=k)

    return neighbor_len


def get_neighborhoods(graph: Graph, source: int, max_distance: int) -> List[List[int]]:
    neighborhoods = [[source]]

    for k in range(1, max_distance + 1):
        new_neighborhood = []
        for front in neighborhoods[k - 1]:
            for neighbor in graph.neighbors(front):
                present = False
                for neighborhood in neighborhoods:
                    if neighbor in neighborhood or neighbor in new_neighborhood:
                        present = True
                        break

                if not present:
                    new_neighborhood.append(neighbor)

        if not new_neighborhood:
            break

        neighborhoods.append(new_neighborhood)

    return neighborhoods


def get_random_df_sequence(graph: Graph, start: int, visited: set, dfs_traversal: list) -> List[int]:
    if start not in visited:
        dfs_traversal.append(start)
        visited.add(start)

        for neighbor_node in graph.neighbors(start):
            get_random_df_sequence(graph, neighbor_node,
                                   visited, dfs_traversal)

    return dfs_traversal
