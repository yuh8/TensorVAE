import tensorflow as tf
from rdkit import Chem
TF_EPS = tf.keras.backend.epsilon()

# data gen
NUM_CONFS_PER_MOL = 5

# Bound features {Type:4, Stereo: 6}
BOND_DICT = Chem.rdchem.BondType.values
BOND_STEREO_DICT = Chem.rdchem.BondStereo.values
BOND_NAMES = list(BOND_DICT.values())[1:4]
BOND_NAMES.append(list(BOND_DICT.values())[12])
BOND_STEREO_NAMES = list(BOND_STEREO_DICT.values())
RING_SIZES = range(3, 10)


# Atom features
MAX_NUM_ATOMS = 69
ATOM_LIST = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'I', 'Na', 'B', 'Si', 'Se', 'K', 'Bi']
CHARGES = [-2, -1, 0, 1, 2, 3]

CHIR_DICT = Chem.rdchem.ChiralType.values
ATOM_CHIR_NAMES = list(CHIR_DICT.values())

# chemistry
ATOM_TYPE_SIZE = len(ATOM_LIST)
CHARGE_TYPE_SIZE = len(CHARGES)
CHIR_TYPE_SIZE = len(ATOM_CHIR_NAMES)
BOND_TYPE_SIZE = len(BOND_NAMES) + 2  # include 1 virtual bond and bond length
BOND_STEREOTYPE_SIZE = len(BOND_STEREO_NAMES)
BOND_RINGTYPE_SIZE = len(RING_SIZES) + 1
BOND_ISCONJ_SIZE = 2

FEATURE_DEPTH = ATOM_TYPE_SIZE + CHARGE_TYPE_SIZE + CHIR_TYPE_SIZE +\
    BOND_TYPE_SIZE + BOND_STEREOTYPE_SIZE + BOND_RINGTYPE_SIZE

# hps
BATCH_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DFF = HIDDEN_SIZE * 2
VAL_BATCH_SIZE = 16
MIN_KL_WEIGHT = 1e-4
MAX_KL_WEIGHT = 0.0256
MAX_EPOCH = 640
Q_PERIOD = 40  # 12496
CYCLE_PERIOD = 19525
