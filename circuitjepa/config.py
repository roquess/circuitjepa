"""Global constants for the CircuitJEPA experiments."""

N_DATA = 3
N_ANC = 2
N_QUBITS = 5
LATENT = (0, 1)
J = 1.0
H_FIELD = 1.0
STEPS = (1, 2, 3)

# Independent held-out test set: a data seed never used before (train 1234, test 5678).
# Fixed before any evaluation on it; the configuration was selected on the original
# test set (seed 5678) and is not re-tuned on this one. Train data, configuration and
# trained parameters are unchanged; only the test samples differ.
HELDOUT_TEST_DATA_SEED = 424242
