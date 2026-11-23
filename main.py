"""Samples |psi|^2 of a trained model and prints E/N, Var(E)/N and the V-score N Var(E) / E^2.

    python main.py parameters/square_j1j2/L12_layers2
    python main.py parameters/square_j1j2/L12_layers2 --N_steps 100

On several GPUs, start one process per GPU (srun python main.py ..., mpirun -np 4 python main.py ...).
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization

from ansatz import ViT
from tachys.lattice.bond_exchange import BondExchange
from tachys.lattice.lattice_database import square, triangular
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_hamiltonian
from tachys.lattice.spins.spin_state import SpinState
from tachys.montecarlo import sample
from tachys.wavefunction import WaveFunction

# lattice, first- and second-neighbour displacements in units of a1 and a2
GEOMETRY = {'square': (square, [(1, 0), (0, 1)], [(1, 1), (1, -1)]),
            'triangular': (triangular, [(1, 0), (0, 1), (1, -1)], [(1, 1), (2, -1), (1, -2)])}


def load_model(folder):
    """config (dict), lattice, Hamiltonian, WaveFunction and stored walkers of the model in `folder`.

    walkers: (16384, L*L) int8 spins (+-1), the training's Markov chains at its last step.
    Site k = i*L + j sits at j*a1 + i*a2 of the L x L torus.
    """
    with open(os.path.join(folder, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    h, m = cfg['hamiltonian'], cfg['model']
    L = h['L']

    make_lattice, nn1, nn2 = GEOMETRY[h['lattice']]
    lattice = make_lattice(shape=(L, L))
    bonds = [(d, h['J1']) for d in nn1]
    if h['J2'] != 0:
        bonds += [(d, h['J2']) for d in nn2]
    H = heisenberg_hamiltonian(lattice, nn=bonds)

    model = ViT(num_layers=m['num_layers'], d_model=m['d_model'], heads=m['heads'], L=L, b=m['b'], real=m['real'])
    with open(os.path.join(folder, 'params.msgpack'), 'rb') as f:
        params = serialization.msgpack_restore(f.read())
    wf = WaveFunction(params=jax.tree.map(jnp.asarray, params), apply_fn=model.apply)

    up = np.load(os.path.join(folder, 'walkers.npz'))['up']
    walkers = 2 * np.unpackbits(up, axis=1, count=L * L).astype(np.int8) - 1
    return cfg, lattice, H, wf, walkers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help='a model folder, e.g. parameters/square_j1j2/L12_layers2')
    parser.add_argument('--N_mc', type=int, default=16384, help='Markov chains, started from the stored walkers')
    parser.add_argument('--N_steps', type=int, default=10, help='samples per chain, one sweep apart')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg, lattice, H, wf, walkers = load_model(args.model)
    N = lattice.Ns
    state = SpinState(spins=jnp.asarray(walkers[:args.N_mc]), lattice=lattice)
    action = BondExchange.create(lattice, max_dist=2)     # exchanges two opposite spins, first or second neighbours

    key = jax.random.key(args.seed)
    E, E2 = [], []
    for t in range(args.N_steps):
        key, subkey = jax.random.split(key)
        state, log_amps, acceptance = sample(1, state, action, jax.random.split(subkey, args.N_mc), wf)
        _, E_mean, E2_mean = compute_expectation(H, wf, state, log_amps)
        E.append(float(E_mean.real))
        E2.append(float(E2_mean))
        print(f"{t:5d} | E/N {E[-1] / N:.8f} | V-score {N * (E2[-1] - E[-1]**2) / E[-1]**2:.4e} "
              f"| acceptance {float(acceptance[0]):.3f}", flush=True)

    E_mean, var = np.mean(E), np.mean(E2) - np.mean(E)**2
    ref = cfg['reference']
    print(f"E/N = {E_mean / N:.8f} +- {np.std(E) / np.sqrt(args.N_steps) / N:.8f} | Var(E)/N = {var / N:.4e} "
          f"| V-score = {N * var / E_mean**2:.4e}")
    print(f"training, last 100 steps: E/N = {ref['energy']:.8f} | Var(E)/N = {ref['variance']:.4e} "
          f"| V-score = {ref['vscore']:.4e}")


if __name__ == '__main__':
    main()
