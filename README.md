# Scaling laws for neural-network quantum states

Trained transformer wavefunctions of

> R. Rende, A. Sinibaldi, L. L. Viteritti, R. Wiersema, A. Georges and G. Carleo,
> *Scaling Laws for Neural-Network Quantum States*, arXiv:2606.02794 (2026),

and the code to load and sample them: the models of Fig. 1, ground states of

    H = J1 sum_<ij> S_i.S_j + J2 sum_<<ij>> S_i.S_j

(first and second neighbours) on L x L tori, with J1 = 1:

- `parameters/square_heisenberg`: square lattice, J2 = 0;
- `parameters/square_j1j2`: square lattice, J2 = 0.5;
- `parameters/triangular_heisenberg`: triangular lattice, J2 = 0;
- `parameters/triangular_j1j2`: triangular lattice, J2 = 0.125.

Each model is a folder `L<L>_layers<n_l>` with `config.yaml` (Hamiltonian, architecture, E/N and V-score of the last
100 training steps), `params.msgpack` (the parameters) and `walkers.npz` (the 16384 Markov chains at the end of the
training, `up = np.packbits(spins == 1, axis=1)`).

## Install

    pip install "tachys[cuda13]==0.1.7"     # or tachys[cuda12]; plain tachys==0.1.7 runs on the CPU

Python >= 3.12. [tachys](https://github.com/tachys-dev/tachys) provides the lattice, the Hamiltonian and the sampler.

The parameters are on [Zenodo](https://doi.org/10.5281/zenodo.24476230); download them into this folder with

    curl -L "https://zenodo.org/records/24476230/files/parameters.tar.gz?download=1" | tar -xz

## Sample a model

    python main.py parameters/square_j1j2/L12_layers2
    python main.py parameters/square_j1j2/L20_layers8 --N_steps 100

The Markov chains start from the stored walkers. At every step (one sweep per chain) `main.py` prints E/N and the
V-score N Var(E)/E^2, and at the end their values over all samples, next to those of the last 100 training steps.

| option | default | |
|---|---|---|
| `--N_mc` | 16384 | Markov chains, at most 16384 (the largest models need about 20 GB of GPU memory) |
| `--N_steps` | 10 | samples per chain |
| `--seed` | 0 | |

On several GPUs, start one process per GPU: `srun python main.py ...`.

## Use a model in Python

```python
from main import load_model
from tachys.lattice.spins.spin_state import SpinState

cfg, lattice, H, wf, walkers = load_model('parameters/square_j1j2/L12_layers2')
log_psi = wf.apply_fn(wf.params, SpinState(spins=walkers[:8], lattice=lattice))    # (8,) complex
```

The spins are +-1, and site k = i L + j sits at j a1 + i a2, with a1 = (1, 0) and a2 = (0, 1) on the square and
(1/2, sqrt(3)/2) on the triangular lattice.

## Wavefunction

`ansatz.ViT` is the transformer of the paper (Methods): b x b patches (b = 4 on the square, 3 on the triangular
lattice), embedding dimension 72, 12 heads of spatial attention (Eq. 5) and log phi = sum_k log cosh z_k. The
wavefunction is psi = 2 Re phi on the square lattice and psi = phi on the triangular lattice.

## Disclaimers

- We provide one trained model (a single random seed) per system size and number of layers, while the data in the
  paper take into account several seeds. Do not expect to reproduce the values of the paper exactly with these models.
- In this repository, the spatial attention on the triangular lattice does not use the triangular distances: the
  distance between two patches is computed from their indices as on a square lattice, (dx^2 + dy^2)^(1/2), with dx
  and dy the minimum-image index differences. This does not change the results.

## Citation

```bibtex
@article{rende2026scaling,
  title   = {Scaling Laws for Neural-Network Quantum States},
  author  = {Rende, Riccardo and Sinibaldi, Alessandro and Viteritti, Luciano Loris and Wiersema, Roeland and
             Georges, Antoine and Carleo, Giuseppe},
  journal = {arXiv:2606.02794},
  year    = {2026}
}
```
