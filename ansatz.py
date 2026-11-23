"""Vision transformer wavefunction with spatial attention.

    psi(s) = phi(s) + phi(s)^* = 2 Re phi(s)  (real=True)  or  psi(s) = phi(s),   log phi(s) = sum_k log cosh(z_k(s))

The spins s (+-1) of the L x L lattice (row-major) are cut into n = (L/b)^2 patches of b x b spins, embedded in
R^d and passed through num_layers blocks  x <- x + A(LN(x)),  x <- x + FF(LN(x)).  The attention of head h is

    A(x)_i = sum_j K_h(r_j - r_i) V x_j,    K_h(r) = J_h(r) exp(-gamma_h |r|) / sum_r' exp(-gamma_h |r'|),

with |r| the minimum-image distance between patches. z = LN(W0 y) + i LN(W1 y), with y = LN(sum_i x_i).
"""
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn


def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def extract_patches(x, b):
    """(batch, L*L) -> (batch, n, b*b)."""
    batch = x.shape[0]
    n_side = int((x.shape[1] // b**2)**0.5)
    x = x.reshape(batch, n_side, b, n_side, b)
    x = x.transpose(0, 1, 3, 2, 4)
    return x.reshape(batch, n_side * n_side, b * b)


def patch_distances(n_side):
    """(n_side**2,) minimum-image distances of the patches from patch (0, 0)."""
    I, J = jnp.indices((n_side, n_side))
    dx = jnp.minimum(I, n_side - I)
    dy = jnp.minimum(J, n_side - J)
    return jnp.hypot(dx, dy).flatten()


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(K, i, j):
    """(h, n) -> (h, n_side, n_side, n): K rolled by (i, j) on the patch torus."""
    n_side = int(K.shape[-1]**0.5)
    K = K.reshape(K.shape[0], n_side, n_side)
    K = jnp.roll(jnp.roll(K, i, axis=-2), j, axis=-1)
    return K.reshape(K.shape[0], -1)


class SpatialAttention(nn.Module):
    d_model: int
    heads: int
    n: int
    dtype: Any

    def setup(self):
        n_side = int(self.n**0.5)
        assert n_side * n_side == self.n

        self.v = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)

        J = self.param("J", nn.initializers.xavier_uniform(), (self.heads, self.n), self.dtype)
        gamma = self.param("alpha", nn.initializers.constant(1.0), (self.heads, 1), self.dtype)
        K = J * jax.nn.softmax(-gamma * patch_distances(n_side).astype(self.dtype))
        self.K = roll2d(K, jnp.arange(n_side), jnp.arange(n_side)).reshape(self.heads, -1, self.n)

        self.W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)

    def __call__(self, x):
        v = rearrange(self.v(x), 'batch n (h d) -> batch h n d', h=self.heads)
        x = jnp.matmul(self.K, v)
        x = rearrange(x, 'batch h n d -> batch n (h d)')
        return self.W(x)


class EncoderBlock(nn.Module):
    d_model: int
    heads: int
    n: int
    dtype: Any

    def setup(self):
        self.attn = SpatialAttention(d_model=self.d_model, heads=self.heads, n=self.n, dtype=self.dtype)
        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.dtype)
        self.ff = nn.Sequential([
            nn.Dense(4 * self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype),
        ])

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    heads: int
    n: int
    dtype: Any

    def setup(self):
        self.layers = [EncoderBlock(d_model=self.d_model, heads=self.heads, n=self.n, dtype=self.dtype)
                       for _ in range(self.num_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Embed(nn.Module):
    d_model: int
    b: int
    dtype: Any

    def setup(self):
        self.embed = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)

    def __call__(self, s):
        return self.embed(extract_patches(s, self.b))


class OutputHead(nn.Module):
    d_model: int
    dtype: Any

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.dtype)
        self.norm2 = nn.LayerNorm(param_dtype=self.dtype)
        self.output_layer0 = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)
        self.norm3 = nn.LayerNorm(param_dtype=self.dtype)
        self.output_layer1 = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)

    def __call__(self, x):
        y = self.out_layer_norm(x.sum(axis=1))
        z = self.norm2(self.output_layer0(y)) + 1j * self.norm3(self.output_layer1(y))
        return jnp.sum(log_cosh(z), axis=-1)


class ViT(nn.Module):
    """log psi of a batch of spin configurations: state.spins (batch, L*L) -> (batch,) complex."""
    num_layers: int
    d_model: int
    heads: int
    L: int
    b: int
    real: bool = True
    dtype: Any = jnp.float64

    def setup(self):
        n = (self.L // self.b)**2
        self.patches_and_embed = Embed(self.d_model, self.b, dtype=self.dtype)
        self.encoder = Encoder(num_layers=self.num_layers, d_model=self.d_model, heads=self.heads, n=n,
                               dtype=self.dtype)
        self.output = OutputHead(self.d_model, dtype=self.dtype)

    def __call__(self, state):
        x = self.patches_and_embed(state.spins)
        log_phi = self.output(self.encoder(x))
        if self.real:
            return log_phi + jnp.log(1. + jnp.exp(-2j * log_phi.imag))
        return log_phi
