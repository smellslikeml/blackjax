# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for DMALA (the Discrete Langevin Proposal kernel).

The correctness gate compares the empirical distribution of the chain against
the exactly enumerated target on small binary distributions: a factorized
Bernoulli target and a correlated pairwise Ising model.
"""

import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.mcmc.discrete_langevin as discrete_langevin
from tests.fixtures import BlackJAXTest

BERNOULLI_PROBS = jnp.array([0.7, 0.3, 0.1, 0.65, 0.45])
ISING_FIELDS = jnp.array([0.3, -0.5, 0.8, 0.1, -0.2, 0.6])
ISING_COUPLINGS = jnp.array(
    [
        [0.0, 0.7, -0.4, 0.0, 0.5, 0.0],
        [0.7, 0.0, 0.6, 0.0, 0.0, -0.3],
        [-0.4, 0.6, 0.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.0, -0.5, 0.4],
        [0.5, 0.0, 0.0, -0.5, 0.0, 0.6],
        [0.0, -0.3, 0.0, 0.4, 0.6, 0.0],
    ]
)


def bernoulli_logdensity(x, probs):
    """Unnormalized log-density of independent Bernoulli(probs) coordinates."""
    return jnp.sum(x * jnp.log(probs) + (1 - x) * jnp.log1p(-probs))


def bernoulli07_logdensity(x):
    """A Bernoulli(0.7) target of any dimension, for API-level tests."""
    return bernoulli_logdensity(x, 0.7)


bernoulli5_logdensity = functools.partial(bernoulli_logdensity, probs=BERNOULLI_PROBS)


def ising_logdensity(x):
    """Unnormalized log-density of a pairwise Ising model on {0, 1}^6."""
    return jnp.dot(ISING_FIELDS, x) + jnp.sum(
        jnp.triu(ISING_COUPLINGS, 1) * x[:, None] * x[None, :]
    )


def enumerate_binary_states(dim):
    """All 2**dim binary states, in lexicographic (MSB-first) order."""
    return jnp.array(list(itertools.product((0.0, 1.0), repeat=dim)))


def binary_state_indices(samples, dim):
    """Index of each sample in the enumeration order of the binary hypercube."""
    powers = 2.0 ** jnp.arange(dim - 1, -1, -1)
    return (samples * powers).sum(axis=-1).astype(jnp.int32)


def exact_binary_distribution(logdensity_fn, dim):
    """Exact (normalized) distribution of a binary target, by enumeration."""
    states = enumerate_binary_states(dim)
    probabilities = jax.nn.softmax(jax.vmap(logdensity_fn)(states))
    return states, probabilities


def run_dmala_chains(algorithm, rng_key, initial_positions, num_steps, num_burn_in):
    """Run independent chains; return post-burn-in samples and acceptance rate."""
    initial_states = jax.vmap(algorithm.init)(initial_positions)
    chain_keys = jax.random.split(rng_key, initial_positions.shape[0])

    def chain(initial_state, chain_key):
        def step(state, key):
            state, info = algorithm.step(key, state)
            return state, (state.position, info.is_accepted)

        _, (positions, accepted) = jax.lax.scan(
            step, initial_state, jax.random.split(chain_key, num_steps)
        )
        return positions[num_burn_in:], accepted[num_burn_in:]

    positions, accepted = jax.vmap(chain)(initial_states, chain_keys)
    return positions.reshape(-1, positions.shape[-1]), accepted.mean()


# ---------------------------------------------------------------------------
# discrete_langevin.init
# ---------------------------------------------------------------------------


class DiscreteLangevinInitTest(BlackJAXTest):
    """Tests for discrete_langevin.init."""

    def test_init_computes_logdensity_and_grad(self):
        """init stores logdensity and logdensity_grad at the initial position."""
        position = jnp.array([1.0, 0.0])
        state = discrete_langevin.init(position, bernoulli07_logdensity)
        np.testing.assert_allclose(
            float(state.logdensity), float(bernoulli07_logdensity(position))
        )
        np.testing.assert_allclose(
            state.logdensity_grad, jax.grad(bernoulli07_logdensity)(position)
        )

    def test_init_pytree_position(self):
        """init works with PyTree positions."""

        def logdensity_fn(pos):
            return -0.5 * jnp.sum(pos["a"] ** 2) + jnp.sum(pos["b"])

        position = {"a": jnp.zeros(3), "b": jnp.zeros(2)}
        state = discrete_langevin.init(position, logdensity_fn)
        chex.assert_trees_all_equal_shapes(state.position, position)
        assert jnp.isfinite(state.logdensity)


# ---------------------------------------------------------------------------
# discrete_langevin.build_kernel / kernel
# ---------------------------------------------------------------------------


class DiscreteLangevinKernelTest(BlackJAXTest):
    """Tests for the DMALA kernel."""

    def setUp(self):
        super().setUp()
        self.step_size = 1.0

    def test_returns_state_and_info(self):
        """Kernel returns (DLPState, DLPInfo)."""
        position = jnp.zeros(4)
        state = discrete_langevin.init(position, bernoulli07_logdensity)
        kernel = discrete_langevin.build_kernel()
        new_state, info = kernel(
            self.next_key(), state, bernoulli07_logdensity, self.step_size
        )
        self.assertIsInstance(new_state, discrete_langevin.DLPState)
        self.assertIsInstance(info, discrete_langevin.DLPInfo)

    def test_output_position_shape(self):
        """Output position has same shape as input."""
        position = jnp.zeros(5)
        state = discrete_langevin.init(position, bernoulli07_logdensity)
        kernel = discrete_langevin.build_kernel()
        new_state, _ = kernel(
            self.next_key(), state, bernoulli07_logdensity, self.step_size
        )
        self.assertEqual(new_state.position.shape, (5,))

    def test_positions_stay_on_the_binary_grid(self):
        """Positions remain in {0, 1}^d however long the chain runs."""
        position = jnp.zeros(5)
        state = discrete_langevin.init(position, bernoulli07_logdensity)
        kernel = discrete_langevin.build_kernel()

        def step(state, key):
            state, _ = kernel(key, state, bernoulli07_logdensity, self.step_size)
            return state, state.position

        _, positions = jax.lax.scan(step, state, jax.random.split(self.next_key(), 500))
        assert jnp.all((positions == 0) | (positions == 1))

    def test_acceptance_rate_in_range(self):
        """Acceptance rate is in [0, 1]."""
        position = jnp.zeros(3)
        state = discrete_langevin.init(position, bernoulli07_logdensity)
        kernel = discrete_langevin.build_kernel()
        _, info = kernel(self.next_key(), state, bernoulli07_logdensity, self.step_size)
        assert 0.0 <= float(info.acceptance_rate) <= 1.0

    def test_logdensity_updated(self):
        """Stored logdensity is consistent with the accepted position."""
        position = jnp.zeros(2)
        state = discrete_langevin.init(position, bernoulli07_logdensity)
        kernel = discrete_langevin.build_kernel()
        new_state, _ = kernel(
            self.next_key(), state, bernoulli07_logdensity, self.step_size
        )
        np.testing.assert_allclose(
            float(new_state.logdensity),
            float(bernoulli07_logdensity(new_state.position)),
            atol=1e-5,
        )

    def test_step_size_controls_flip_rate(self):
        """The step size modulates how often the chain moves at all.

        Unlike MALA, DMALA has no "absurd proposal" regime on a bounded
        discrete domain: the gradient term of the flip probability does not
        scale with the step size, so even huge steps keep the proposal
        gradient-informed (step-size robustness, as with Barker's proposal).
        The observable effect of the step size is instead the distance
        penalty ``-delta^2 / (2 * step_size)``: as it vanishes the chain
        keeps proposing flips, while a tiny step size freezes it in place.
        """
        kernel = discrete_langevin.build_kernel()

        def flip_rate(step_size):
            position = jnp.zeros(5)
            state = discrete_langevin.init(position, bernoulli07_logdensity)

            def step(state, key):
                new_state, _ = kernel(key, state, bernoulli07_logdensity, step_size)
                moved = jnp.any(new_state.position != state.position)
                return new_state, moved

            _, moved = jax.lax.scan(step, state, jax.random.split(self.next_key(), 200))
            return float(jnp.mean(moved))

        assert flip_rate(0.1) < 0.1
        assert flip_rate(2.0) > 0.5

    def test_large_step_size_saturates_gradient_only_flip(self):
        """As the step size grows the distance penalty vanishes.

        The flip probability then converges to the gradient-only logistic
        ``sigmoid(grad * (1 - 2 x) / 2)``.
        """
        values = jnp.array([0.0, 1.0])
        position = jnp.array([0.0, 1.0, 0.0])
        grad = jnp.array([1.3, -0.7, 0.4])

        logits = discrete_langevin._dlp_logits(position, grad, values, 1e6)
        probabilities = jax.nn.softmax(logits, axis=-1)
        expected_flip = jax.nn.sigmoid(0.5 * grad * (1.0 - 2.0 * position))
        expected_one = position + (1.0 - 2.0 * position) * expected_flip
        np.testing.assert_allclose(probabilities[:, 1], expected_one, atol=1e-4)

    def test_small_step_size_high_acceptance(self):
        """With a tiny step size, almost all proposals should be accepted."""
        position = jnp.zeros(2)
        state = discrete_langevin.init(position, bernoulli07_logdensity)
        kernel = discrete_langevin.build_kernel()

        def one_step(state, key):
            new_state, info = kernel(key, state, bernoulli07_logdensity, step_size=1e-5)
            return new_state, info.is_accepted

        keys = jax.random.split(self.next_key(), 200)
        _, accepted = jax.lax.scan(one_step, state, keys)
        acceptance_rate = jnp.mean(accepted.astype(jnp.float32))
        assert float(acceptance_rate) > 0.9

    def test_pytree_position(self):
        """Kernel works with PyTree positions."""

        def logdensity_fn(pos):
            return jnp.sum(
                pos["a"] * jnp.log(0.7) + (1 - pos["a"]) * jnp.log1p(-0.7)
            ) + jnp.sum(pos["b"] * jnp.log(0.4) + (1 - pos["b"]) * jnp.log1p(-0.4))

        position = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
        state = discrete_langevin.init(position, logdensity_fn)
        kernel = discrete_langevin.build_kernel()
        new_state, info = kernel(self.next_key(), state, logdensity_fn, self.step_size)
        chex.assert_trees_all_equal_shapes(new_state.position, position)
        assert 0.0 <= float(info.acceptance_rate) <= 1.0
        for leaf in jax.tree.leaves(new_state.position):
            assert jnp.all((leaf == 0) | (leaf == 1))

    def test_non_binary_values(self):
        """A custom candidate grid with more than two values is supported."""
        values = jnp.array([-1.0, 0.0, 2.0])

        def logdensity_fn(pos):
            return -0.5 * jnp.sum((pos - 0.5) ** 2)

        position = jnp.zeros(4)
        state = discrete_langevin.init(position, logdensity_fn)
        kernel = discrete_langevin.build_kernel()

        def step(state, key):
            state, _ = kernel(key, state, logdensity_fn, 1.0, values=values)
            return state, state.position

        _, positions = jax.lax.scan(step, state, jax.random.split(self.next_key(), 500))
        assert jnp.all((positions[..., None] == values).any(axis=-1))

    def test_proposal_logits_match_binary_flip_probability(self):
        """The K=2 proposal reduces to the documented Bernoulli flip probability.

        Flipping coordinate i should happen with probability
        sigmoid(grad_i * (1 - 2 x_i) / 2 - 1 / (2 * step_size)).
        """
        values = jnp.array([0.0, 1.0])
        position = jnp.array([0.0, 1.0, 0.0])
        grad = jnp.array([1.3, -0.7, 0.4])
        step_size = 0.8

        logits = discrete_langevin._dlp_logits(position, grad, values, step_size)
        probabilities = jax.nn.softmax(logits, axis=-1)
        expected_flip = jax.nn.sigmoid(
            0.5 * grad * (1.0 - 2.0 * position) - 1.0 / (2.0 * step_size)
        )
        expected_one = position + (1.0 - 2.0 * position) * expected_flip
        np.testing.assert_allclose(probabilities[:, 1], expected_one, atol=1e-6)

    def test_jit_compatible(self):
        """Kernel is JIT-compilable."""
        position = jnp.zeros(3)
        state = discrete_langevin.init(position, bernoulli07_logdensity)
        kernel = jax.jit(discrete_langevin.build_kernel(), static_argnums=(2,))
        new_state, info = kernel(
            self.next_key(), state, bernoulli07_logdensity, self.step_size
        )
        self.assertEqual(new_state.position.shape, (3,))


# ---------------------------------------------------------------------------
# discrete_langevin.as_top_level_api / registration
# ---------------------------------------------------------------------------


class DiscreteLangevinTopLevelAPITest(BlackJAXTest):
    """Tests for the DMALA top-level API."""

    def test_module_level_api(self):
        """Module-level as_top_level_api init + step runs and returns DLPState."""
        position = jnp.zeros(4)
        algo = discrete_langevin.as_top_level_api(bernoulli07_logdensity, step_size=1.0)
        state = algo.init(position)
        new_state, info = algo.step(self.next_key(), state)
        self.assertIsInstance(new_state, discrete_langevin.DLPState)
        self.assertEqual(new_state.position.shape, (4,))
        assert 0.0 <= float(info.acceptance_rate) <= 1.0

    def test_registered_at_top_level(self):
        """blackjax.dmala exposes the init / build_kernel attributes."""
        self.assertIs(blackjax.dmala.init, discrete_langevin.init)
        position = jnp.zeros(4)
        algo = blackjax.dmala(bernoulli07_logdensity, 1.0)
        state = algo.init(position)
        new_state, info = algo.step(self.next_key(), state)
        self.assertIsInstance(new_state, discrete_langevin.DLPState)
        self.assertIsInstance(info, discrete_langevin.DLPInfo)

    def test_top_level_jit(self):
        """Top-level step is JIT-compilable."""
        position = jnp.zeros(3)
        algo = blackjax.dmala(bernoulli07_logdensity, 1.0)
        state = algo.init(position)
        new_state, _ = jax.jit(algo.step)(self.next_key(), state)
        self.assertEqual(new_state.position.shape, (3,))


# ---------------------------------------------------------------------------
# Sampling correctness: empirical vs exact enumerated distribution
# ---------------------------------------------------------------------------


class DiscreteLangevinSamplingTest(BlackJAXTest):
    """The empirical distribution of the chain matches the exact target.

    These are the correctness gates for DMALA: on targets whose distribution
    can be enumerated exactly, the chain's empirical distribution must agree
    with it — a wrong proposal or Metropolis-Hastings ratio would yield a
    different stationary distribution. Independent chains are started from
    random initial states (several seeds).
    """

    def _run(self, logdensity_fn, dim, step_size, num_chains=8):
        algorithm = blackjax.dmala(logdensity_fn, step_size)
        key_init, key_inference = jax.random.split(self.next_key())
        initial_positions = jax.random.bernoulli(
            key_init, 0.5, (num_chains, dim)
        ).astype(jnp.float32)
        samples, acceptance_rate = run_dmala_chains(
            algorithm,
            key_inference,
            initial_positions,
            num_steps=30_000,
            num_burn_in=2_000,
        )
        return samples, float(acceptance_rate)

    def _assert_matches_exact(self, samples, logdensity_fn, dim, max_tv):
        states, exact = exact_binary_distribution(logdensity_fn, dim)
        indices = binary_state_indices(samples, dim)
        empirical = jnp.bincount(indices, length=2**dim) / len(indices)

        tv_distance = 0.5 * float(jnp.abs(empirical - exact).sum())
        assert tv_distance < max_tv, (
            f"Total-variation distance {tv_distance:.4f} to the exact "
            f"distribution exceeds {max_tv}"
        )
        # Per-coordinate marginals as a second, more readable view.
        np.testing.assert_allclose(empirical @ states, exact @ states, atol=2 * max_tv)
        return tv_distance

    def test_factorized_bernoulli_matches_exact_distribution(self):
        """Independent Bernoulli target: empirical law matches enumeration."""
        dim = len(BERNOULLI_PROBS)
        for _ in range(2):  # two independent draws of chain keys / initial states
            samples, acceptance_rate = self._run(bernoulli5_logdensity, dim, 1.0)
            assert 0.1 < acceptance_rate <= 1.0
            self._assert_matches_exact(samples, bernoulli5_logdensity, dim, max_tv=0.02)

    def test_ising_model_matches_exact_distribution(self):
        """Correlated pairwise Ising target: empirical law matches enumeration."""
        dim = len(ISING_FIELDS)
        for _ in range(2):
            samples, acceptance_rate = self._run(ising_logdensity, dim, 1.0)
            assert 0.1 < acceptance_rate <= 1.0
            self._assert_matches_exact(samples, ising_logdensity, dim, max_tv=0.03)


if __name__ == "__main__":
    absltest.main()
