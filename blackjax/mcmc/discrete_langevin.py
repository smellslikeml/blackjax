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
"""Public API for the (Metropolis-adjusted) Discrete Langevin Proposal kernel.

DMALA is the discrete analogue of MALA: a gradient-based Metropolis-Hastings
kernel for distributions over discrete domains, by default binary ``{0, 1}``
vectors :cite:p:`zhang2022langevin` (requested in issue #835).

The Discrete Langevin Proposal (DLP) is derived from a first-order Taylor
expansion of the log-density :math:`U(x) = \\log \\pi(x)` around the current
state: the overdamped-Langevin transition
:math:`x' = x + (\\alpha / 2) \\nabla U(x) + \\sqrt{\\alpha}\\,\\epsilon` is
localized on the discrete domain. For each coordinate :math:`i`, the
probability of proposing the candidate value :math:`v` is

.. math::

    q(v \\mid x) \\propto \\exp\\left(
        \\frac{1}{2} (v - x_i)\\, \\nabla_i U(x)
        - \\frac{(v - x_i)^2}{2 \\alpha}
    \\right),

i.e. a per-coordinate categorical distribution whose (unnormalized) logits
combine the local gradient with a quadratic penalty on the distance moved, in
exact analogy with the Gaussian kernel of MALA. Proposals are then accepted
with the usual Metropolis-Hastings probability, which corrects for the
asymmetry of :math:`q`; the reverse proposal :math:`q(x \\mid x')` is computed
the same way but from the gradient evaluated at :math:`x'`.

In the binary case the flip probability simplifies to
:math:`\\sigma\\left(\\nabla_i U(x) (1 - 2 x_i) / 2 - 1 / (2\\alpha)\\right)`
with :math:`\\sigma` the logistic function, so small step sizes keep the chain
in place while large ones favor flips.

The gradient is that of the continuously-relaxed log-density evaluated at the
current discrete position, obtained with ``jax.grad``; positions must therefore
be (arrays of) floats that live exactly on the candidate grid ``values``.

This is a clean-room implementation from the paper alone
(arXiv:2206.09914); the authors' reference implementation was neither read nor
ported. The unadjusted variant (DULA) is not implemented.
"""

import operator
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, Numeric, PRNGKey

__all__ = ["DLPState", "DLPInfo", "init", "build_kernel", "as_top_level_api"]

_DEFAULT_VALUES: tuple[float, ...] = (0.0, 1.0)


class DLPState(NamedTuple):
    """State of the DMALA algorithm.

    The DMALA algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of the
    (continuously relaxed) log-probability density.

    """

    position: ArrayLikeTree
    logdensity: float
    logdensity_grad: ArrayTree


class DLPInfo(NamedTuple):
    """Additional information on the DMALA transition.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance rate of the transition.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The proposed state, whether it was accepted or not.

    """

    acceptance_rate: float
    is_accepted: bool
    proposal: DLPState


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> DLPState:
    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, logdensity_grad = grad_fn(position)
    return DLPState(position, logdensity, logdensity_grad)


def build_kernel():
    """Build a DMALA kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def transition_energy(state, new_state, values, step_size):
        """Transition energy to go from `state` to `new_state`.

        Following the convention of
        :func:`blackjax.mcmc.proposal.compute_asymmetric_acceptance_ratio`, this
        is the negative log-density of `new_state` minus the log-density of
        `state`'s position under the DLP proposal centered at `new_state` (and
        therefore built from the gradient at `new_state`); its asymmetry in its
        two arguments yields the Metropolis-Hastings correction of DMALA.
        """
        logits = jax.tree.map(
            lambda y, g: _dlp_logits(y, g, values, step_size),
            new_state.position,
            new_state.logdensity_grad,
        )
        indices = jax.tree.map(lambda x: _value_index(x, values), state.position)
        return -new_state.logdensity - _dlp_log_proposal(logits, indices)

    compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    def kernel(
        rng_key: PRNGKey,
        state: DLPState,
        logdensity_fn: Callable,
        step_size: float,
        values: Array | None = None,
    ) -> tuple[DLPState, DLPInfo]:
        """Generate a new sample with the DMALA kernel."""
        if values is None:
            values = jnp.asarray(_DEFAULT_VALUES)
        values = jnp.asarray(values)
        grad_fn = jax.value_and_grad(logdensity_fn)

        key_proposal, key_rmh = jax.random.split(rng_key)

        forward_logits = jax.tree.map(
            lambda x, g: _dlp_logits(x, g, values, step_size),
            state.position,
            state.logdensity_grad,
        )
        proposal_indices = _dlp_sample_indices(key_proposal, forward_logits)
        proposed_position = jax.tree.map(lambda idx: values[idx], proposal_indices)

        proposed_logdensity, proposed_logdensity_grad = grad_fn(proposed_position)
        new_state = DLPState(
            proposed_position, proposed_logdensity, proposed_logdensity_grad
        )

        log_p_accept = compute_acceptance_ratio(
            state, new_state, values=values, step_size=step_size
        )
        accepted_state, info = sample_proposal(key_rmh, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = DLPInfo(p_accept, do_accept, new_state)

        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    values: Array | None = None,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the DMALA kernel.

    The general discrete Langevin kernel builder
    (:meth:`blackjax.mcmc.discrete_langevin.build_kernel`, alias
    `blackjax.dmala.build_kernel`) can be cumbersome to manipulate. Since most
    users only need to specify the kernel parameters at initialization time, we
    provide a helper function that specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.dmala` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new DMALA kernel can be initialized and used with the following code:

    .. code::

        dmala = blackjax.dmala(logdensity_fn, step_size)
        state = dmala.init(position)
        new_state, info = dmala.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(dmala.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       kernel = blackjax.dmala.build_kernel(logdensity_fn)
       state = blackjax.dmala.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from. It must be
        differentiable (its gradient is taken at the current, discrete
        position through the continuous relaxation).
    step_size
        The value to use for the step size of the discrete Langevin proposal.
    values
        The candidate values each coordinate can take, defaulting to the
        binary domain ``[0, 1]``. Shared across all coordinates of all leaves
        of the position PyTree.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    kernel = build_kernel()
    return build_sampling_algorithm(
        kernel, init, logdensity_fn, kernel_args=(step_size, values)
    )


def _dlp_logits(
    position_leaf: ArrayLikeTree, grad_leaf: ArrayLikeTree, values: Array, step_size
) -> Array:
    """Unnormalized log-probabilities of the DLP proposal for one position leaf.

    For every scalar coordinate of the leaf this returns one logit per
    candidate value in ``values``.
    """
    delta = values - position_leaf[..., None]
    return 0.5 * grad_leaf[..., None] * delta - delta**2 / (2.0 * step_size)


def _dlp_sample_indices(rng_key: PRNGKey, logits: ArrayTree) -> ArrayTree:
    """Draw one candidate index per scalar coordinate, independently."""
    leaves = jax.tree.leaves(logits)
    keys = jax.random.split(rng_key, len(leaves))
    keys_tree = jax.tree.unflatten(jax.tree.structure(logits), keys)
    return jax.tree.map(jax.random.categorical, keys_tree, logits)


def _value_index(position_leaf: ArrayLikeTree, values: Array) -> Array:
    """Index in ``values`` of the value taken by each scalar coordinate."""
    return jnp.argmin((values - position_leaf[..., None]) ** 2, axis=-1)


def _dlp_log_proposal(logits: ArrayTree, indices: ArrayTree) -> Numeric:
    """Log-probability of a fully factorized DLP proposal."""
    log_probs = jax.tree.map(
        lambda leaf_logits, leaf_indices: (
            jnp.take_along_axis(leaf_logits, leaf_indices[..., None], axis=-1)[..., 0]
            - logsumexp(leaf_logits, axis=-1)
        ),
        logits,
        indices,
    )
    return jax.tree.reduce(operator.add, jax.tree.map(jnp.sum, log_probs))
