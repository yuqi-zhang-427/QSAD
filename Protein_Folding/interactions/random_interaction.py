# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0.
# You may obtain a copy of this license at http://www.apache.org/licenses/LICENSE-2.0.

"""A class defining a random interaction between beads of a peptide."""

import numpy as np

from .interaction import Interaction

# pylint: disable=too-few-public-methods

# Random source for the synthetic pair energies below. Call
# ``seed_random_interaction(seed)`` to make them reproducible.
_RNG = np.random.default_rng()


def seed_random_interaction(seed: int) -> None:
    """Re-seed the RNG used by :class:`RandomInteraction`."""
    global _RNG  # pylint: disable=global-statement
    _RNG = np.random.default_rng(seed)


class RandomInteraction(Interaction):
    """A class defining a random interaction between beads of a peptide."""

    def calculate_energy_matrix(self, residue_sequence: str) -> np.ndarray:
        """
        Calculates an energy matrix for a random interaction.

        Args:
            residue_sequence: Dummy residue sequence; only used to infer the length of a chain.

        Returns:
            Numpy array of pair energies for amino acids.
        """
        chain_len = len(residue_sequence)
        pair_energies = -1 - 4 * _RNG.random(
            (chain_len + 1, 2, chain_len + 1, 2)
        )
        return pair_energies
