"""neuralchess - an AlphaZero-style chess engine written from scratch.

The package has no required third-party dependencies beyond NumPy: the chess
rules, the neural network, the autograd engine and the search are all
implemented here.

Sub-packages
------------
``neuralchess.chess``       bitboards, legal move generation, SAN/PGN
``neuralchess.encoding``    position -> tensor, move <-> policy index
``neuralchess.nn``          autograd, layers, the policy-value network
``neuralchess.search``      batched PUCT Monte-Carlo tree search
``neuralchess.selfplay``    game generation and the replay buffer
``neuralchess.train``       the training loop, losses and metrics
``neuralchess.evaluation``  baselines, arena and Elo estimation
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
