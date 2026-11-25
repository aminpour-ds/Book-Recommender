"""Weights & Biases logging for recommender evaluation runs."""

import wandb
from typing import Literal

DEFAULT_PROJECT = "book-recommender"

class ExperimentLogger:
    """A wrapper around a single wandb run."""

    def __init__(self, name=None, config=None, tags=None,
                 project=DEFAULT_PROJECT, 
                 mode: Literal["online", "offline", "disabled", "shared"] = "offline",
    ):

        self.run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            tags=tags or [],
            mode=mode,
            reinit="finish_previous",
        )

    def log(self, metrics, step=None):
        """Log a dictionary of metrics."""
        if step is None:
            self.run.log(metrics)
        else:
            self.run.log(metrics, step=step)

    def finish(self):
        """Close the run."""
        self.run.finish()
