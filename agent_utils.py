import numpy as np


class OUActionNoise:
    """
    Implementation of the Ornstein-Uhlenbeck process.
    To implement better exploration by the Actor network, noisy perturbation is used:
    an Ornstein-Uhlenbeck process to generate noise, as described in the paper.
    It samples noise from a correlated normal distribution.
    From: https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
    """

    def __init__(self, mean, std_deviation, theta=1, dt=1e-2, x_initial=None):
        # original values in paper: dt=1e-2, theta=0.15
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self) -> float:
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # store x into x_prev - makes next noise dependent on current one
        self.x_prev = x

        return x

    def reset(self) -> None:
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
