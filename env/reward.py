"""
Differential Sharpe Ratio (Moody et al. 1998)

Provides a per-timestep reward signal for risk-adjusted returns,
enabling online RL training without needing to compute the Sharpe
ratio over an entire episode.

D_t = (B_{t-1} * dA_t - 0.5 * A_{t-1} * dB_t) / (B_{t-1} - A_{t-1}^2)^{3/2}

where:
  dA_t = R_t - A_{t-1}
  dB_t = R_t^2 - B_{t-1}
  A_t  = A_{t-1} + eta * dA_t   (EMA of returns)
  B_t  = B_{t-1} + eta * dB_t   (EMA of squared returns)
  eta  ~ 1/252
"""


class DifferentialSharpeRatio:
    """Stateful reward calculator for the Differential Sharpe Ratio."""

    def __init__(self, eta: float = 1.0 / 252.0):
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns

    def reset(self):
        """Reset state at the beginning of a new episode."""
        self.A = 0.0
        self.B = 0.0

    def step(self, portfolio_return: float) -> float:
        """
        Compute the differential Sharpe ratio for this timestep,
        then update internal state.

        Args:
            portfolio_return: Simple return of the portfolio at this step.

        Returns:
            D_t: The differential Sharpe ratio reward.
        """
        R = portfolio_return
        A_prev = self.A
        B_prev = self.B

        delta_A = R - A_prev
        delta_B = R * R - B_prev

        # Compute denominator: (B_{t-1} - A_{t-1}^2)^{3/2}
        variance_est = B_prev - A_prev * A_prev

        if variance_est < 1e-12:
            # Insufficient data or zero variance — return 0
            D_t = 0.0
        else:
            denom = variance_est ** 1.5
            numerator = B_prev * delta_A - 0.5 * A_prev * delta_B
            D_t = numerator / denom

        # Update EMA estimates
        self.A = A_prev + self.eta * delta_A
        self.B = B_prev + self.eta * delta_B

        return D_t
