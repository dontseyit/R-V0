from utils import clamp


class PIDController:
    """A simple PID controller."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float,
        output_limit: float,
        fps: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.fps = fps
        self._integral = 0.0
        self._prev_error = None

    def reset(self) -> None:
        """Reset the internal state of the controller."""
        self._integral = 0.0
        self._prev_error = None

    def compute(self, error: float, dt: float) -> float:
        """Compute the PID output given the current error and time delta."""
        if dt <= 0:
            dt = 1.0 / self.fps

        self._integral = clamp(
            self._integral + error * dt, -self.integral_limit, self.integral_limit
        )

        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        return clamp(output, -self.output_limit, self.output_limit)
