class Policy:
    """Base class for agent policies.

    Subclasses must implement ``__call__``, which receives the current agent
    and perception model and returns a force impulse to apply, or ``None`` to
    do nothing.
    """

    def __call__(self, agent, model) -> tuple[float, float] | None:
        """Return a force impulse (fx, fy) or None.

        Args:
            agent: The Agent instance (position, velocity, perceive, kick, …).
            model: The PerceptionModel instance (depth_profile, camera_image, …).

        Returns:
            (fx, fy) force impulse to pass to agent.apply_force(), or None.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# TODO: implement your policies below
# ---------------------------------------------------------------------------
