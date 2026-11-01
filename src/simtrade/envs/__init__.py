"""Optional reinforcement-learning environments (``pip install simtrade[rl]``)."""


def __getattr__(name):
    if name in ("TradingEnv", "PolicyAgent"):
        from . import trading_env

        return getattr(trading_env, name)
    raise AttributeError(name)


__all__ = ["TradingEnv", "PolicyAgent"]
