"""Experiment 1: geometric dead-ball detection and margin-gated possession hysteresis.

Importable as a package so that later experiments can reuse its layers without colliding with
them. Every experiment folder has a config.py and a features.py, and plain sys.path insertion
makes those names fight: whichever directory lands first on the path wins, and the loser silently
gets the wrong module's constants.

As a package, `from exp01_geometry_hysteresis import deadball` resolves exp01's own siblings
through its own namespace, so exp02 can keep its own config.py and features.py with no ambiguity.

The modules still run standalone -- `python experiments/exp01_geometry_hysteresis/deadball.py`
works -- because each one falls back to a flat import when it is run as a script rather than
imported as part of the package.
"""
