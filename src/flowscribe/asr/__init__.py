"""ASR backends.

Nothing is imported here: engines pull in heavy optional dependencies, and
importing this package must stay free. Resolve backends through
``flowscribe.registry.ASR`` instead.
"""

__all__: list[str] = []
