# cubepi is now cubeloop

The `cubepi` PyPI package has been renamed to **cubeloop**.

```bash
pip install cubeloop
```

```python
from cubeloop import Agent, tool
```

This distribution is a transitional wrapper: it depends on `cubeloop==0.14.0`,
re-exports the public API, and aliases `cubepi.*` imports onto `cubeloop.*`.
Installing it prints a deprecation warning on first import.

See https://cubeloop.dev/docs/migration/from-cubepi
