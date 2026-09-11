# cubepi is now cubeloop

The `cubepi` PyPI package has been renamed to **cubeloop**.

```bash
pip install cubeloop
```

```python
from cubeloop import Agent, tool
```

This distribution is a dependency-free tombstone. It does not install, import,
or proxy CubeLoop. Any `cubepi` import or command fails with migration
instructions so stale dependencies cannot silently continue.

See https://cubeloop.dev/docs/migration/from-cubepi
