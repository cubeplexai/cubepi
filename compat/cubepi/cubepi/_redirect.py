"""On-demand cubepi.* → cubeloop.* import alias."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real_name: str) -> None:
        self.real_name = real_name

    def create_module(self, spec: importlib.machinery.ModuleSpec):  # noqa: ARG002
        return importlib.import_module(self.real_name)

    def exec_module(self, module: object) -> None:  # noqa: ARG002
        return None

    def get_code(self, fullname: str):  # noqa: ARG002
        # runpy (`python -m cubepi.cli`) requires get_code; delegate to
        # cubeloop's real loader so aliased packages stay executable.
        spec = importlib.util.find_spec(self.real_name)
        if spec is None or spec.loader is None:
            return None
        getter = getattr(spec.loader, "get_code", None)
        if getter is None:
            return None
        return getter(self.real_name)


class CubepiRedirectFinder(importlib.abc.MetaPathFinder):
    """Resolve cubepi.foo to the already-named cubeloop.foo module object."""

    def find_spec(  # type: ignore[override]
        self,
        fullname: str,
        path: object,  # noqa: ARG002
        target: object = None,  # noqa: ARG002
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith("cubepi."):
            return None
        real_name = "cubeloop" + fullname[len("cubepi") :]
        real_spec = importlib.util.find_spec(real_name)
        if real_spec is None:
            return None
        spec = importlib.machinery.ModuleSpec(
            fullname,
            _AliasLoader(real_name),
            origin=real_spec.origin,
            is_package=real_spec.submodule_search_locations is not None,
        )
        spec.submodule_search_locations = real_spec.submodule_search_locations
        return spec


def install() -> None:
    if not any(isinstance(f, CubepiRedirectFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, CubepiRedirectFinder())
