"""Test for ensuring parts of a module is self contained."""
from __future__ import annotations

import importlib
import multiprocessing
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple


class TestSelfContained(unittest.TestCase):
    """Test if a list of modules are self contained."""

    @dataclass(frozen=True)
    class PyModule:
        """A python module with a name and dependencies."""

        name: str
        allowed_to_import_from: Tuple[str]

    def test_is_tools_self_contained(self):
        """Test if parts of tools module are self contained."""
        for module in (
            TestSelfContained.PyModule(
                name="tools/common",
                allowed_to_import_from=(),
            ),
            TestSelfContained.PyModule(
                name="tools/host",
                allowed_to_import_from=("tools/common",),
            ),
        ):
            self._is_module_self_contained(module)

    def _is_module_self_contained(self, module: TestSelfContained.PyModule):
        """Check if a module is self contained.

        This function spawns a clean process and performs the
        imports on-the-fly, effectively keeping track of
        the set of imported modules.
        """
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(
            target=TestSelfContained._track_imports,
            args=(
                module.name,
                module.allowed_to_import_from,
                "tools",
                queue,
            ),
        )
        proc.start()
        imported_modules = queue.get()
        proc.join()
        self.assertIsNone(
            imported_modules,
            msg=(
                "\n\nThe following imported modules are outside of the toplevel "
                f"module and not allowed {imported_modules}"
            ),
        )

    @staticmethod
    def _track_imports(
        module_toplevel: str,
        allowed_module_toplevels: Tuple[str],
        global_toplevel: str,
        queue: multiprocessing.Queue,
    ):
        target_py_modules = TestSelfContained._build_target_py_modules_list(
            module_toplevel
        )
        allowed_modules = TestSelfContained._build_allowed_modules_list(
            allowed_module_toplevels
        )

        pre_import_modules = set(sys.modules.keys())
        for py_module in target_py_modules:
            importlib.import_module(str(py_module))
        post_import_modules = set(sys.modules.keys())

        imported = post_import_modules.difference(pre_import_modules)
        diff = imported.difference(target_py_modules).difference(allowed_modules)
        diff = set(filter(lambda ss: ss.startswith(global_toplevel), diff))
        queue.put(diff or None)

    @staticmethod
    def _build_target_py_modules_list(module_toplevel: str) -> List[str]:
        module_toplevel_path = Path(module_toplevel)
        target_py_files = filter(
            lambda path: path.suffix == ".py" or path.is_dir(),
            module_toplevel_path.rglob("*"),
        )
        target_list = list(target_py_files) + [module_toplevel_path]
        return list(
            map(TestSelfContained._file_path_to_module_dot_separated, target_list)
        )

    @staticmethod
    def _build_allowed_modules_list(allowed_module_toplevels: Tuple[str]) -> List[str]:
        allowed_files: Set[Path] = set()
        for module in allowed_module_toplevels:
            allowed_files.update(Path(module).rglob("*"))

        return list(
            map(TestSelfContained._file_path_to_module_dot_separated, allowed_files)
        ) + [
            TestSelfContained._file_path_to_module_dot_separated(Path(module))
            for module in allowed_module_toplevels
        ]

    @staticmethod
    def _file_path_to_module_dot_separated(py_file_path: Path) -> str:
        return str(py_file_path.with_suffix("")).replace("/", ".")
