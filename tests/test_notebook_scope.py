"""The legacy study notebook must reject full execution before any work."""
import ast
import json
from pathlib import Path
import unittest


class StudyScopeTests(unittest.TestCase):
    def test_full_notebook_stops_before_imports_or_outputs(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               '03_main_study.ipynb').read_text())
        source = ''.join(next(c for c in notebook['cells']
                              if c['cell_type'] == 'code')['source'])
        tree = ast.parse(source)
        self.assertEqual(tree.body[0].targets[0].id, 'MODE')
        self.assertEqual(tree.body[0].value.value, 'quick')
        gate = ast.Module(body=[tree.body[1]], type_ignores=[])
        # Only run the gate: importing or executing notebook study cells is forbidden.
        with self.assertRaisesRegex(RuntimeError, 'manifest runner'):
            exec(compile(gate, '<scope gate>', 'exec'), {'MODE': 'full'})
        exec(compile(gate, '<scope gate>', 'exec'), {'MODE': 'quick'})
