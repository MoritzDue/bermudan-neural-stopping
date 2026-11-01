import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import archival as a
import m5_runner as r
import stopping as st


class ArchivalTests(unittest.TestCase):
    def record(self):
        contract = dict(execution_code_hash='a'*16, runner_sha256='runner',
                        unit={'id': 'one'}, plan_sha256='plan')
        result = {'value': 1}
        return dict(contract=contract, complete=True, result=result,
                    result_sha256=st.stable_hash(result))

    def test_only_execution_hash_exception(self):
        record = self.record()
        expected = dict(record['contract'], execution_code_hash='b'*16)
        self.assertEqual(a.verify_record(record, expected), 'a'*16)
        for change in ('unit', 'runner_sha256', 'plan_sha256', 'extra'):
            changed = copy.deepcopy(record)
            changed['contract'][change] = 'altered'
            with self.assertRaises(ValueError):
                a.verify_record(changed, expected)
        for key, value in [('complete', False), ('result', {'value': 2})]:
            changed = copy.deepcopy(record)
            changed[key] = value
            with self.assertRaises(ValueError):
                a.verify_record(changed, expected)

    def test_policy_and_artifact_rejection(self):
        a.verify_policy({'policy_sha256': 'good'}, 'good')
        with self.assertRaises(ValueError):
            a.verify_policy({'policy_sha256': 'bad'}, 'good')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'evidence.txt'
            path.write_text('original')
            artifacts = {'evidence.txt': r.sha256(path)}
            a.verify_artifacts(directory, artifacts)
            path.write_text('altered')
            with self.assertRaises(ValueError):
                a.verify_artifacts(directory, artifacts)
            with self.assertRaises(ValueError):
                a.verify_artifacts(directory, {'../escape': 'hash'})

    def test_resume_still_rejects_historical_execution_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.json'
            record = self.record()
            path.write_text(json.dumps(record))
            before = path.read_bytes()
            execute = Mock()
            with self.assertRaises(ValueError):
                r.checkpoint(path, dict(record['contract'],
                                       execution_code_hash='b'*16), execute)
            execute.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
