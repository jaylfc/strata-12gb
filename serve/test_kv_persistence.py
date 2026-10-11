"""Disk KV configuration, request metrics and graceful close contract."""
import io
import subprocess
import unittest
from unittest import mock
from serve.server import EngineStuck, StrataEngine, engine_close_s


class Persistence(unittest.TestCase):
    def test_close_deadline_configuration(self):
        cfg = {"args": ["--kv-persist", "--kv-persist-identity", "model-v1:int8"]}
        self.assertEqual(engine_close_s(cfg), 300)
        cfg["engine_close_s"] = 600
        self.assertEqual(engine_close_s(cfg), 600)
        self.assertEqual(engine_close_s({"args": []}), 20)

    def test_request_disk_metrics(self):
        engine = StrataEngine("missing", [], lazy=True)
        engine._parse_done("DONE 64 4096 120.1 650.2 length 31 49 2048 100 200 0 3 1.2 2048 25 2048 2800000 12.3 0 0.0")
        self.assertEqual(engine.last["offloaded"], 25)
        self.assertEqual(engine.last["kv_persist_restored_tokens"], 2048)
        self.assertEqual(engine.last["kv_persist_read_bytes"], 2800000)
        self.assertEqual(engine.last["kv_persist_restore_ms"], 12.3)
        self.assertEqual(engine.last["kv_persist_write_bytes"], 0)
        self.assertEqual(engine.last["kv_persist_commit_ms"], 0.0)

    def engine_and_process(self, code):
        engine = StrataEngine("missing", ["--kv-persist"], lazy=True)
        proc = mock.Mock(stdin=io.StringIO(), stdout=io.StringIO(), returncode=None)
        proc.poll.side_effect = lambda: proc.returncode
        def wait(timeout):
            proc.returncode = code
            return code
        proc.wait.side_effect = wait
        engine.proc, engine.close_s = proc, 300
        return engine, proc

    def test_successful_close_waits_for_commit(self):
        engine, proc = self.engine_and_process(0)
        engine.close()
        proc.wait.assert_called_once_with(timeout=300)
        proc.terminate.assert_not_called()
        self.assertIsNone(engine.proc)

    def test_failed_commit_cannot_report_successful_unload(self):
        engine, proc = self.engine_and_process(1)
        with self.assertRaises(EngineStuck):
            engine.unload()

    def test_forced_termination_cannot_report_successful_unload(self):
        engine, proc = self.engine_and_process(-15)
        proc.wait.side_effect = [subprocess.TimeoutExpired("strata", 300), -15]
        proc.terminate.side_effect = lambda: setattr(proc, "returncode", -15)
        with self.assertRaises(EngineStuck):
            engine.unload()


if __name__ == "__main__":
    unittest.main()
