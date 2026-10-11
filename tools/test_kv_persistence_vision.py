"""CPU-only fake-record gates; never launch an engine or access a model."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kv_persistence_vision import METRICS, NAMES, SETTINGS, STATE_KEYS, private_args, verify


def fake_results():
    records = []
    for i, name in enumerate(NAMES):
        state = {k: '0011' for k in STATE_KEYS}
        state['L'] = '259' if i == 7 else '131'
        if i in (1, 3, 4, 7):
            state['gdn'] = str(100 + i)
        records.append({'name': name, 'ids': [42], 'finish': 'length', 'state': state,
                        'input_tokens': int(state['L']), 'prompt_tokens': int(state['L']),
                        'reused': 128, **dict.fromkeys(METRICS, 0)})
    info = dict.fromkeys(SETTINGS, 0)
    info.update(spec=2, mtp_max=1)
    results = {'baseline': records, 'candidate': copy.deepcopy(records),
               'baseline_restart_prime': dict(records[6]['state']),
               'engine_info': {label: dict(info) for label in
                               ('baseline', 'baseline_restart', 'candidate', 'candidate_restart')},
               'shutdown': {'baseline': {'exit_code': 0, 'saved_bytes': 0},
                            'candidate': {'exit_code': 0, 'saved_bytes': 1000}}}
    results['engine_info']['candidate_restart']['kv_persist_loaded'] = 3
    for i in (1, 3, 4, 5):
        results['candidate'][i]['kv_persist_write_bytes'] = 1000
    for i in (2, 5, 7):
        results['candidate'][i].update(kv_persist_restored_tokens=128, kv_persist_read_bytes=1000)
    return results


class VerifyTest(unittest.TestCase):
    def test_accepts_disk_parity(self):
        verify(fake_results())

    def test_rejects_no_restoration(self):
        for index in (2, 5, 7):
            with self.subTest(index=index):
                results = fake_results()
                results['candidate'][index]['kv_persist_restored_tokens'] = 0
                with self.assertRaisesRegex(AssertionError, 'no actual disk restoration'):
                    verify(results)

    def test_rejects_different_output(self):
        results = fake_results()
        results['candidate'][7]['ids'] = [43]
        with self.assertRaisesRegex(AssertionError, 'output differs'):
            verify(results)

    def test_rejects_valid_state_difference(self):
        for key in ('gdn', 'ple', 'tail', 'pooled', 'kv', 'ple_prev'):
            with self.subTest(key=key):
                results = fake_results()
                results['candidate'][7]['state'][key] = 'ffff'
                with self.assertRaisesRegex(AssertionError, 'main-model state differs'):
                    verify(results)

    def test_allows_only_explained_spare_rows(self):
        results = fake_results()
        results['candidate'][7]['state'].update(dead='ffff', pooled_full='ffff')
        verify(results)

    def test_rejects_missing_state_even_spare(self):
        results = fake_results()
        del results['candidate'][7]['state']['dead']
        with self.assertRaisesRegex(AssertionError, 'incomplete state'):
            verify(results)

    def test_rejects_same_tokens_b_without_write(self):
        results = fake_results()
        results['candidate'][1]['kv_persist_write_bytes'] = 0
        with self.assertRaisesRegex(AssertionError, 'no switch write evidence'):
            verify(results)

    def test_rejects_repeat_write(self):
        results = fake_results()
        results['candidate'][6]['kv_persist_write_bytes'] = 1000
        with self.assertRaisesRegex(AssertionError, 'repeat rewrote'):
            verify(results)

    def test_rejects_missing_metrics(self):
        results = fake_results()
        del results['candidate'][7]['kv_persist_read_bytes']
        with self.assertRaisesRegex(AssertionError, 'missing disk metrics'):
            verify(results)

    def test_rejects_invalid_tail_extent(self):
        results = fake_results()
        results['candidate'][7]['state']['L'] = '260'
        with self.assertRaisesRegex(AssertionError, 'tail hash includes invalid'):
            verify(results)

    def test_rejects_quit_without_commit(self):
        results = fake_results()
        results['shutdown']['candidate']['saved_bytes'] = 0
        with self.assertRaisesRegex(AssertionError, 'QUIT did not'):
            verify(results)

    def test_rejects_unmatched_live_prefix_baseline(self):
        results = fake_results()
        results['baseline_restart_prime']['gdn'] = 'ffff'
        with self.assertRaisesRegex(AssertionError, 'same live image prefix'):
            verify(results)

    def test_preserves_layer_split_and_removes_inherited_sinks(self):
        cfg = {'gpus': [0, 1], 'layer_split': '18', 'expert_profile_save': 'production.bin',
               'args': ['--layer-split', '18', '--kv-persist', '--kv-persist-dir', 'production',
                        '--kv-persist-identity', 'production', '--kv-persist-max-mib', '1',
                        '--conversation-cache-mib', '8192', '--spec', '8',
                        '--expert-profile-save', 'production.bin']}
        baseline = private_args(cfg)
        self.assertNotIn('--kv-persist', baseline)
        self.assertNotIn('production', baseline)
        self.assertNotIn('production.bin', baseline)
        self.assertEqual(baseline[baseline.index('--layer-split') + 1], '18')
        self.assertEqual(baseline[baseline.index('--conversation-cache-mib') + 1], '0')
        candidate = private_args(cfg, Path('private'))
        self.assertIn('--vision', candidate)
        self.assertEqual(candidate[candidate.index('--kv-persist-dir') + 1], 'private')

    def test_dry_run_needs_no_paths_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'not-created'
            command = [sys.executable, '-B', str(Path(__file__).with_name('kv_persistence_vision.py')),
                       '--config', 'missing.json', '--engine', 'missing.exe', '--tokenizer', 'missing',
                       '--output', str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn('Dry run', completed.stdout)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
