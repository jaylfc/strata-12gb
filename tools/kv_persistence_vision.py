"""Private, sequential vision/disk-KV parity gate. Dry-run unless --run.

Synthetic SVE files test image identity and M-RoPE, not encoder quality. No HTTP
server is contacted. Schedule --run only in a separately available GPU window.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from conversation_cache_parity import STATE_KEYS, load_tokenizer, require, state_hashes

NAMES = ('image-A', 'same-tokens-image-B', 'image-A-return', 'same-embedding-new-grid',
         'text', 'image-A-final', 'same-image-repeat', 'image-A-continuation-after-QUIT')
# generate.cpp hashes dead and pooled_full's extra row even though QSA reads only
# completed pooled rows [0, L//4). That spare row is overwritten at completion;
# checkpoint restore can populate it differently without changing valid state.
# Keep the raw hashes in JSON, require every STATE_KEY, and exclude ONLY these
# two keys. tail is fully valid here: all measured live lengths have L % 4 == 3,
# so all three indexer tail slots have been filled. Never exclude tail/gdn/kv.
EXCLUDED_STATE_KEYS = {
    'dead': 'spare indexer key, overwritten on the next block completion',
    'pooled_full': 'includes the non-readable spare row at L//4; compare pooled instead',
}
VALID_STATE_KEYS = tuple(k for k in STATE_KEYS if k not in EXCLUDED_STATE_KEYS)
METRICS = ('kv_persist_restored_tokens', 'kv_persist_read_bytes', 'kv_persist_restore_ms',
           'kv_persist_write_bytes', 'kv_persist_commit_ms')
SETTINGS = ('expert_slots', 'kv', 'kv_resident', 'context', 'spec', 'mtp_max', 'lookup', 'cvec',
            'conversation_cache_mib')


def valid_state(record):
    require(set(STATE_KEYS) <= record['state'].keys(), 'incomplete state fingerprint')
    return {k: record['state'][k] for k in VALID_STATE_KEYS}


def verify(results):
    """Fail closed on parity AND real disk evidence; usable with CPU fake records."""
    baseline, candidate = results['baseline'], results['candidate']
    for records in (baseline, candidate):
        require([r['name'] for r in records] == list(NAMES), 'incomplete vision sequence')
        for r in records:
            require(len(r['ids']) == 1, 'missing one-token greedy output')
            require(r['finish'] in ('stop', 'length'), 'request did not finish normally')
            require(set(METRICS) <= r.keys(), 'missing disk metrics (updated engine/server required)')
            require(all(r[k] >= 0 for k in METRICS), 'negative disk metric')
            require(int(r['state']['L']) % 4 == 3, 'tail hash includes invalid slots: L must be 3 mod 4')
            require(int(r['state']['L']) == r['input_tokens'], 'unexpected measured live length')
            require(r['prompt_tokens'] == r['input_tokens'], 'prompt length mismatch')
            valid_state(r)
    for before, after in zip(baseline, candidate):
        require(before['ids'] == after['ids'], f"output differs: {after['name']}")
        require(valid_state(before) == valid_state(after), f"main-model state differs: {after['name']}")
    require(valid_state({'state': results['baseline_restart_prime']}) == valid_state(baseline[6]),
            'baseline continuation must start with the same live image prefix')
    infos = results['engine_info']
    reference = infos['baseline']
    for label in ('baseline', 'baseline_restart', 'candidate', 'candidate_restart'):
        info = infos[label]
        require(all(k in info and k in reference for k in SETTINGS), 'missing engine settings')
        require(all(info[k] == reference[k] for k in SETTINGS), 'engine settings differ')
        require(info['conversation_cache_mib'] == 0, 'RAM parking enabled')
        require(info['spec'] == 2 and info['mtp_max'] == 1, 'spec configuration differs')
    require(all(r[k] == 0 for r in baseline for k in METRICS), 'baseline used disk persistence')
    for i in (1, 3, 4, 5):
        require(candidate[i]['kv_persist_write_bytes'] > 0,
                f'no switch write evidence: {NAMES[i]}')
    for i in (2, 5, 7):
        r = candidate[i]
        require(r['kv_persist_restored_tokens'] > 0 and r['kv_persist_read_bytes'] > 0,
                f'no actual disk restoration: {NAMES[i]}')
        require(r['reused'] >= r['kv_persist_restored_tokens'], 'disk prefix was not actually reused')
    require(candidate[6]['kv_persist_write_bytes'] == 0, 'same-image repeat rewrote disk KV')
    require(candidate[6]['kv_persist_restored_tokens'] == 0, 'same-image repeat unnecessarily restored disk KV')
    require(infos['candidate_restart'].get('kv_persist_loaded', 0) > 0, 'restart did not load disk index')
    shutdown = results['shutdown']['candidate']
    require(shutdown['exit_code'] == 0 and shutdown['saved_bytes'] > 0,
            'QUIT did not successfully commit live A')
    require(results['shutdown']['baseline']['exit_code'] == 0, 'baseline did not exit cleanly')
    for i in (1, 3):
        require(valid_state(baseline[0]) != valid_state(baseline[i]), 'image/grid fixture did not alter model state')
    for i in (2, 5, 6):
        require(valid_state(baseline[0]) == valid_state(baseline[i]), 'A replay baseline is not stable')


def private_args(cfg, cache_dir=None):
    """Preserve GPU/layer split; remove inherited persistence and mutable sinks."""
    from serve.server import engine_args
    clean_cfg = {k: v for k, v in cfg.items()
                 if k not in ('expert_profile_save', 'expert_profile_save_every')}
    values = {'--kv-persist-dir', '--kv-persist-identity', '--kv-persist-max-mib',
              '--conversation-cache-mib', '--conversation-cache-slots', '--conversation-cache-min-free-mib',
              '--prompt-cache', '--prompt-cache-every', '--prompt-cache-root', '--turn-token',
              '--adapt-swaps', '--spec', '--mtp-max-t', '--suffix-draft', '--spec-min-p',
              '--expert-profile-save', '--expert-profile-save-every', '--trace-routing'}
    flags = {'--kv-persist', '--vision'}
    source = engine_args(clean_cfg)
    command = []
    i = 0
    while i < len(source):
        arg = source[i]
        key = arg.split('=', 1)[0]
        if key in values:
            require('=' in arg or i + 1 < len(source), f'missing argument for {key}')
            i += 1 if '=' in arg else 2
        elif key in flags:
            i += 1
        else:
            command.append(arg)
            i += 1
    command += ['--vision', '--conversation-cache-mib', '0', '--conversation-cache-slots', '0',
                '--prompt-cache', '6', '--prompt-cache-every', '0', '--prompt-cache-root', '0',
                '--greedy', '--adapt-swaps', '0', '--spec', '2', '--mtp-max-t', '1',
                '--suffix-draft', '0', '--spec-min-p', '0']
    if cache_dir is not None:
        command += ['--kv-persist', '--kv-persist-dir', str(cache_dir),
                    '--kv-persist-identity', 'private-vision-regression-v1', '--kv-persist-max-mib', '30720']
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'engine', 'tokenizer', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    if not args.run:
        print('Dry run: sequential private baseline/candidate + QUIT restart; no files created or model loaded.')
        print('Pass --run only during an available GPU/test window; production services are never contacted.')
        return
    # Heavy imports and filesystem/model work happen only after explicit opt-in.
    from conversation_cache_isolation import fixtures
    from gguf_reader import GGUFFile
    from serve.frontend import ChatTemplate
    from serve.server import StrataEngine, child_env
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    output = args.output.resolve()
    require(not output.exists(), 'output must be a NEW private directory')
    native = Path(cfg['args'][cfg['args'].index('--native') + 1])
    if not native.is_absolute():
        native = Path(cfg.get('cwd') or Path.cwd()) / native
    width = GGUFFile(str(native)).metadata['qwen4exp.embedding_length']
    tok_path = args.tokenizer.resolve()
    tok = load_tokenizer(tok_path)
    tpl = ChatTemplate(tok_path / 'chat_template.jinja')
    encode = lambda text: list(tok.encode(text, parse_special=True))
    turn = encode('<|im_start|>')
    require(len(turn) == 1, 'tokenizer must have a single turn marker')
    tail = encode(' blue square green triangle.' * 64)
    require(len(tail) >= 128, 'need >=128 tokens after final turn marker for short-prompt checkpoint')
    def padded(ids):
        ids = list(ids) + tail
        ids += [tail[-1]] * ((3 - len(ids)) % 4)
        return ids
    image_tokens = '<|vision_start|>' + '<|image_pad|>' * 4 + '<|vision_end|>'
    def prompt(text):
        return padded(encode(tpl.render([{'role': 'user', 'content': text}], enable_thinking=False)))
    A = prompt('Conversation A: inspect this image. ' + image_tokens)
    text = prompt('Unrelated text conversation: name a color.')
    suffix = encode('<|im_end|>\n<|im_start|>user\nName its color.<|im_end|>\n'
                    '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    fixtures(output, width)
    cache_dir = output / 'candidate-kv'
    cache_dir.mkdir(mode=0o700)
    env = child_env(cfg)
    env['STRATA_STATE_HASH'] = '1'
    results = {'engine_info': {}, 'commands': {}, 'shutdown': {},
               'state_hash_scope': 'primary GPU session; output parity also exercises all configured stages',
               'excluded_state_keys': EXCLUDED_STATE_KEYS, 'valid_state_keys': VALID_STATE_KEYS,
               'prompts': {'A': A, 'text': text}, 'tail_tokens': len(tail)}
    continuation = None
    for label in ('baseline', 'candidate'):
        command = private_args(cfg, cache_dir if label == 'candidate' else None)
        command += ['--turn-token', str(turn[0])]
        results['commands'][label] = command
        records = []
        results[label] = records
        for restart in (False, True):
            stage = label + ('_restart' if restart else '')
            log = output / (stage + '.log')
            engine = StrataEngine(str(args.engine.resolve()), command, cwd=cfg.get('cwd'),
                                  log=str(log), env=env)
            engine.close_s = (max(300.0, float(cfg.get('engine_close_s', 300)))
                              if label == 'candidate' else float(cfg.get('engine_close_s', 20)))
            results['engine_info'][stage] = dict(engine.info)
            start_index = len(records)
            def generate(name, ids, image):
                embeddings = str(output / (image + '.sve')) if image else None
                out = [t for t in engine.generate(ids, 1, {'temperature': 0}, threading.Event(),
                                                 embeddings=embeddings) if t is not None]
                records.append({'name': name, 'ids': out, 'input_tokens': len(ids),
                                'image': image, **dict(engine.last)})
                return out
            try:
                if restart:
                    if label == 'baseline':
                        # Compare identical prefill boundaries: a live, non-disk
                        # A prefix versus a restored A prefix. Cold full prefill
                        # can round differently from split prefix/continuation.
                        warm = [t for t in engine.generate(A, 1, {'temperature': 0}, threading.Event(),
                                embeddings=str(output / 'image-a.sve')) if t is not None]
                        require(warm == records[0]['ids'], 'baseline prefix priming output differs')
                    generate(NAMES[7], continuation, 'image-a')
                else:
                    generate(NAMES[0], A, 'image-a')
                    generate(NAMES[1], A, 'image-b')
                    generate(NAMES[2], A, 'image-a')
                    generate(NAMES[3], A, 'image-grid')
                    generate(NAMES[4], text, None)
                    head = generate(NAMES[5], A, 'image-a')
                    generate(NAMES[6], A, 'image-a')
                    if label == 'baseline':
                        continuation = padded(A + head + suffix)
                        results['prompts']['A+'] = continuation
                proc = engine.proc
                before_quit = log.read_text(encoding='utf-8')
            finally:
                engine.close()  # sends QUIT, waits for successful persistence commit
            log_text = log.read_text(encoding='utf-8')
            hashes = state_hashes(log_text)
            if restart and label == 'baseline':
                require(len(hashes) == 2, 'missing baseline prime/continuation fingerprints')
                require(valid_state({'state': hashes[0]}) == valid_state(records[6]),
                        'baseline live prime differs from the committed image prefix')
                results['baseline_restart_prime'] = hashes.pop(0)
            require(len(hashes) == len(records) - start_index, 'missing state hashes')
            for record, state in zip(records[start_index:], hashes):
                record['state'] = state
            shutdown_text = log_text[len(before_quit):]
            saved = re.findall(r'KV_PERSIST saved .*?bytes=(\d+)', shutdown_text)
            results['shutdown'][stage] = {'exit_code': proc.returncode,
                                          'saved_bytes': sum(map(int, saved)), 'saved_entries': len(saved)}
            (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    verify(results)
    print('PASS: vision/disk identity, repeat no-write, QUIT restart restoration, output and valid-state parity')
    print(f'Results: {output / "results.json"}')


if __name__ == '__main__':
    main()
