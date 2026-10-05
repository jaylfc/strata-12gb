"""Private GPU gate for completed batch slots, overwrite, cancellation and QUIT restoration."""
import argparse
import json
import re
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from conversation_cache_parity import load_tokenizer, require
from kv_persistence_vision import private_args
from serve.frontend import ChatTemplate
from serve.server import StrataEngine, child_env


def admission(engine, slot, prompt, count):
    engine._send(f"BGEN {slot} {count} " + ','.join(map(str, prompt)))
    output, metrics = [], None
    while True:
        line = engine.lines.get(timeout=300)
        require(line is not None, 'engine ended during admission')
        require(not line.startswith('ERR'), line)
        if line.startswith('T '):
            output.append(int(line.split()[1]))
        elif line.startswith('DONE '):
            engine._parse_done(line)
            metrics = dict(engine.last)
        elif line.startswith('BADM '):
            require(int(line.split()[1]) == slot and metrics is not None, 'wrong admission response')
            return output, metrics, line.split()[2] == '1'


def finish(engine, slot, output):
    while True:
        line = engine.slot_q[slot].get(timeout=300)
        require(line is not None, 'engine ended during batch decode')
        if line.startswith('BT '):
            output.append(int(line.split()[2]))
        elif line.startswith('BDONE '):
            return line.split()[3]


def pair(engine, prompts, count, cross_group=False):
    records = []
    for slot, prompt in enumerate(prompts):
        output, metrics, running = admission(engine, slot, prompt, count)
        require(running, 'request did not enter a batch slot')
        records.append({'ids': output, **metrics})
    if cross_group:
        for slot in (0, len(prompts) // 2):
            with engine.slot_q[slot].mutex:
                require(not any(line.startswith('BDONE ') for line in engine.slot_q[slot].queue),
                        'pipeline groups did not have concurrent active requests')
    for slot, record in enumerate(records):
        record['finish'] = finish(engine, slot, record['ids'])
        require(record['finish'] in ('stop', 'length'), 'batch request failed')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'engine', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--slots', type=int, default=2)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=4096)
    args = parser.parse_args()
    args.output.mkdir(parents=False, exist_ok=False)
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    tok = load_tokenizer(Path(cfg['tokenizer']))
    template = ChatTemplate(ROOT / 'serve/chat_template.jinja')
    prompts = []
    for slot in range(args.slots):
        label = chr(ord('A') + slot)
        sentence = f'Conversation {label}: the archive contains blue folders and green notebooks. '
        text = sentence * (args.tokens // len(tok.encode(sentence)) + 1)
        text += '\nList 100 practical ways to organize this archive, with a detailed explanation of each.'
        prompts.append(tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False),
                                  parse_special=True))
    base = private_args(cfg)
    base.remove('--vision')
    base += ['--batch', str(args.slots), '--batch-groups', str(args.groups), '--pcie-frac', '0',
             '--adapt-every', '1000000', '--no-prefill-borrow']
    env = child_env(cfg)
    env['STRATA_IQ_MT_MIN'] = '1'
    results = {'slots': args.slots, 'groups': args.groups, 'input_tokens': list(map(len, prompts))}
    count = 128 if args.groups > 1 else 12
    def chat(text):
        return tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
    cancel_prompt = chat('Write a long essay about music.')
    other_prompt = chat('Name three colors.')
    def open_engine(name, persist):
        command = list(base)
        if persist:
            command += ['--kv-persist', '--kv-persist-dir', str(args.output / (persist + '-kv')),
                        '--kv-persist-identity', 'private-batch-v1', '--kv-persist-max-mib', '30720']
        engine = StrataEngine(str(args.engine.resolve()), command, cwd=cfg.get('cwd'), env=env,
                              log=str(args.output / (name + '.log')))
        engine.close_s = 300
        require(engine.batch == args.slots, 'configured batch slots did not fit')
        results[name + '_info'] = dict(engine.info)
        return engine
    reference = open_engine('reference', 'reference' if args.groups > 1 else None)
    try:
        warm = pair(reference, prompts, count)
        continued = [p + r['ids'] for p, r in zip(prompts, warm)]
        expected = pair(reference, continued, count)
        if args.groups == 1:
            require(all(r['reused'] > 0 and r['kv_persist_read_bytes'] == 0 for r in expected),
                    'resident slot continuation did not reuse memory')
        # GEN exercises a restored batch history on the single-request MTP path.
        solo_prompt = continued[0] + expected[0]['ids']
        solo_expected = [t for t in reference.generate(solo_prompt, 16, {'temperature': 0}, threading.Event())
                         if t is not None]
        results['reference'] = expected
        results['solo_reference'] = solo_expected
        if args.groups == 1:
            cancel_first = [t for t in reference.generate(cancel_prompt, 1, {'temperature': 0}, threading.Event())
                            if t is not None]
    finally:
        reference.close()
    candidate = open_engine('candidate', 'candidate')
    try:
        actual_warm = pair(candidate, prompts, count, cross_group=args.groups > 1)
        require([r['ids'] for r in warm] == [r['ids'] for r in actual_warm], 'initial batch output differs')
        if args.groups == 1:
            # C is cancelled in a slot, then completed on the solo path. Its prompt checkpoint must survive.
            output, metrics, running = admission(candidate, 0, cancel_prompt, 2000)
            require(running, 'cancellation request did not enter a slot')
            require(metrics['kv_persist_write_bytes'] > 0, 'slot overwrite did not commit completed state')
            candidate._send('BSTOP 0')
            require(finish(candidate, 0, output) == 'cancel', 'slot was not cancelled')
            saved = re.findall(r'KV_PERSIST saved .*?tokens=(\d+)', (args.output / 'candidate.log').read_text())
            require(len(saved) == 1 and int(saved[0]) == len(prompts[0]) + count - 1, 'cancelled slot was committed')
            list(candidate.generate(other_prompt, 1, {'temperature': 0}, threading.Event()))
            resumed = [t for t in candidate.generate(cancel_prompt + output, 16, {'temperature': 0}, threading.Event())
                       if t is not None]
            require(candidate.last['reused'] >= len(cancel_prompt) + len(output) - 1, 'cancelled slot was not reused')
            results['cancelled_to_solo'] = {'partial': output, 'completed': resumed}
    finally:
        candidate.close()
    if args.groups == 1:
        saved = re.findall(r'KV_PERSIST saved .*?tokens=(\d+)', (args.output / 'candidate.log').read_text())
        require(sorted(map(int, saved)) == sorted([len(p) + count - 1 for p in prompts] +
                    [len(other_prompt), len(cancel_prompt) + len(output) + len(resumed) - 1]),
                'committed conversations differ from completed work')
    restored = open_engine('restored', 'candidate')
    try:
        actual = pair(restored, continued, count)
        require([r['ids'] for r in expected] == [r['ids'] for r in actual], 'restored batch output differs')
        require(all(r['kv_persist_restored_tokens'] > 0 and r['kv_persist_read_bytes'] > 0 for r in actual),
                'a completed slot did not restore actual disk state')
        results['restored'] = actual
        if args.groups == 1:
            restarted = [t for t in restored.generate(cancel_prompt, 1, {'temperature': 0}, threading.Event())
                         if t is not None]
            require(restarted == cancel_first, 'restored solo prompt output differs')
            turn_end = cancel_prompt.index(tok.encode('<|im_end|>', parse_special=True)[0]) + 1
            require(restored.last['kv_persist_restored_tokens'] >= turn_end and
                    restored.last['kv_persist_read_bytes'] > 0, 'slot-to-solo prompt checkpoint did not restore')
            results['slot_to_solo_checkpoint'] = dict(restored.last)
    finally:
        restored.close()
    solo = open_engine('solo-restored', 'candidate')
    try:
        actual_solo = [t for t in solo.generate(solo_prompt, 16, {'temperature': 0}, threading.Event()) if t is not None]
        require(actual_solo == solo_expected, 'restored batch history differs on the solo MTP path')
        require(solo.last['kv_persist_read_bytes'] > 0, 'solo request did not restore disk state')
        results['solo_restored'] = {'ids': actual_solo, **solo.last}
    finally:
        solo.close()
    results['passed'] = True
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {args.slots} batch slots, {args.groups} groups, restart and solo MTP continuation', flush=True)


if __name__ == '__main__':
    main()
