# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Check PR #587 through normal SGLang startup, HTTP requests and kvctl.

This experiment can produce a negative result. It never injects revisioned
control into the scheduler or calls an allocator/manager method directly.
"""

import argparse
import fcntl
import hashlib
import json
import os
import random
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def http(port, path, body=None, timeout=20):
    request = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}',
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload, status = response.read(), response.status
    except urllib.error.HTTPError as error:
        payload, status = error.read(), error.code
    try:
        result = json.loads(payload)
    except ValueError:
        result = {'raw': payload.decode(errors='replace')}
    return status, result


def generate(port, tokens, count):
    status, result = http(port, '/generate', {
        'input_ids': tokens,
        'sampling_params': {'temperature': 0, 'max_new_tokens': count,
                            'ignore_eos': True}}, timeout=180)
    assert status == 200 and len(result.get('output_ids', [])) == count, (status, result)
    return result


def sample(ipc):
    with (Path('/dev/shm') / ipc).open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        limit, used, reserved = struct.unpack('=qqq', stream.read(24))
    return dict(monotonic=time.monotonic(), limit_bytes=limit,
                used_bytes=used, reserved_bytes=reserved)


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def make_inputs(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    words = 'alpha beta gamma delta epsilon river mountain cloud engine memory page block cache tensor kernel graph scheduler request token stream buffer vector matrix'.split()

    def prompt(length, seed):
        generator = random.Random(seed)
        text = ' '.join(generator.choice(words) for _ in range(length))
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert len(tokens) >= length
        return tokens[:length]

    return {'warm': [prompt(384, 18000 + index) for index in range(28)],
            'probe': prompt(10000, 29000)}


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', args.port))
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert hasattr(fcntl, 'F_OFD_SETLK') and hasattr(fcntl, 'F_OFD_GETLK')
    inputs = make_inputs(args.model)
    if args.inputs:
        assert inputs == json.loads(args.inputs.read_text()), 'Inputs differ from original experiments'
    save(args.out / 'inputs.json', inputs)
    enabled = args.variant != 'native'
    ipc = f'pr587_public_{args.variant}_{args.scenario}_{os.getpid()}'
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('QUOTA_') or key in ('LD_PRELOAD', 'PYTHONSTARTUP'):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(args.source),
               ENABLE_KVCACHED='true' if enabled else 'false',
               KVCACHED_AUTOPATCH='1' if enabled else '0', KVCACHED_IPC_NAME=ipc,
               KVCACHED_MAX_CACHED_TOKENS='-1', KVCACHED_MAX_CACHED_BLOCKS='-1',
               KVCACHED_CONTIGUOUS_LAYOUT='false', KVCACHED_GPU_UTILIZATION='0.95',
               SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS='false', HF_HUB_OFFLINE='1')
    # Exercise the normal preallocation default, identically in all cases.
    env.pop('KVCACHED_PAGE_PREALLOC_ENABLED', None)
    command = [args.python, '-m', 'sglang.launch_server', '--model-path', str(args.model),
               '--host', '127.0.0.1', '--port', str(args.port), '--base-gpu-id', '0',
               '--tp-size', '1', '--random-seed', '42', '--enable-cache-report',
               '--dtype', 'bfloat16', '--context-length', '16384',
               '--max-total-tokens', '32768', '--max-running-requests', '16',
               '--mem-fraction-static', '0.8', '--attention-backend', 'triton',
               '--sampling-backend', 'pytorch', '--disable-cuda-graph',
               '--chunked-prefill-size', '4096']
    row: dict[str, Any] = dict(
        variant=args.variant, scenario=args.scenario, command=command,
        ipc_name=ipc, engine_instrumentation='none', completed=0,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment={key: value for key, value in env.items()
                     if key.startswith(('KVCACHED_', 'SGLANG_')) or key in (
                         'ENABLE_KVCACHED', 'CUDA_VISIBLE_DEVICES', 'CUDA_HOME',
                         'PYTHONPATH', 'LD_LIBRARY_PATH', 'CPATH', 'HF_HUB_OFFLINE')},
        source_sha256={name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                       for name in ['kvcached/kv_cache_manager.py', 'csrc/allocator.cpp',
                                    'csrc/page_allocator.cpp']},
        backend_sha256={file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                        for file in (args.source / 'kvcached').glob('_C*.so')},
        classification='incomplete', observations=[])
    assert row['backend_sha256']
    save(args.out / 'launch.json', row)
    process = None
    pool = ThreadPoolExecutor(max_workers=1)

    def observe(phase):
        state = sample(ipc)
        row['observations'].append(dict(phase=phase, **state))
        return state

    def limit(mib):
        control = subprocess.run(
            [args.python, '-m', 'kvcached.cli.kvctl', 'limit', ipc, f'{mib}M'],
            env=env, capture_output=True, text=True, timeout=30)
        row.setdefault('controls', []).append(dict(command=control.args,
            code=control.returncode, stdout=control.stdout, stderr=control.stderr))
        assert control.returncode == 0, row['controls'][-1]
        assert observe('limit-published')['limit_bytes'] == mib * 1024**2

    def checked_http(path, body):
        result = http(args.port, path, body)
        row.setdefault('http_controls', []).append(dict(path=path, body=body,
                                                       status=result[0], result=result[1]))
        assert result[0] == 200, result
        return result

    try:
        with (args.out / 'server.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            row['server_pid'] = process.pid
            deadline = time.monotonic() + 420
            while True:
                assert process.poll() is None, 'Server exited before readiness'
                try:
                    if http(args.port, '/health', timeout=2)[0] == 200:
                        break
                except (OSError, TimeoutError):
                    pass
                assert time.monotonic() < deadline, 'Server startup timed out'
                time.sleep(0.5)
            row['ready'] = True
            row['warm'] = [generate(args.port, tokens, 1) for tokens in inputs['warm']]
            row['completed'] = 28
            row['before'] = generate(args.port, inputs['probe'], 32)
            row['completed'] += 1
            if enabled:
                row['before_limit'] = observe('before-limit')
                if args.scenario == 'paused':
                    checked_http('/pause_generation', {'mode': 'retract'})
                limit(96)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    observe('before-flush')
                    time.sleep(0.05)
                row['before_flush'] = observe('before-flush-final')
            row['flush'] = checked_http('/flush_cache', {})
            if enabled:
                row['after_flush'] = observe('after-flush')
                if args.scenario == 'paused':
                    checked_http('/continue_generation', {'torch_empty_cache': False})
            row['small'] = generate(args.port, inputs['warm'][0], 1)
            row['completed'] += 1
            if not enabled:
                row['after'] = generate(args.port, inputs['probe'], 32)
                row['completed'] += 1
                row['classification'] = 'native_reference_completed'
            else:
                row['after_small'] = observe('after-small-request')
                pending = pool.submit(generate, args.port, inputs['probe'], 32)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    observe('large-request-under-96MiB')
                    if pending.done():
                        break
                    time.sleep(0.05)
                row['large_completed_under_96MiB'] = pending.done()
                row['before_growth'] = observe('before-growth')
                if pending.done():
                    row['after'] = pending.result()
                    row['classification'] = 'candidate_quota_escape_requires_memory_assessment'
                else:
                    limit(192)
                    time.sleep(1)
                    row['completed_on_growth_alone'] = pending.done()
                    if not pending.done():
                        checked_http('/pause_generation', {'mode': 'retract'})
                        checked_http('/continue_generation', {'torch_empty_cache': False})
                        row['public_retry_used'] = True
                    row['after'] = pending.result(timeout=90)
                    row['classification'] = 'quota_enforced_no_normal_serving_reproduction'
                row['completed'] += 1
                row['final'] = observe('after-large-request')
    except BaseException as error:
        row['classification'] = 'experiment_error'
        row['error'] = repr(error)
        raise
    finally:
        if process is not None:
            stop(process)
            row['server_exit'] = process.returncode
        pool.shutdown(wait=True, cancel_futures=True)
        save(args.out / 'result.json', row)
    print(json.dumps({key: row[key] for key in ['variant', 'scenario', 'completed', 'classification']}))


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['original', 'fixed', 'native'], required=True)
    parser.add_argument('--scenario', choices=['idle', 'paused'], default='idle')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--port', type=int, default=18879)
    args = parser.parse_args()
    for name in ['source', 'model', 'out']:
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == '__main__':
    main()
