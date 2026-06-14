"""Shared helpers: JSON IO, resumable parallel map."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm


def save_json(data, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_resume(instances, output_path, key="id"):
    """Merge previously saved fields back into instances (resume support)."""
    if not os.path.exists(output_path):
        return instances
    try:
        saved = load_json(output_path)
    except Exception:  # noqa: BLE001
        return instances
    saved_map = {it.get(key): it for it in saved if it.get(key) is not None}
    for i, it in enumerate(instances):
        k = it.get(key)
        if k in saved_map:
            merged = dict(it)
            merged.update(saved_map[k])
            instances[i] = merged
    return instances


def parallel_process(instances, fn, output_path, num_threads=8,
                     save_every=True, desc="processing"):
    """Run fn(item, index) over instances with threads, saving incrementally."""
    results = list(instances)
    lock = Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        fut2idx = {ex.submit(fn, it, i): i for i, it in enumerate(results)}
        for fut in tqdm(as_completed(fut2idx), total=len(results), desc=desc):
            idx = fut2idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[worker_error] idx={idx}: {type(e).__name__}: {e}")
            done += 1
            if save_every and done % 20 == 0:
                with lock:
                    save_json(results, output_path)
    save_json(results, output_path)
    return results
