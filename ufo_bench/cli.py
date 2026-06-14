"""Unified command-line interface for the UFO benchmark toolkit.

Subcommands:
    ufo-eval download ...  pre-download the UFO dataset from the HF Hub
    ufo-eval infer   ...   generate cues + answers under the 4 protocols
    ufo-eval score   ...   score predictions (MCQ letter-match / open LLM judge)
    ufo-eval tables  ...   aggregate scored files into CSV + LaTeX + Markdown
    ufo-eval cue-eval ...  score generated-cue quality vs ground truth
    ufo-eval run     ...   one command: infer -> score -> tables (per config)

Run `ufo-eval <subcommand> -h` for options. The same commands are available as
`python -m ufo_bench ...`.
"""

import argparse
import glob
import os

from .aggregate import (
    accuracy_by_category,
    accuracy_by_task,
    overall_accuracy,
    summary_markdown,
    to_latex_main,
    write_csv,
)
from .config import PROTOCOLS, QUESTION_TYPES
from .cue_eval import evaluate_cues
from .data import filter_instances, load_any, load_hf
from .inference import Reasoner, model_tag
from .registry import (
    build_judge,
    build_model_providers,
    load_registry,
    resolve_models,
)
from .runner import load_json, merge_resume, parallel_process, save_json
from .scoring import score_item


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_env():
    """Load a local .env if python-dotenv is installed (optional)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for path in (".env", os.path.join(os.getcwd(), ".env")):
        if os.path.exists(path):
            load_dotenv(path)
            return


def detect_model_tag(instances):
    """Infer the model tag from `score_<protocol>_<tag>` / `pred_<protocol>_<tag>`
    keys in a results file."""
    for it in instances:
        for k in it:
            for prefix in ("score_", "pred_"):
                if k.startswith(prefix):
                    rest = k[len(prefix):]
                    proto, _, tag = rest.partition("_")
                    if proto in PROTOCOLS and tag:
                        return tag
    return None


def _load_run_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def cmd_download(args):
    """Pre-download the UFO benchmark from the HF Hub and materialize images to a
    local cache, so later runs are offline and fast."""
    splits = args.splits or list(QUESTION_TYPES)
    for split in splits:
        cache_dir = os.path.join(args.out, split)
        print(f"Downloading {args.repo} [{split}] -> {cache_dir} ...")
        instances = load_hf(args.repo, split=split, cache_dir=cache_dir)
        print(f"  {len(instances)} instances cached "
              f"(images under {os.path.join(cache_dir, 'images')}).")
    print(f"\nDone. Now run e.g.:  ufo-eval run --source {args.repo} ...")


# --------------------------------------------------------------------------- #
# infer
# --------------------------------------------------------------------------- #
def cmd_infer(args):
    cfg = load_registry(args.config)
    models = resolve_models(cfg, args.models)
    instances = load_any(args.source, image_root=args.image_root, split=args.split)
    instances = filter_instances(instances, tasks=args.tasks, limit=args.limit)
    print(f"Loaded {len(instances)} instances from {args.source}.")

    if args.dry_run:
        print("\n[dry-run] Plan:")
        print(f"  split      : {args.split}")
        print(f"  protocols  : {args.protocols}")
        print(f"  out        : {args.out}")
        print(f"  models     : {[m['name'] for m in models]}")
        for m in models:
            try:
                build_model_providers(m, cfg, api_key=args.api_key)
                status = "providers OK"
            except Exception as e:  # noqa: BLE001
                status = f"provider ERROR: {type(e).__name__}: {e}"
            print(f"    - {m['name']:18s} [{m.get('provider', 'openai')}] {status}")
        print("\n[dry-run] No API calls made.")
        return []

    os.makedirs(args.out, exist_ok=True)
    written = []
    for m in models:
        print(f"\n=== Model: {m['name']} ({m.get('id', m.get('model_path'))}) "
              f"[provider={m.get('provider', 'openai')}] ===")
        try:
            answer_p, image_p = build_model_providers(m, cfg, api_key=args.api_key)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] cannot build provider: {e}")
            continue
        reasoner = Reasoner(
            answer_p, image_provider=image_p, protocols=args.protocols,
            model_name=m["name"],
            output_img_dir=os.path.join(args.out, "generated_cues"),
        )
        out_path = os.path.join(args.out, f"{model_tag(m['name'])}.json")
        data = merge_resume([dict(x) for x in instances], out_path)
        parallel_process(data, reasoner.process, out_path,
                         num_threads=args.num_threads, desc=m["name"])
        print(f"Saved -> {out_path}")
        written.append(out_path)
    return written


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def cmd_score(args):
    data = load_json(args.pred)
    tag = model_tag(args.model) if args.model else detect_model_tag(data)
    if not tag:
        raise SystemExit(
            f"Could not detect model tag in {args.pred}; pass --model explicitly.")
    print(f"Scoring model tag: {tag}")

    qtypes = {it.get("question_type") for it in data}
    needs_judge = "open" in qtypes
    judge = None
    if needs_judge:
        cfg = load_registry(args.config)
        judge = build_judge(cfg, judge_model=args.judge_model, api_key=args.api_key)
        print(f"Open-ended scoring uses judge: "
              f"{args.judge_model or cfg.get('judge_model')}")

    out_path = args.out or args.pred.replace(".json", "_scored.json")

    def fn(item, _idx):
        return score_item(item, tag, args.protocols, judge_client=judge)

    if needs_judge:
        results = parallel_process(data, fn, out_path,
                                   num_threads=args.num_threads, desc="scoring")
    else:
        results = [fn(it, i) for i, it in enumerate(data)]
        save_json(results, out_path)

    for p in args.protocols:
        sk = f"score_{p}_{tag}"
        vals = [float(it[sk]) for it in results if sk in it]
        if vals:
            print(f"  {p:8s} acc = {sum(vals) / len(vals) * 100:.2f}  (N={len(vals)})")
    print(f"Saved -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def cmd_tables(args):
    files = sorted(glob.glob(args.scored))
    if not files:
        print(f"No files match: {args.scored}")
        return
    os.makedirs(args.out, exist_ok=True)

    per_model_task, per_model_cat, per_model_overall, order = {}, {}, {}, []
    for fp in files:
        data = load_json(fp)
        tag = detect_model_tag(data)
        if not tag:
            print(f"  [skip] no score keys in {fp}")
            continue
        name = os.path.splitext(os.path.basename(fp))[0].replace("_scored", "")
        order.append(name)
        per_model_task[name] = accuracy_by_task(data, tag, args.protocols)
        per_model_cat[name] = accuracy_by_category(data, tag, args.protocols)
        per_model_overall[name] = overall_accuracy(data, tag, args.protocols)
        print(f"  aggregated {name}  ({len(data)} items)")

    csv_path = os.path.join(args.out, "results.csv")
    write_csv(csv_path, per_model_task, args.protocols)
    print(f"CSV   -> {csv_path}")

    tex = to_latex_main(per_model_cat, order, args.protocols)
    with open(os.path.join(args.out, "main_table.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"LaTeX -> {os.path.join(args.out, 'main_table.tex')}")

    md = summary_markdown(per_model_overall, order, args.protocols)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# UFO results (overall accuracy %)\n\n" + md + "\n")
    print(f"\nOverall accuracy:\n{md}")
    print(f"\nMarkdown -> {os.path.join(args.out, 'summary.md')}")


# --------------------------------------------------------------------------- #
# cue-eval
# --------------------------------------------------------------------------- #
def cmd_cue_eval(args):
    cfg = load_registry(args.config)
    judge = build_judge(cfg, judge_model=args.judge_model, api_key=args.api_key)
    data = load_json(args.pred)
    tag = model_tag(args.model) if args.model else detect_model_tag(data)
    out_path = args.out or args.pred.replace(".json", "_cueeval.json")

    def fn(item, _idx):
        return evaluate_cues(item, tag, judge, targets=tuple(args.targets))

    results = parallel_process(data, fn, out_path,
                               num_threads=args.num_threads, desc="cue-eval")
    for tgt in args.targets:
        sk = f"cue_{tgt}_score_{tag}"
        vals = [float(it[sk]) for it in results if sk in it]
        if vals:
            print(f"  cue-{tgt:6s} acc = {sum(vals)/len(vals)*100:.2f} (N={len(vals)})")
    print(f"Saved -> {out_path}")


# --------------------------------------------------------------------------- #
# run (one-shot: infer -> score -> tables)
# --------------------------------------------------------------------------- #
def cmd_run(args):
    # YAML config provides defaults; explicit CLI flags override them.
    conf = _load_run_config(args.run_config) if args.run_config else {}
    source = args.source or conf.get("source", "yzzyu/UFO")
    split = args.split or conf.get("split", "mcq")
    models = args.models or conf.get("models")
    protocols = args.protocols or conf.get("protocols", list(PROTOCOLS))
    limit = args.limit if args.limit is not None else conf.get("limit", 0)
    out = args.out or conf.get("out", "outputs/run")
    num_threads = args.num_threads or conf.get("num_threads", 8)
    tasks = args.tasks or conf.get("tasks")
    judge_model = args.judge_model or conf.get("judge_model")

    infer_args = argparse.Namespace(
        config=args.config, models=models, source=source, split=split,
        image_root=None, tasks=tasks, limit=limit, protocols=protocols,
        num_threads=num_threads, out=out, api_key=args.api_key, dry_run=args.dry_run)
    pred_paths = cmd_infer(infer_args)
    if args.dry_run or not pred_paths:
        return

    for pred in pred_paths:
        score_args = argparse.Namespace(
            pred=pred, model=None, protocols=protocols, out=None,
            num_threads=num_threads, config=args.config,
            judge_model=judge_model, api_key=args.api_key)
        cmd_score(score_args)

    tables_args = argparse.Namespace(
        scored=os.path.join(out, "*_scored.json"),
        protocols=protocols, out=os.path.join(out, "tables"))
    cmd_tables(tables_args)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(prog="ufo-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--config", default=None, help="Path to models.yaml.")
        sp.add_argument("--api_key", default=None,
                        help="Override API key (else from env per provider).")

    # download
    sp = sub.add_parser("download", help="Pre-download the UFO dataset from HF.")
    sp.add_argument("--repo", default="yzzyu/UFO", help="HF dataset repo id.")
    sp.add_argument("--splits", nargs="+", default=None, choices=QUESTION_TYPES,
                    help="Splits to fetch (default: mcq and open).")
    sp.add_argument("--out", default="data_cache", help="Local cache directory.")
    sp.set_defaults(func=cmd_download)

    # infer
    sp = sub.add_parser("infer", help="Run inference (cues + answers).")
    sp.add_argument("--source", required=True, help="Local .jsonl OR HF repo id.")
    sp.add_argument("--split", default="mcq", choices=["mcq", "open"])
    sp.add_argument("--image_root", default=None)
    sp.add_argument("--models", nargs="+", default=None,
                    help="Names/ids, or 'all', or 'group:proprietary|open_source|unified'.")
    sp.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    sp.add_argument("--tasks", nargs="+", default=None)
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--num_threads", type=int, default=8)
    sp.add_argument("--out", default="outputs/inference")
    sp.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="Load data + build providers + print plan; no API calls.")
    add_common(sp)
    sp.set_defaults(func=cmd_infer)

    # score
    sp = sub.add_parser("score", help="Score predictions.")
    sp.add_argument("--pred", required=True)
    sp.add_argument("--model", default=None,
                    help="Model NAME used at inference; auto-detected if omitted.")
    sp.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    sp.add_argument("--out", default=None)
    sp.add_argument("--num_threads", type=int, default=8)
    sp.add_argument("--judge_model", default=None)
    add_common(sp)
    sp.set_defaults(func=cmd_score)

    # tables
    sp = sub.add_parser("tables", help="Aggregate scored files into tables.")
    sp.add_argument("--scored", required=True, help="Glob, e.g. 'outputs/mcq/*_scored.json'.")
    sp.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    sp.add_argument("--out", default="outputs/tables")
    sp.set_defaults(func=cmd_tables)

    # cue-eval
    sp = sub.add_parser("cue-eval", help="Score generated cue quality.")
    sp.add_argument("--pred", required=True)
    sp.add_argument("--model", default=None)
    sp.add_argument("--targets", nargs="+", default=["text", "visual"],
                    choices=["text", "visual"])
    sp.add_argument("--out", default=None)
    sp.add_argument("--num_threads", type=int, default=8)
    sp.add_argument("--judge_model", default=None)
    add_common(sp)
    sp.set_defaults(func=cmd_cue_eval)

    # run (one-shot)
    sp = sub.add_parser("run", help="One command: infer -> score -> tables.")
    sp.add_argument("--run-config", dest="run_config", default=None,
                    help="YAML run config (configs/run.yaml). CLI flags override it.")
    sp.add_argument("--source", default=None)
    sp.add_argument("--split", default=None, choices=[None, "mcq", "open"])
    sp.add_argument("--models", nargs="+", default=None)
    sp.add_argument("--protocols", nargs="+", default=None, choices=PROTOCOLS)
    sp.add_argument("--tasks", nargs="+", default=None)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--num_threads", type=int, default=None)
    sp.add_argument("--out", default=None)
    sp.add_argument("--judge_model", default=None)
    sp.add_argument("--dry-run", dest="dry_run", action="store_true")
    add_common(sp)
    sp.set_defaults(func=cmd_run)

    return p


def main(argv=None):
    load_env()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
