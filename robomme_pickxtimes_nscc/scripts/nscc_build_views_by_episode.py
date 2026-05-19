#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pickle
import shutil


TASK_SPECS = {
    "patternlock": {"task": "PatternLock", "episode_ranges": [(0, 99)]},
    "buttonunmaskswap": {"task": "ButtonUnmaskSwap", "episode_ranges": [(100, 199)]},
    "buttonunmask": {"task": "ButtonUnmask", "episode_ranges": [(200, 299)]},
    "videoplacebutton": {"task": "VideoPlaceButton", "episode_ranges": [(300, 399)]},
    "videounmask": {"task": "VideoUnmask", "episode_ranges": [(400, 499)]},
    "pickxtimes": {"task": "PickXtimes", "episode_ranges": [(500, 599)]},
    "stopcube": {"task": "StopCube", "episode_ranges": [(600, 699)]},
    "swingxtimes": {"task": "SwingXtimes", "episode_ranges": [(700, 799)]},
    "pickhighlight": {"task": "PickHighlight", "episode_ranges": [(800, 899)]},
    "movecube": {"task": "MoveCube", "episode_ranges": [(900, 999)]},
    "insertpeg": {"task": "InsertPeg", "episode_ranges": [(1000, 1099)]},
    "routestick": {"task": "RouteStick", "episode_ranges": [(1100, 1199)]},
    "binfill": {"task": "BinFill", "episode_ranges": [(1200, 1299)]},
    "videoplaceorder": {"task": "VideoPlaceOrder", "episode_ranges": [(1300, 1399)]},
    "videorepick": {"task": "VideoRepick", "episode_ranges": [(1400, 1499)]},
    "videounmaskswap": {"task": "VideoUnmaskSwap", "episode_ranges": [(1500, 1599)]},
}


def read_item(data_dir: Path, idx: int) -> dict:
    with (data_dir / f"{idx}.pkl").open("rb") as f:
        return pickle.load(f)


def read_episode(data_dir: Path, idx: int) -> int:
    item = read_item(data_dir, idx)
    value = item["epis_idx"]
    try:
        return int(value.reshape(-1)[0])
    except AttributeError:
        return int(value)


def first_index_ge_episode(data_dir: Path, max_idx: int, target: int) -> int:
    lo, hi = 0, max_idx + 1
    while lo < hi:
        mid = (lo + hi) // 2
        ep = read_episode(data_dir, mid)
        if ep < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def reset_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def view_complete(dst: Path) -> bool:
    stats_path = dst / "meta" / "stats.json"
    if not stats_path.exists():
        return False
    try:
        stats = json.loads(stats_path.read_text())
    except Exception:
        return False
    return int(stats.get("execution_samples", 0)) > 100 and int(stats.get("num_episodes", 0)) > 0


def build_view(src: Path, out_root: Path, slug: str, suffix: str, overwrite: bool) -> dict:
    spec = TASK_SPECS[slug]
    data_src = src / "data"
    features_src = src / "features"
    dst = out_root / f"robomme_preprocessed_{slug}_{suffix}"

    if view_complete(dst) and not overwrite:
        stats = json.loads((dst / "meta" / "stats.json").read_text())
        print(f"[skip] {slug}: existing complete view with {stats['execution_samples']} samples")
        return stats

    max_idx = max(int(p.stem) for p in data_src.glob("*.pkl"))
    data_dst = dst / "data"
    meta_dst = dst / "meta"
    features_dst = dst / "features"
    reset_dir(data_dst)
    meta_dst.mkdir(parents=True, exist_ok=True)

    if features_dst.exists() or features_dst.is_symlink():
        if features_dst.is_symlink() or features_dst.is_file():
            features_dst.unlink()
        else:
            shutil.rmtree(features_dst)
    features_dst.symlink_to(features_src.resolve(), target_is_directory=True)

    source_indices = []
    episodes = {}
    prompts = {}
    for start_ep, end_ep in spec["episode_ranges"]:
        start_idx = first_index_ge_episode(data_src, max_idx, start_ep)
        stop_idx = first_index_ge_episode(data_src, max_idx, end_ep + 1)
        print(f"[range] {slug}: episodes {start_ep}-{end_ep} -> source indices {start_idx}-{stop_idx - 1}")
        for src_idx in range(start_idx, stop_idx):
            source_indices.append(src_idx)

    for out_idx, src_idx in enumerate(source_indices):
        src_file = data_src / f"{src_idx}.pkl"
        (data_dst / f"{out_idx}.pkl").symlink_to(src_file.resolve())

    # Build stats from one binary-search interval per episode, plus a first
    # sample prompt. This is cheap and avoids reopening every sample.
    for start_ep, end_ep in spec["episode_ranges"]:
        for ep in range(start_ep, end_ep + 1):
            start_idx = first_index_ge_episode(data_src, max_idx, ep)
            stop_idx = first_index_ge_episode(data_src, max_idx, ep + 1)
            if stop_idx <= start_idx:
                continue
            episodes[str(ep)] = stop_idx - start_idx
            prompt = str(read_item(data_src, start_idx).get("prompt", ""))
            prompts[prompt] = prompts.get(prompt, 0) + (stop_idx - start_idx)

    stats = {
        "task": spec["task"],
        "slug": slug,
        "episode_ranges": spec["episode_ranges"],
        "execution_samples": len(source_indices),
        "total_samples": len(source_indices),
        "episodes": dict(sorted(episodes.items(), key=lambda kv: int(kv[0]))),
        "num_episodes": len(episodes),
        "num_prompts": len(prompts),
        "prompts": dict(sorted(prompts.items(), key=lambda kv: (-kv[1], kv[0]))[:50]),
        "source_dataset": str(src),
        "view_dataset": str(dst),
        "build_method": "episode_range_binary_search",
    }
    (meta_dst / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))
    if len(source_indices) < 100:
        raise SystemExit(f"too few samples for {slug}: {len(source_indices)}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/scratch/users/nus/<NSCC_USER>/robomme_nscc/data/robomme_preprocessed_data")
    parser.add_argument("--output-root", default="/scratch/users/nus/<NSCC_USER>/robomme_nscc/data")
    parser.add_argument("--slugs", default="stopcube,videoplaceorder,videounmask")
    parser.add_argument("--suffix", default="full")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.source).resolve()
    out_root = Path(args.output_root).resolve()
    all_stats = {}
    for slug in [s.strip() for s in args.slugs.split(",") if s.strip()]:
        if slug not in TASK_SPECS:
            raise ValueError(f"unknown slug: {slug}")
        stats = build_view(src, out_root, slug, suffix=args.suffix, overwrite=args.overwrite)
        all_stats[slug] = stats

    summary_path = out_root / f"robomme_task_views_episode_range_summary_{args.suffix}.json"
    summary_path.write_text(json.dumps(all_stats, indent=2, sort_keys=True))
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
