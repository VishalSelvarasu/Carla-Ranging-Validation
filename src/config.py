from __future__ import annotations

import argparse
import os
from functools import lru_cache

import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "conditions.yaml")


@lru_cache(maxsize=1)
def load(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("conditions", "baseline", "evaluation"):
        if key not in cfg:
            raise ValueError(f"{path} missing required key '{key}'")
    names = [c["name"] for c in cfg["conditions"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate condition names")
    if cfg["baseline"] not in names:
        raise ValueError(
            f"baseline '{cfg['baseline']}' is not in the condition list")
    return cfg


def condition_names() -> list[str]:
    return [c["name"] for c in load()["conditions"]]


def severity_map() -> dict[str, int]:
    return {c["name"]: c.get("severity", 0) for c in load()["conditions"]}


def baseline() -> str:
    return load()["baseline"]


def eval_params() -> dict:
    return load()["evaluation"]


def sort_by_severity(names):
    """
    Order conditions by severity, then name.

    Results tables must be monotone in severity or a reader cannot see the
    degradation trend without re-sorting by hand.
    """
    sev = severity_map()
    return sorted(names, key=lambda n: (sev.get(n, 99), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--severity", metavar="NAME")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.names:
        print("\n".join(condition_names()))
    elif args.baseline:
        print(baseline())
    elif args.severity:
        print(severity_map().get(args.severity, ""))
    else:
        cfg = load()
        print(f"config: {CONFIG_PATH}")
        print(f"baseline: {cfg['baseline']}")
        print(f"conditions ({len(cfg['conditions'])}):")
        for c in cfg["conditions"]:
            print(
                f"  [{c.get('severity', 0)}] {c['name']:16s} {c.get('note', '')}")
        print(f"evaluation: {cfg['evaluation']}")


if __name__ == "__main__":
    main()
