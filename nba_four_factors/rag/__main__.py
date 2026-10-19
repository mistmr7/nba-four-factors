"""CLI for the research RAG.

python -m nba_four_factors.rag build
python -m nba_four_factors.rag ask "What caused the 2017-18 calibration break?"
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    ask_p = sub.add_parser("ask")
    ask_p.add_argument("question")
    ask_p.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    if args.cmd == "build":
        from .index import build_index

        build_index()
    else:
        from .ask import TOP_K, answer

        answer(args.question, args.k or TOP_K)


if __name__ == "__main__":
    main()
