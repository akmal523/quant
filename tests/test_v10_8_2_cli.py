"""test_v10_8_2_cli.py — six user commands, internals hidden (v10.8.2, section 9)."""
from __future__ import annotations

import re

from quant.cli import build_parser

USER_COMMANDS = {"dash", "refresh", "daily", "upgrade", "backup", "doctor"}
HIDDEN = {"run", "reconcile", "all", "news-doctor",
          "validate-tiers", "repair-tiers", "health-check",
          "clear-cache", "cache-stats", "schedule",
          "notify-setup", "ack"}


def test_help_lists_only_the_six_user_commands():
    text = build_parser().format_help()
    listed = set(re.findall(r"^ {4}([a-z][\w-]*)", text, re.M))
    assert listed == USER_COMMANDS


def test_hidden_commands_still_parse():
    # They remain dispatchable for the scheduler/tests; only the help is hidden.
    extras = {"ack": ["1", "--status", "done"]}
    for name in HIDDEN:
        args = build_parser().parse_args([name, *extras.get(name, [])])
        assert args.command == name
