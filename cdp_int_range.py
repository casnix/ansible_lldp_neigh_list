#!/usr/bin/env python3
"""
cdp_int_range.py

Given a CDP/LLDP neighbor CSV, filter rows by neighbor_device regex and
emit a summary CSV with each local_device and its Cisco-syntax interface
range for all matching ports.

Usage:
    python cdp_int_range.py \
        --csv-source neighbors.csv \
        --output ranges.csv \
        --filter "core-sw.*"
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Row = dict[str, str]
PortMap = dict[str, list[int]]          # prefix  -> sorted port numbers
RangeStr = str                          # final Cisco range string


# ---------------------------------------------------------------------------
# Interface parsing helpers
# ---------------------------------------------------------------------------

# Matches Cisco-style interface names, e.g.:
#   GigabitEthernet1/0/3  Gi1/0/3  TenGigabitEthernet1/1/1  Te2/1
#   FastEthernet0/1        Fa0/24   Ethernet1/5
_INTF_RE = re.compile(
    r"""
    ^
    (?P<prefix>
        (?:GigabitEthernet|TenGigabitEthernet|HundredGigE|
           FastEthernet|Ethernet|Gi|Te|Fa|Hu|Eth)
        [\d/]*?          # optional slot/module portion (greedy-to-last-/)
        [\d/]*           # keep consuming until the last digit group
    )
    (?P<port>\d+)        # final port number
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _split_intf(name: str) -> tuple[str, int] | None:
    """
    Split an interface name into (prefix, port_number).

    The prefix includes everything up to (but not including) the final
    digit sequence, e.g.:
        "GigabitEthernet1/0/3"  -> ("GigabitEthernet1/0/", 3)
        "Gi1/0/24"              -> ("Gi1/0/",              24)
        "FastEthernet0/1"       -> ("FastEthernet0/",       1)

    Returns None if the name cannot be parsed.
    """
    # Strip whitespace
    name = name.strip()
    # Find the last '/' or fall back to the boundary between letters and digits
    last_slash = name.rfind("/")
    if last_slash != -1:
        prefix = name[: last_slash + 1]   # e.g. "GigabitEthernet1/0/"
        suffix = name[last_slash + 1 :]   # e.g. "3"
        if suffix.isdigit():
            return prefix, int(suffix)
    # No slash — try letter/digit boundary (e.g. "Ethernet5")
    m = re.match(r"^([A-Za-z]+)(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2))
    return None


def _collapse_to_range_segments(prefix: str, numbers: list[int]) -> list[str]:
    """
    Collapse a sorted list of port integers into Cisco-correct range segments
    for a given interface prefix.

    Rules:
    - Consecutive runs of 2+ ports are written as ``<prefix><start> - <end>``
      (only the leading segment needs the full prefix for a run).
    - A single isolated port is written as ``<prefix><port>`` (full name,
      because Cisco does not allow bare port numbers without their module path
      in a non-consecutive position).

    Examples (prefix="Gi1/0/"):
        [1, 2, 3]          -> ["Gi1/0/1 - 3"]
        [1, 2, 3, 8]       -> ["Gi1/0/1 - 3", "Gi1/0/8"]
        [6, 37, 39]        -> ["Gi1/0/6", "Gi1/0/37", "Gi1/0/39"]
        [1, 2, 5, 6, 9]    -> ["Gi1/0/1 - 2", "Gi1/0/5 - 6", "Gi1/0/9"]
    """
    segments: list[str] = []
    for _, group in groupby(enumerate(sorted(set(numbers))), lambda t: t[1] - t[0]):
        run = list(group)
        start, end = run[0][1], run[-1][1]
        if start == end:
            segments.append(f"{prefix}{start}")
        else:
            segments.append(f"{prefix}{start} - {end}")
    return segments


def build_cisco_range(ports: list[str]) -> str:
    """
    Given a list of raw interface names (all from the same local_device),
    return a Cisco ``interface range`` value string.

    Cisco syntax rules applied:
    - Segments are separated by `` , `` (space before *and* after the comma).
    - Every segment, whether a consecutive run or a lone port, carries the
      full interface prefix (e.g. ``Gi1/0/``) so the IOS parser never sees
      a bare port number without its module path.

    Example output:
        Gi1/0/1 - 3 , Gi1/0/8 , TenGigabitEthernet1/1/1 - 2

    Interfaces that cannot be parsed are appended verbatim at the end.
    """
    prefix_map: PortMap = defaultdict(list)
    unparseable: list[str] = []

    for port in ports:
        parsed = _split_intf(port)
        if parsed is None:
            unparseable.append(port.strip())
        else:
            prefix, num = parsed
            prefix_map[prefix].append(num)

    all_segments: list[str] = []
    for prefix in sorted(prefix_map):
        all_segments.extend(_collapse_to_range_segments(prefix, prefix_map[prefix]))

    all_segments.extend(unparseable)
    return " , ".join(all_segments)


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"local_device", "local_port", "neighbor_device"}
)


def process(
    csv_source: Path,
    output: Path,
    filter_pattern: str,
) -> None:
    """Read *csv_source*, filter by *filter_pattern*, write *output*."""

    try:
        neighbor_re = re.compile(filter_pattern, re.IGNORECASE)
    except re.error as exc:
        sys.exit(f"Invalid regex filter: {exc}")

    # device -> list of local_port strings
    device_ports: dict[str, list[str]] = defaultdict(list)

    try:
        with csv_source.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)

            if reader.fieldnames is None:
                sys.exit("CSV file appears to be empty.")

            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                sys.exit(
                    f"CSV is missing required column(s): {', '.join(sorted(missing))}"
                )

            matched_rows = 0
            for lineno, row in enumerate(reader, start=2):  # 1-based, header is 1
                neighbor = row.get("neighbor_device", "").strip()
                if not neighbor_re.search(neighbor):
                    continue

                local_device = row.get("local_device", "").strip()
                local_port = row.get("local_port", "").strip()

                if not local_device:
                    print(
                        f"  Warning: line {lineno} has an empty local_device — skipping.",
                        file=sys.stderr,
                    )
                    continue
                if not local_port:
                    print(
                        f"  Warning: line {lineno} ({local_device}) has an empty "
                        f"local_port — skipping.",
                        file=sys.stderr,
                    )
                    continue

                device_ports[local_device].append(local_port)
                matched_rows += 1

    except FileNotFoundError:
        sys.exit(f"Source file not found: {csv_source}")
    except PermissionError:
        sys.exit(f"Permission denied reading: {csv_source}")

    if matched_rows == 0:
        print(
            f"No rows matched filter '{filter_pattern}'. Output file not written.",
            file=sys.stderr,
        )
        return

    try:
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["local_device", "int_range", "filtered_by"]
            )
            writer.writeheader()
            for device in sorted(device_ports):
                int_range = build_cisco_range(device_ports[device])
                writer.writerow(
                    {
                        "local_device": device,
                        "int_range": int_range,
                        "filtered_by": filter_pattern,
                    }
                )
    except PermissionError:
        sys.exit(f"Permission denied writing: {output}")

    print(
        f"Done. {matched_rows} matching row(s) across "
        f"{len(device_ports)} device(s) → {output}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a CDP/LLDP neighbor CSV by neighbor_device regex and "
            "output Cisco interface ranges per local_device."
        )
    )
    parser.add_argument(
        "--csv-source",
        required=True,
        metavar="PATH",
        type=Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        type=Path,
        help="Path for the output CSV file.",
    )
    parser.add_argument(
        "--filter",
        required=True,
        metavar="REGEX",
        help="Case-insensitive regex matched against the neighbor_device column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process(
        csv_source=args.csv_source,
        output=args.output,
        filter_pattern=args.filter,
    )


if __name__ == "__main__":
    main()