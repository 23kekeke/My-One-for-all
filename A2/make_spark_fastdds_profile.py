#!/usr/bin/env python3
"""Derive a Spark-local FastDDS profile from the active Orin profile.

Only interfaceWhiteList/address entries are changed. All vendor transport and
participant settings are preserved. The source file is never modified.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_xml", type=Path)
    parser.add_argument("output_xml", type=Path)
    parser.add_argument("--local-address", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parsed_address = ipaddress.ip_address(args.local_address)
    if parsed_address.version != 4:
        raise SystemExit("--local-address must be an IPv4 address")
    address = str(parsed_address)

    source = args.source_xml.resolve()
    output = args.output_xml.resolve()
    if not source.is_file():
        raise SystemExit(f"source XML not found: {source}")
    if source == output:
        raise SystemExit("source and output must be different files")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists (use --overwrite): {output}")

    source_text = source.read_text(encoding="utf-8")
    block_pattern = re.compile(
        r"(<interfaceWhiteList\b[^>]*>)(.*?)(</interfaceWhiteList>)",
        re.DOTALL,
    )
    blocks = list(block_pattern.finditer(source_text))
    if len(blocks) != 1:
        raise SystemExit(
            f"expected exactly one interfaceWhiteList, found {len(blocks)}"
        )

    block = blocks[0]
    body = block.group(2)
    address_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)<address>(?P<value>[^<]+)</address>[ \t]*$"
    )
    address_matches = list(address_pattern.finditer(body))
    old_addresses = [match.group("value").strip() for match in address_matches]
    if not old_addresses:
        raise SystemExit("source interfaceWhiteList contains no address")

    indentation = address_matches[0].group("indent")
    newline = "\r\n" if "\r\n" in source_text else "\n"
    new_body = (
        newline
        + indentation
        + f"<address>{address}</address>"
        + newline
        + indentation[:-2]
    )
    replacement = block.group(1) + new_body + block.group(3)
    output_text = source_text[: block.start()] + replacement + source_text[block.end() :]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(output_text, encoding="utf-8")
    ET.parse(temporary)
    os.replace(temporary, output)

    print(f"source={source}")
    print(f"old_interface_addresses={old_addresses}")
    print(f"new_interface_address={address}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
