#!/usr/bin/env python3
"""Utility to visualise RDX base-class and attribute distributions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt

from plugin.datasets.map_utils.rdx_schema import (
    BASE_CLASSES,
    ATTRIBUTE_SCHEMAS,
    normalize_category_and_attributes,
)


def collect_counts(annotation_root: str) -> Tuple[Counter, Dict[str, Counter]]:
    """Scan all JSON files under ``annotation_root`` and accumulate counts."""
    class_counter: Counter = Counter()
    attribute_counters: Dict[str, Counter] = {
        name: Counter() for name in ATTRIBUTE_SCHEMAS.keys()
    }

    for dirpath, _, filenames in os.walk(annotation_root):
        for fname in filenames:
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            for inst in data.get('instances', []):
                category = inst.get('category')
                for shape in inst.get('shapes', []):
                    base_class, attrs = normalize_category_and_attributes(
                        category,
                        shape.get('attributes', {}),
                    )
                    if base_class is None:
                        continue

                    class_counter[base_class] += 1
                    for attr_name, schema in ATTRIBUTE_SCHEMAS.items():
                        applies_to = schema['applies_to']  # type: ignore[index]
                        if base_class not in applies_to:
                            continue
                        value = attrs.get(attr_name, schema['default'])  # type: ignore[index]
                        attribute_counters[attr_name][value] += 1

    return class_counter, attribute_counters


def plot_bar(
    labels,
    counts: Counter,
    title: str,
    output_path: str,
):
    values = [counts.get(label, 0) for label in labels]
    width = max(6.0, len(labels) * 0.8)
    plt.figure(figsize=(width, 4.0))
    plt.bar(range(len(labels)), values, color='skyblue')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot base-class and attribute histograms for RDX annotations.'
    )
    parser.add_argument(
        '--annotation-root',
        required=True,
        help='Directory containing RDX annotation JSON files.',
    )
    parser.add_argument(
        '--output-dir',
        default='rdx_histograms',
        help='Directory to store histogram plots.',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    class_counts, attr_counts = collect_counts(args.annotation_root)

    # Base classes
    plot_bar(
        labels=BASE_CLASSES,
        counts=class_counts,
        title='RDX Base Class Counts',
        output_path=os.path.join(args.output_dir, 'base_classes.png'),
    )

    # Attributes
    for attr_name, schema in ATTRIBUTE_SCHEMAS.items():
        labels = list(schema['classes'])  # type: ignore[index]
        plot_bar(
            labels=labels,
            counts=attr_counts[attr_name],
            title=f'{attr_name} Distribution',
            output_path=os.path.join(args.output_dir, f'{attr_name}.png'),
        )

    # Also dump raw counts for inspection.
    summary_path = os.path.join(args.output_dir, 'counts.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'classes': dict(class_counts),
                'attributes': {
                    attr: dict(counter) for attr, counter in attr_counts.items()
                },
            },
            f,
            indent=2,
        )
    print(f'Saved histograms and counts to "{args.output_dir}".')


if __name__ == '__main__':
    main()

