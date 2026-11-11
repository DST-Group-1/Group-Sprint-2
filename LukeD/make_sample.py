#!/usr/bin/env python
"""Make a small extract of train.parquet for developing the data notebook.

train.parquet is about 3 GB, which is awkward to pass around or to iterate on.
This script makes ONE streaming pass over it (memory stays low) and writes three
things into <repo>/enveda-CASMI26-molecule-id-mass-spectra-sample/, a folder the
existing .gitignore rule already keeps out of the repo:

  schema.txt                the Parquet schema and row-group layout
  train_meta_XX.parquet     every non-list column for EVERY row, plus the length
                            of each list column (e.g. number of peaks per
                            spectrum). No peak arrays, so it is small.
  train_sample_XX.parquet   complete rows, peaks included, for about 4% of
                            molecules. Molecules are chosen by a stable hash of
                            their key, so every spectrum of a chosen molecule is
                            kept and the same molecules are chosen on any machine.

Usage, from the repo root:

    .venv/bin/pip install pyarrow
    .venv/bin/python LukeD/make_sample.py

An optional argument gives a different path to train.parquet.
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "enveda-CASMI26-molecule-id-mass-spectra" / "train.parquet"
OUT_DIR = REPO / "enveda-CASMI26-molecule-id-mass-spectra-sample"

HASH_BUCKETS = 10_000        # each molecule key hashes to a bucket 0..9999
SAMPLE_BUCKETS = 400         # keep buckets below this: 400 / 10,000 = 4% of molecules
N_SAMPLE_PARTS = 16          # the sample is spread over this many small files
META_ROWS_PER_PART = 250_000
BATCH_ROWS = 20_000
# The column that identifies a molecule, in order of preference.
KEY_CANDIDATES = ["inchikey14", "inchikey", "molecule_id", "smiles"]


def is_list(dtype: pa.DataType) -> bool:
    return (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    )


def bucket_of(key: object, first14: bool) -> int:
    """Stable hash bucket of a molecule key (crc32 is the same on every machine)."""
    key = "" if key is None else str(key)
    if first14:
        key = key[:14]
    return zlib.crc32(key.encode("utf-8")) % HASH_BUCKETS


def main() -> None:
    train = Path(sys.argv[1]) if len(sys.argv) > 1 else TRAIN
    if not train.exists():
        sys.exit(f"Can't find {train}. Pass the path to train.parquet as an argument.")
    OUT_DIR.mkdir(exist_ok=True)

    pf = pq.ParquetFile(train, memory_map=True)
    md = pf.metadata
    schema = pf.schema_arrow

    # --- schema.txt ---------------------------------------------------------
    lines = [
        f"file: {train.name}",
        f"bytes: {train.stat().st_size:,}",
        f"rows: {md.num_rows:,}",
        f"row groups: {md.num_row_groups}",
        f"columns: {md.num_columns}",
        f"created by: {md.created_by}",
        "",
        "schema:",
    ]
    lines += [f"  {field.name}: {field.type}" for field in schema]
    lines += ["", "row groups (rows, uncompressed bytes):"]
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        lines.append(f"  {i}: {rg.num_rows:,} rows, {rg.total_byte_size:,} bytes")
    rg0 = md.row_group(0)
    lines += ["", "row group 0 columns (compression, compressed bytes, uncompressed bytes):"]
    for j in range(rg0.num_columns):
        col = rg0.column(j)
        lines.append(
            f"  {col.path_in_schema}: {col.compression}, "
            f"{col.total_compressed_size:,}, {col.total_uncompressed_size:,}"
        )
    (OUT_DIR / "schema.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8 + len(schema)]))

    # --- which columns go where ---------------------------------------------
    list_cols = [f.name for f in schema if is_list(f.type)]
    scalar_cols = [
        f.name for f in schema
        if not is_list(f.type) and not pa.types.is_nested(f.type)
    ]
    by_lower = {name.lower(): name for name in schema.names}
    key = next((by_lower[c] for c in KEY_CANDIDATES if c in by_lower), None)
    first14 = key is not None and key.lower() == "inchikey"
    print(f"\nmolecule key for sampling: {key!r}")
    if key is None:
        print("No molecule key found, so the sample is 4% of rows instead.")

    # --- one streaming pass ---------------------------------------------------
    part_width = SAMPLE_BUCKETS // N_SAMPLE_PARTS
    sample_writers: dict[int, pq.ParquetWriter] = {}
    sample_rows = 0
    sample_keys: set[str] = set()
    meta_writer: pq.ParquetWriter | None = None
    meta_part = 0
    meta_rows_in_part = 0
    rows_done = 0
    start = time.perf_counter()

    for n_batch, batch in enumerate(pf.iter_batches(batch_size=BATCH_ROWS)):
        table = pa.Table.from_batches([batch])

        # Metadata: scalar columns as they are, list columns reduced to a length.
        meta_columns = {name: table[name] for name in scalar_cols}
        for name in list_cols:
            meta_columns[f"{name}__len"] = pc.list_value_length(table[name])
        meta = pa.table(meta_columns)
        if meta_writer is None:
            path = OUT_DIR / f"train_meta_{meta_part:02d}.parquet"
            meta_writer = pq.ParquetWriter(path, meta.schema, compression="zstd")
        meta_writer.write_table(meta)
        meta_rows_in_part += meta.num_rows
        if meta_rows_in_part >= META_ROWS_PER_PART:
            meta_writer.close()
            meta_writer, meta_part, meta_rows_in_part = None, meta_part + 1, 0

        # Sample: whole rows for the molecules whose hash bucket is small enough.
        if key is not None:
            keys = table[key].to_pylist()
            buckets = [bucket_of(k, first14) for k in keys]
        else:
            keys = [None] * table.num_rows
            buckets = [bucket_of(str(rows_done + i), False) for i in range(table.num_rows)]
        parts = [b // part_width if b < SAMPLE_BUCKETS else -1 for b in buckets]
        for part in sorted(set(parts) - {-1}):
            mask = pa.array([p == part for p in parts], type=pa.bool_())
            chosen = table.filter(mask)
            if part not in sample_writers:
                path = OUT_DIR / f"train_sample_{part:02d}.parquet"
                sample_writers[part] = pq.ParquetWriter(path, chosen.schema, compression="zstd")
            sample_writers[part].write_table(chosen)
            sample_rows += chosen.num_rows
        sample_keys.update(k for k, p in zip(keys, parts) if p != -1 and k is not None)

        rows_done += table.num_rows
        if n_batch % 10 == 0 or rows_done == md.num_rows:
            elapsed = time.perf_counter() - start
            print(
                f"\r{rows_done:>10,} / {md.num_rows:,} rows   "
                f"{sample_rows:,} sampled   {elapsed:5.0f} s",
                end="", flush=True,
            )

    if meta_writer is not None:
        meta_writer.close()
    for writer in sample_writers.values():
        writer.close()

    # --- summary ----------------------------------------------------------------
    print("\n")
    total = 0
    for path in sorted(OUT_DIR.iterdir()):
        size = path.stat().st_size
        total += size
        print(f"  {path.name:28s} {size / 1e6:8.1f} MB")
    print(f"  {'total':28s} {total / 1e6:8.1f} MB")
    print(
        f"\nSampled {sample_rows:,} of {rows_done:,} spectra "
        f"({sample_rows / max(rows_done, 1):.1%}) from {len(sample_keys):,} molecules."
    )
    print(f"Written to {OUT_DIR}")


if __name__ == "__main__":
    main()
