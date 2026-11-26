"""Independent Task-1 oracle: do not replace it with the optimized converter."""
import datetime as dt
from decimal import Decimal
import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flowyforge.data_plugins.collide_v2 import streaming as s
from flowyforge.data_plugins.collide_v2.provenance import (
    ORIGIN_COLUMNS, event_id, logical_origin, pop_origin, with_origin, sha256_file,
)


def reference_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, (dt.date, dt.datetime, dt.time, Decimal)):
        return {type(value).__name__: str(value)}
    raise TypeError(type(value).__name__)


def reference_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=reference_default).encode()


def reference_field(field):
    dtype = field.type
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        signature = ["large_list" if pa.types.is_large_list(dtype) else "list", getattr(dtype, "list_size", None), reference_field(dtype.value_field)[1:]]
    elif pa.types.is_struct(dtype):
        signature = ["struct", sorted((reference_field(child) for child in dtype), key=lambda x: x[0])]
    else:
        signature = str(dtype)
    return [field.name, signature, field.nullable, sorted((k.hex(), v.hex()) for k, v in (field.metadata or {}).items())]


def reference_schema(schema):
    return hashlib.sha256(reference_json({"fields": sorted((reference_field(f) for f in schema if f.name not in ORIGIN_COLUMNS), key=lambda x: x[0]),
        "metadata": sorted((k.hex(), v.hex()) for k, v in (schema.metadata or {}).items())})).hexdigest()


def reference_records(table, name, digest="a" * 64, row_start=0):
    signature = reference_schema(table.schema)
    records = []
    for index, row in enumerate(table.to_pylist()):
        origin = pop_origin(row) or {"source": "local", "revision": None, "process": name,
            "file": "sha256:" + digest, "row": row_start + index, "sha256": digest}
        records.append({"id": event_id(origin), "origin": logical_origin(origin),
                        "content_sha256": hashlib.sha256(reference_json([signature, row])).hexdigest()})
    return records


@pytest.mark.parametrize("dtype", [pa.float16(), pa.float32(), pa.float64(), pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()])
def test_numeric_buffers_offsets_nulls_extremes_and_nested_lists(dtype):
    rng = np.random.default_rng(2718)
    if pa.types.is_floating(dtype):
        numeric = np.dtype(dtype.to_pandas_dtype())
        values = [None, -0., 0., float('nan'), float('inf'), -float('inf'), np.finfo(numeric).max,
                  np.nextafter(numeric.type(0), numeric.type(1)), *rng.normal(size=200)]
        if dtype == pa.float16():
            values = [None if v is None else np.float16(v) for v in values]
    else:
        limits = np.iinfo(dtype.to_pandas_dtype())
        values = [None, int(limits.min), int(limits.max), 0, 1, *rng.integers(0, 50, 200).tolist()]
    array = pa.array(values, type=dtype)
    for sliced in (array, array.slice(1), array.slice(2, 9), array.slice(0, 0)):
        assert reference_json(s._arrow_values(sliced)) == reference_json(sliced.to_pylist())
    lists = pa.ListArray.from_arrays(pa.array([0, 0, 3, 5, len(values)]), array,
                                     mask=pa.array([False, True, False, False]))
    for nested in (lists, pa.LargeListArray.from_arrays(pa.array([0, 2, 4], type=pa.int64()), lists)):
        for sliced in (nested, nested.slice(1)):
            assert reference_json(s._arrow_values(sliced)) == reference_json(sliced.to_pylist())


def mixed_table():
    cols = {
        "float": pa.array([[-0., float('nan'), float('inf'), -float('inf')], [], None, [1e-250], [1e250]], type=pa.large_list(pa.float64())),
        "fixed": pa.array([[1, 2], None, [2**63-1, None], [-2**63, 0], [5, 6]], type=pa.list_(pa.int64(), 2)),
        "struct": pa.array([{"a": [1., None], "b": None}, None, {"a": [], "b": "é"}, {"a": None, "b": "\x00"}, {"a": [2.], "b": "x"}],
                           type=pa.struct([pa.field("a", pa.list_(pa.float32())), pa.field("b", pa.string())])),
        "bytes": pa.array([b"\x00\xff", None, b"", b"abc", b"x"]),
        "decimal": pa.array([Decimal("1.00"), None, Decimal("-0.01"), Decimal("999999999999999.99"), Decimal("0.00")], type=pa.decimal128(20, 2)),
        "timestamp": pa.array([None, 1, 123456789, 0, -123456789], type=pa.timestamp("ns", tz="UTC")),
        "date": pa.array([None, 0, -1, 2, 300], type=pa.date32()),
        "time": pa.array([None, 0, 1, 123, 124], type=pa.time64("us")),
        "map": pa.array([[('z', 2), ('a', 1), ('a', 3)], [], None, [('é', None)], []], type=pa.map_(pa.string(), pa.int64())),
        "dictionary": pa.array(['a', None, 'b', 'a', 'é']).dictionary_encode(),
        "bool": pa.array([True, False, None, False, True]),
        "null": pa.nulls(5),
    }
    schema = pa.schema([pa.field(name, array.type, metadata={b'opaque': b'\xff\x00'})
                        for name, array in cols.items()], metadata={b"opaque": b"\x00\xff", b"units": b"raw"})
    return pa.table(cols, schema=schema)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 64, 2048])
def test_canonical_json_and_fingerprints_are_exact_for_mixed_types_and_chunks(batch_size):
    table = mixed_table()
    for column in table.columns:
        assert reference_json(s._arrow_values(column)) == reference_json(column.to_pylist())
    table = with_origin(table, {"source": "local", "revision": None, "process": "a", "file": "file:///raw",
                               "sha256": "a" * 64}, list(range(len(table))))
    table = pa.concat_tables([table.slice(0, 2), table.slice(2)])
    for candidate in (table, table.slice(1, 3), table.slice(0, 0)):
        assert reference_json(list(s._arrow_rows(candidate))) == reference_json(candidate.to_pylist())
        actual = [record for batch in candidate.to_batches(max_chunksize=batch_size)
                  for record in s._event_records(pa.Table.from_batches([batch]), 'a', policy='strict')]
        assert actual == reference_records(candidate, 'a')


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_original_strict_manifest_still_matches_exactly_and_fast_features_agree(tmp_path, workers):
    from flowyforge.data_plugins.collide_v2.synthetic import make_tiny_parquet_sample
    make_tiny_parquet_sample(tmp_path / 'data', events_per_class=12, files_per_class=2)
    source = {'root': str(tmp_path / 'data')}
    original = s._event_records
    # Build a saved v1 fixture using the literal reference conversion, not optimized code.
    def oracle(table, name, **kw):
        return reference_records(table, name, kw.get('digest'), kw.get('row_start', 0))
    s._event_records = oracle
    try:
        expected = s.create_partition_manifest(source, read_batch_size=3)
    finally:
        s._event_records = original
    path = tmp_path / 'reference-v1.json'
    s.save_partition_manifest(expected, path)
    actual = s.create_partition_manifest(source, read_batch_size=7, num_workers=workers)
    assert actual == s.load_partition_manifest(path)
    fast = s.create_partition_manifest(source, check_integrity_SHA256='fast')
    limits = {'PUPPIPart': 16, 'JetPuppiAK4': 4, 'Electron': 4, 'MuonTight': 4, 'PhotonTight': 4, 'PUPPIMET': 1}
    for view in ('FullReco', 'L1T'):
        strict_binding = s.bind_partition_manifest(actual, source, view=view, collection_limits=limits)
        fast_binding = s.bind_partition_manifest(fast, source, view=view, collection_limits=limits)
        for split in s.SPLITS:
            def rows(binding):
                return {e.metadata['event_id']: e for e in s.StreamingDataset(binding, split, read_batch_size=3)}
            a, b = rows(strict_binding), rows(fast_binding)
            assert a.keys() == b.keys()
            for key in a:
                assert a[key].targets.equal(b[key].targets)
                assert all(group.features.equal(b[key].objects[name].features) for name, group in a[key].objects.items())


def test_schema_computed_once_per_file_not_per_batch(tmp_path, monkeypatch):
    from flowyforge.data_plugins.collide_v2.synthetic import make_tiny_parquet_sample
    make_tiny_parquet_sample(tmp_path / 'data', events_per_class=5, files_per_class=1)
    original, calls = s._schema_digest, []
    def counted(schema):
        calls.append(1)
        return original(schema)
    monkeypatch.setattr(s, '_schema_digest', counted)
    s.create_partition_manifest({'root': str(tmp_path / 'data')}, read_batch_size=1)
    assert len(calls) == 2


def test_python_expansion_slices_skewed_batches_without_changing_values():
    table = pa.table({'x': [[1.] * 10000, [], None, [2.], [3.] * 10]})
    assert reference_json(list(s._arrow_rows(table, max_bytes=128))) == reference_json(table.to_pylist())


def test_tiny_batches_keep_arrow_conversion_without_numeric_setup(monkeypatch):
    def unnecessary(*args):
        raise AssertionError('Small batches should use the cheaper original conversion')
    monkeypatch.setattr(s, '_arrow_values', unnecessary)
    table = mixed_table()
    assert reference_json(list(s._arrow_rows(table))) == reference_json(table.to_pylist())


def test_large_mixed_table_fingerprints_match_reference_with_native_conversion():
    table = pa.concat_tables([mixed_table()] * 256).combine_chunks()
    assert table.nbytes > 64 * 1024
    actual = s._event_records(table, 'a', policy='strict', digest='a' * 64)
    assert actual == reference_records(table, 'a')


def test_null_parents_do_not_convert_unrepresentable_hidden_children():
    child = pa.array([2**63 - 1, 0], type=pa.timestamp('s'))
    mask = pa.array([True, False])
    arrays = [pa.ListArray.from_arrays([0, 1, 2], child, mask=mask),
              pa.FixedSizeListArray.from_arrays(child, 1, mask=mask),
              pa.StructArray.from_arrays([child], ['time'], mask=mask)]
    for array in arrays:
        assert reference_json(s._arrow_values(array)) == reference_json(array.to_pylist())


def test_strict_audit_reuses_only_operation_local_proof_and_rechecks_content(tmp_path, monkeypatch):
    from flowyforge.data_plugins.collide_v2 import sample_preparation as prep, provenance as prov
    from flowyforge.data_plugins.collide_v2.synthetic import make_tiny_parquet_sample
    files = make_tiny_parquet_sample(tmp_path / 'data', events_per_class=5, files_per_class=1)
    path = files[0]
    readers = {view: s._reader_config(view=view, collection_limits={
        'PUPPIPart': 16, 'JetPuppiAK4': 4, 'Electron': 4, 'MuonTight': 4, 'PhotonTight': 4, 'PUPPIMET': 1}) for view in ('FullReco', 'L1T')}
    calls, original_hash = [], prov.sha256_file
    def counted(file):
        calls.append(str(file))
        return original_hash(file)
    monkeypatch.setattr(prov, 'sha256_file', counted)
    proof = prov.verify_file(path)
    prep._validate_file((path, 'background', readers, 2, proof[1], proof))
    assert len(calls) == 2  # Initial SHA + mandatory final SHA, shared by both views.
    prep._validate_file((path, 'background', readers, 2, proof[1]))
    assert len(calls) == 4  # A new audit does not trust an earlier operation.
    content, stat = path.read_bytes(), path.stat()
    original_inspect = prep.inspect_dataset
    audited = []
    def mutate_after_read(*args, **kwargs):
        result = original_inspect(*args, **kwargs)
        audited.append(1)
        if len(audited) == 2:
            payload = bytearray(content); payload[20] ^= 1
            path.write_bytes(payload)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return result
    monkeypatch.setattr(prep, 'inspect_dataset', mutate_after_read)
    # Simulate an unhelpful/stale metadata service: final SHA must STILL catch corruption.
    monkeypatch.setattr(prov, '_file_identity', lambda path: proof[0])
    with pytest.raises(ValueError, match='content changed'):
        prep._validate_file((path, 'background', readers, 2, proof[1], proof))
    assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns


def test_verified_proof_rejects_inode_replacement_and_same_mtime_write(tmp_path):
    from flowyforge.data_plugins.collide_v2.provenance import verify_file, check_verified_file
    path = tmp_path / 'payload'; path.write_bytes(b'original')
    proof = verify_file(path); stat = path.stat()
    replacement = tmp_path / 'replacement'; replacement.write_bytes(b'modified')
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns)); replacement.replace(path)
    with pytest.raises(ValueError, match='identity/metadata'):
        check_verified_file(path, proof)
    with pytest.raises(ValueError, match='content changed'):
        verify_file(path, proof[1])
