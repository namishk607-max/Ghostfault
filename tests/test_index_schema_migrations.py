import sqlite3
import hashlib
from pathlib import Path

from src.indexer import SearchIndex


def _create_versioned_database(path: Path, version: int, with_dataset_columns: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        dataset_column = ", dataset_id TEXT DEFAULT 'DATASET-DEFAULT'" if with_dataset_columns else ""
        chunk_dataset_column = ", dataset_id TEXT DEFAULT 'DATASET-DEFAULT'" if with_dataset_columns else ""
        connection.execute(
            "CREATE TABLE indexed_documents ("
            "document_id TEXT PRIMARY KEY, source_filename TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, chunk_count INTEGER NOT NULL, "
            f"indexed_at TEXT NOT NULL{dataset_column})"
        )
        connection.execute(
            "CREATE TABLE evidence_chunks ("
            "chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, "
            "chunk_index INTEGER NOT NULL, text TEXT NOT NULL, "
            "source_filename TEXT NOT NULL, file_type TEXT NOT NULL, "
            "page_number INTEGER, start_offset INTEGER, end_offset INTEGER, "
            "created_at TEXT NOT NULL, content_hash TEXT NOT NULL"
            f"{chunk_dataset_column}, "
            "FOREIGN KEY(document_id) REFERENCES indexed_documents(document_id) "
            "ON DELETE CASCADE)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "chunk_id UNINDEXED, document_id UNINDEXED, source_filename, text)"
        )
        if with_dataset_columns:
            connection.execute(
                "INSERT INTO indexed_documents VALUES (?, ?, ?, ?, ?, ?)",
                ("DOC-VALID", "valid.txt", "hash-valid", 1, "2026-01-01", "DATASET-CUSTOM"),
            )
            connection.execute(
                "INSERT INTO indexed_documents VALUES (?, ?, ?, ?, ?, ?)",
                ("DOC-NULL", "null.txt", "hash-null", 1, "2026-01-01", None),
            )
            connection.execute(
                "INSERT INTO evidence_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("CHUNK-VALID", "DOC-VALID", 0, "valid evidence", "valid.txt", "txt",
                 None, 0, 14, "2026-01-01", "hash-valid", "DATASET-CUSTOM"),
            )
            connection.execute(
                "INSERT INTO evidence_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("CHUNK-NULL", "DOC-NULL", 0, "unassigned evidence", "null.txt", "txt",
                 None, 0, 20, "2026-01-01", "hash-null", None),
            )
            connection.executemany(
                "INSERT INTO chunks_fts VALUES (?, ?, ?, ?)",
                [
                    ("CHUNK-VALID", "DOC-VALID", "valid.txt", "valid evidence"),
                    ("CHUNK-NULL", "DOC-NULL", "null.txt", "unassigned evidence"),
                ],
            )
        else:
            connection.execute(
                "INSERT INTO indexed_documents VALUES (?, ?, ?, ?, ?)",
                ("DOC-LEGACY", "legacy.txt", "legacy-hash", 1, "2026-01-01"),
            )
            connection.execute(
                "INSERT INTO evidence_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("CHUNK-LEGACY", "DOC-LEGACY", 0, "legacy evidence", "legacy.txt", "txt",
                 None, 0, 15, "2026-01-01", "legacy-hash"),
            )
            connection.execute(
                "INSERT INTO chunks_fts VALUES (?, ?, ?, ?)",
                ("CHUNK-LEGACY", "DOC-LEGACY", "legacy.txt", "legacy evidence"),
            )
        connection.execute(f"PRAGMA user_version = {version}")


def _dataset_defaults(connection: sqlite3.Connection) -> dict[str, object]:
    result = {}
    for table in ("indexed_documents", "evidence_chunks"):
        column = next(
            row for row in connection.execute(f"PRAGMA table_info({table})")
            if row[1] == "dataset_id"
        )
        result[table] = column[4]
    return result


def test_fresh_index_database_has_consistent_latest_version_and_no_owner_defaults(tmp_path):
    database = tmp_path / "fresh.db"

    SearchIndex(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,), (3,)]
        assert _dataset_defaults(connection) == {
            "indexed_documents": None,
            "evidence_chunks": None,
        }


def test_version_two_migration_removes_defaults_without_changing_owners_or_fts(tmp_path):
    database = tmp_path / "version-two.db"
    _create_versioned_database(database, version=2, with_dataset_columns=True)

    SearchIndex(database)

    migration_backups = list(tmp_path.glob("version-two.db.schema-v3-*.bak"))
    assert len(migration_backups) == 1
    with sqlite3.connect(migration_backups[0]) as backup_connection:
        assert backup_connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert _dataset_defaults(connection) == {
            "indexed_documents": None,
            "evidence_chunks": None,
        }
        assert connection.execute(
            "SELECT document_id, dataset_id FROM indexed_documents ORDER BY document_id"
        ).fetchall() == [("DOC-NULL", None), ("DOC-VALID", "DATASET-CUSTOM")]
        assert connection.execute(
            "SELECT chunk_id, dataset_id FROM evidence_chunks ORDER BY chunk_id"
        ).fetchall() == [("CHUNK-NULL", None), ("CHUNK-VALID", "DATASET-CUSTOM")]
        assert connection.execute(
            "SELECT chunk_id, document_id, text FROM chunks_fts ORDER BY chunk_id"
        ).fetchall() == [
            ("CHUNK-NULL", "DOC-NULL", "unassigned evidence"),
            ("CHUNK-VALID", "DOC-VALID", "valid evidence"),
        ]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    migrated_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    SearchIndex(database)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == migrated_hash


def test_failed_version_three_migration_rolls_back_and_can_be_retried(tmp_path, monkeypatch):
    database = tmp_path / "version-two-failure.db"
    _create_versioned_database(database, version=2, with_dataset_columns=True)
    original_migration = SearchIndex._remove_dataset_column_defaults

    def fail_migration(connection):
        raise RuntimeError("simulated v3 migration failure")

    monkeypatch.setattr(SearchIndex, "_remove_dataset_column_defaults", staticmethod(fail_migration))
    try:
        SearchIndex(database)
    except RuntimeError as exc:
        assert "simulated v3 migration failure" in str(exc)
    else:
        raise AssertionError("Injected schema migration failure was not raised.")

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert "schema_migrations" not in {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert _dataset_defaults(connection) == {
            "indexed_documents": "'DATASET-DEFAULT'",
            "evidence_chunks": "'DATASET-DEFAULT'",
        }
        assert connection.execute(
            "SELECT document_id, dataset_id FROM indexed_documents ORDER BY document_id"
        ).fetchall() == [("DOC-NULL", None), ("DOC-VALID", "DATASET-CUSTOM")]

    monkeypatch.setattr(SearchIndex, "_remove_dataset_column_defaults", staticmethod(original_migration))
    SearchIndex(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,), (3,)]
        assert _dataset_defaults(connection) == {
            "indexed_documents": None,
            "evidence_chunks": None,
        }
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_version_one_migration_adds_null_ownership_without_inference(tmp_path):
    database = tmp_path / "version-one.db"
    _create_versioned_database(database, version=1, with_dataset_columns=False)

    SearchIndex(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute(
            "SELECT dataset_id FROM indexed_documents WHERE document_id = 'DOC-LEGACY'"
        ).fetchone() == (None,)
        assert connection.execute(
            "SELECT dataset_id FROM evidence_chunks WHERE chunk_id = 'CHUNK-LEGACY'"
        ).fetchone() == (None,)
        assert _dataset_defaults(connection) == {
            "indexed_documents": None,
            "evidence_chunks": None,
        }
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
