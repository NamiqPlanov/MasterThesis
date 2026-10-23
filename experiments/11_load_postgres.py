"""
Load both datasets into PostgreSQL so they can be inspected with SQL.

Creates a schema `fraud` with two tables, builds the column list from each
CSV's own header, and bulk-loads with COPY (far faster than row inserts).

CREDENTIALS are read from the environment and never stored here:

    $env:PGPASSWORD = "your-password"      # set it in YOUR terminal
    .venv\\Scripts\\python.exe experiments\\11_load_postgres.py

Optional overrides: PGHOST, PGPORT, PGUSER, PGDATABASE.
Add --rows N to load only the first N rows of the big IEEE-CIS table.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
ULB = ROOT / "data" / "ulb" / "creditcard.csv"
IEEE = ROOT / "data" / "ieee" / "train_transaction.csv"
SCHEMA = "fraud"

# Columns that must stay exact; everything else is loaded as double precision
TEXT_COLS = {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
             "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"}


def quote(name: str) -> str:
    """Postgres folds unquoted names to lower case; keep V14 distinct from v14."""
    return '"' + name.replace('"', '""') + '"'


def header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def ddl(table: str, cols: list[str]) -> str:
    defs = []
    for c in cols:
        kind = "text" if c in TEXT_COLS else "double precision"
        defs.append(f"  {quote(c)} {kind}")
    return (f"DROP TABLE IF EXISTS {SCHEMA}.{table};\n"
            f"CREATE TABLE {SCHEMA}.{table} (\n" + ",\n".join(defs) + "\n);")


def load(conn, table: str, path: Path, cols: list[str], limit: int | None):
    target = f"{SCHEMA}.{table} ({', '.join(quote(c) for c in cols)})"
    sql = f"COPY {target} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
    t = time.time()
    with conn.cursor() as cur, cur.copy(sql) as cp, path.open("rb") as fh:
        if limit is None:
            while chunk := fh.read(1 << 20):
                cp.write(chunk)
        else:
            # stream only the first `limit` data rows, header included
            written = 0
            for n, line in enumerate(fh):
                cp.write(line)
                if n:                      # don't count the header
                    written += 1
                    if written >= limit:
                        break
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {SCHEMA}.{table}")
        n = cur.fetchone()[0]
    print(f"    loaded {n:,} rows in {time.time() - t:.1f}s")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None,
                    help="limit rows loaded from IEEE-CIS")
    args = ap.parse_args()

    if not os.environ.get("PGPASSWORD"):
        print("PGPASSWORD is not set. In your terminal:\n"
              '    $env:PGPASSWORD = "your-postgres-password"\n'
              "then run this again. The password is never written to a file.")
        return 2

    dsn = {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": os.environ.get("PGPORT", "5432"),
        "user": os.environ.get("PGUSER", "postgres"),
        "dbname": os.environ.get("PGDATABASE", "postgres"),
    }
    print("=" * 70)
    print("  LOADING THE DATASETS INTO POSTGRESQL")
    print("=" * 70)
    print(f"  {dsn['user']}@{dsn['host']}:{dsn['port']}/{dsn['dbname']}")

    with psycopg.connect(**dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        conn.commit()

        for table, path, limit in (("ulb", ULB, None),
                                   ("ieee", IEEE, args.rows)):
            if not path.exists():
                print(f"\n  {table}: {path.name} missing - run 00_get_data.py")
                continue
            cols = header(path)
            print(f"\n  {table}  ({len(cols)} columns from {path.name})")
            with conn.cursor() as cur:
                cur.execute(ddl(table, cols))
            conn.commit()
            load(conn, table, path, cols, limit)

        # indexes on the columns you will actually filter by
        print("\n  building indexes ...")
        with conn.cursor() as cur:
            for stmt in (
                f'CREATE INDEX IF NOT EXISTS ulb_class ON {SCHEMA}.ulb ("Class")',
                f'CREATE INDEX IF NOT EXISTS ulb_time ON {SCHEMA}.ulb ("Time")',
                f'CREATE INDEX IF NOT EXISTS ieee_fraud ON {SCHEMA}.ieee ("isFraud")',
                f'CREATE INDEX IF NOT EXISTS ieee_dt ON {SCHEMA}.ieee ("TransactionDT")',
            ):
                try:
                    cur.execute(stmt)
                except Exception as exc:
                    print(f"    skipped: {str(exc).splitlines()[0][:70]}")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT 'ulb' AS t, count(*) AS rows, sum("Class") AS fraud
                  FROM {SCHEMA}.ulb
                UNION ALL
                SELECT 'ieee', count(*), sum("isFraud") FROM {SCHEMA}.ieee""")
            print(f"\n  {'table':<8}{'rows':>12}{'fraud':>10}{'rate':>9}")
            for t, rows, fraud in cur.fetchall():
                print(f"  {t:<8}{rows:>12,}{int(fraud or 0):>10,}"
                      f"{(fraud or 0) / max(rows, 1) * 100:>8.3f}%")

    print(f"\n  done. Connect with:  psql -U {dsn['user']} -d {dsn['dbname']}")
    print(f"  then:  SET search_path TO {SCHEMA};")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
