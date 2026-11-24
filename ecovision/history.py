"""SQLite inspection log. No original images or videos are retained."""
import csv, io, json, sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

class History:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('''CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL,
                total INTEGER NOT NULL, latency_ms REAL NOT NULL, counts TEXT NOT NULL,
                details TEXT NOT NULL)''')
    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()
    def add(self, source, result):
        with self.connect() as c:
            cursor = c.execute('INSERT INTO inspections(created_at,source,total,latency_ms,counts,details) VALUES(?,?,?,?,?,?)',
                (datetime.now(timezone.utc).isoformat(), source, result['total'], result['latency_ms'],
                 json.dumps(result['counts']),json.dumps(result)))
            return cursor.lastrowid
    def recent(self, limit=100):
        with self.connect() as c:
            rows = c.execute('SELECT id,created_at,source,total,latency_ms,counts FROM inspections ORDER BY id DESC LIMIT ?', (max(1,min(1000,int(limit))),)).fetchall()
        return [dict(r) | {'counts':json.loads(r['counts'])} for r in rows]
    def summary(self):
        counts = {}
        with self.connect() as c:
            info = dict(c.execute('SELECT COUNT(*) inspections,COALESCE(SUM(total),0) detections,COALESCE(AVG(latency_ms),0) avg_latency_ms FROM inspections').fetchone())
            for row in c.execute('SELECT counts FROM inspections'):
                for label,n in json.loads(row['counts']).items():
                    counts[label]=counts.get(label,0)+n
        return info | {'counts':dict(sorted(counts.items()))}
    def export(self):
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(['id','created_at','source','total','latency_ms','counts_json'])
        with self.connect() as c:
            for row in c.execute('SELECT id,created_at,source,total,latency_ms,counts FROM inspections ORDER BY id'):
                writer.writerow(tuple(row))
        return out.getvalue()
