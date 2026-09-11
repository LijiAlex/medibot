"""SQL RAG: the analytical branch of /chat. Spec, Component 4.

    get_db()       -> SQLDatabase   read-only handle on mediassist.db whose table_info also
                                    lists the exact values of the categorical columns
    data_span() -> (lo, hi)         earliest / latest date in the DB, told to the LLM, echoed on zero rows
    clean_sql(raw) -> str           one bare SELECT out of whatever text the LLM produced
    write_sql(question) -> str      create_sql_query_chain | clean_sql, one retry on unparseable output
    sql_rag_chain(question) -> str  spec's plain function: write -> clean -> run -> LLM explains
    answer(question, role) -> RagResult   role gate + sql_rag_chain, same shape as hybrid_rag.answer

The SQL itself is written by LangChain's create_sql_query_chain (as in the class
reference). It reads the schema through db.get_table_info(), so the value hints are
attached to the SQLDatabase via custom_table_info and the helper stays untouched.
Its default prompt asks for at most top_k = 5 rows (LIMIT 5), so every SQL answer
carries the note "Results limited to 5 rows." (user decision, 2026-09-09).

Read-only is enforced at the driver (SQLite `mode=ro`), not by trusting the LLM's SQL.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from functools import lru_cache

from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from medibot.config import DB_PATH, SQL_RAG_ROLES
from medibot.retrieval.hybrid_rag import RagResult, get_llm
from medibot.retrieval.phrasing import RECORDS, refusal

logger = logging.getLogger(__name__)

_URI = f"sqlite:///file:{DB_PATH}?mode=ro&uri=true"

# Low-cardinality text columns the LLM filters on. get_table_info() shows 3 sample rows,
# which for claims.status exposes 2 of the 5 real values. SQLite `=` is case-sensitive,
# so the LLM needs the exact spellings: 'approved' but 'HDFC Ergo'.
HINT_COLUMNS: dict[str, tuple[str, ...]] = {
    "claims": ("department", "claim_type", "insurer", "status"),
    "maintenance_tickets": ("category", "campus", "issue_type", "status"),
}


_DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "claims": ("submitted_date", "resolved_date"),
    "maintenance_tickets": ("raised_date", "resolved_date"),
}


@lru_cache(maxsize=1)
def data_span() -> tuple[str, str]:
    """(earliest, latest) date across every date column of both tables, e.g. 2024-01-03 .. 2024-12-28.

    Told to the LLM so it knows a "last month" window (real clock, 2026) is empty before
    running, and echoed in zero-row answers so "0" reads as "outside the data", not as a
    finding. Dates are NOT anchored to the data: 'today' stays today (user decision 2026-09-09)."""
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        lows, highs = [], []
        for table, cols in _DATE_COLUMNS.items():
            for col in cols:
                lo, hi = conn.execute(f'SELECT MIN("{col}"), MAX("{col}") FROM "{table}"').fetchone()
                lows.append(lo); highs.append(hi)
    return min(lows), max(highs)


def _value_hints(table: str) -> str:
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        lines = []
        for col in HINT_COLUMNS[table]:
            rows = conn.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL ORDER BY 1')
            lines.append(f"{table}.{col}: " + ", ".join(f"'{v}'" for (v,) in rows))
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_db() -> SQLDatabase:
    """One read-only connection per process. `uri=true` makes SQLite honour `mode=ro`.

    Built in two passes: a plain handle renders DDL + 3 sample rows per table, then the
    real handle gets that text plus the value hints as custom_table_info, so whatever
    calls get_table_info() (create_sql_query_chain does) sees all of it.
    """
    plain = SQLDatabase.from_uri(_URI, sample_rows_in_table_info=3)
    lo, hi = data_span()
    custom = {
        table: (
            f"{plain.get_table_info([table])}\n"
            f"/* exact values (case-sensitive, use verbatim):\n{_value_hints(table)}\n"
            f"dates are TEXT 'YYYY-MM-DD'. Data spans {lo} to {hi}; anything outside returns no rows. */"
        )
        for table in plain.get_usable_table_names()
    }
    return SQLDatabase.from_uri(_URI, custom_table_info=custom)


# --- Step 2: clean -------------------------------------------------------------------
_FENCE = re.compile(r"```(?:sql)?", re.IGNORECASE)
_FIRST_SELECT = re.compile(r"\b(?:SELECT|WITH)\b.*", re.IGNORECASE | re.DOTALL)


def clean_sql(raw: str) -> str:
    """Raw LLM text -> one bare SELECT statement. Spec step 2 (and the tip on line 225).

    The helper's default prompt asks for a "Question: / SQLQuery:" transcript, and the
    model sometimes wraps the SQL in a markdown fence. Observed shapes:
        "Question: ...\\nSQLQuery: SELECT ...;"
        "Question: ...\\nSQLQuery:  \\n```sql\\nSELECT ...\\nLIMIT 5;\\n```"
        ""   (stop token hit before any SQL)

    Steps: drop everything up to "SQLQuery:", drop fences, keep from the first SELECT or
    WITH to the first ";". Anything that does not yield a SELECT is a ValueError, so the
    caller answers "could not form a query" instead of running arbitrary text. The DB
    handle is read-only anyway; this guard is the second layer, not the only one.
    """
    text = raw.split("SQLQuery:", 1)[-1]
    text = _FENCE.sub("", text)
    match = _FIRST_SELECT.search(text)
    if not match:
        raise ValueError(f"LLM output contains no SELECT statement: {raw[:120]!r}")
    return match.group(0).split(";", 1)[0].strip()


# --- Step 3: question -> SQL ---------------------------------------------------------
@lru_cache(maxsize=1)
def _sql_writer() -> Runnable:
    """Class-reference helper as-is: default SQLite prompt, top_k = 5 (see module note),
    reads the schema through get_db().get_table_info(), so the value hints ride along.
    clean_sql is piped on the end, so what comes out is one bare SELECT or a ValueError."""
    chain = create_sql_query_chain(get_llm(), get_db(), k=SQL_ROW_CAP) | RunnableLambda(clean_sql)
    # Retry on a ValueError from clean_sql, meaning the model produced nothing usable.
    # This existed because gpt-oss-20b returned an empty string on roughly a quarter of
    # calls: the helper stops generation at "\nSQLResult:" and that model sometimes began
    # its visible output at that line. Retrying helped less than it looks, since the
    # failures correlate at temperature 0. Switching to gpt-oss-120b removed the cause
    # (5/5 usable there), and this stays as cheap insurance: it costs nothing when the
    # first attempt succeeds, and a genuinely bad query still propagates.
    return chain.with_retry(retry_if_exception_type=(ValueError,), stop_after_attempt=4, wait_exponential_jitter=False)


def write_sql(question: str) -> str:
    """Spec step 1 + 2: LLM translates the question, output is cleaned to bare SQL."""
    return _sql_writer().invoke({"question": question})


# --- Step 4: run + explain -------------------------------------------------------------
# The helper's prompt asks the model for at most this many rows. It defaulted to 5, which
# silently cut "which insurers are empanelled" down to 5 of the 8 in the table while the
# prose still read as a complete list. Raised 2026-09-11: every grouping in this database
# is well under 50 (8 insurers, 7 departments, 6 categories, 5 campuses), so the cap no
# longer hides anything, and 50 rows is still small enough to hand to the model.
SQL_ROW_CAP = 50
LIMIT_NOTE = f"Results limited to {SQL_ROW_CAP} rows."

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are MediBot, answering an analytics question for hospital staff.\n"
     "You are given the question, the SQL that was run, and the rows it returned.\n"
     "Answer in one or two plain sentences using only those rows. Use the numbers exactly as\n"
     "they appear in the rows, without quotation marks.\n"
     "If the result is empty or every count is 0, say that no matching records were found "
     "and state that the records cover {span_lo} to {span_hi}. Do not show the SQL."),
    ("human", "Question: {question}\nSQL: {sql}\nResult rows: {rows}"),
])


def _run(question: str) -> tuple[str, list[tuple], str]:
    """The three spec steps, returning what each produced: (sql, rows, prose)."""
    sql = write_sql(question)                                   # 1 translate + 2 clean
    rows = get_db()._execute(sql)                               # 3a execute (read-only handle)
    chain = ANSWER_PROMPT | get_llm() | StrOutputParser()       # 3b explain
    lo, hi = data_span()
    prose = chain.invoke({"question": question, "sql": sql, "rows": rows, "span_lo": lo, "span_hi": hi})
    # Only disclose the cap when it could actually have hidden something. A COUNT returns
    # one row, so saying "limited to 5 rows" there is noise that competes with the answer.
    capped = len(rows) >= SQL_ROW_CAP
    return sql, rows, f"{prose.strip()}\n\n{LIMIT_NOTE}" if capped else prose.strip()


def _sources(sql: str) -> list[dict]:
    """Spec line 171 shape, honestly filled: the database is the source document and the
    table is the section within it. The query itself goes in RagResult.sql, because
    section_title is defined as a chunk heading (spec line 77), not a place for SQL."""
    lowered = sql.lower()
    return [
        {"source_document": DB_PATH.name, "section_title": table, "collection": "sql"}
        for table in get_db().get_usable_table_names()
        if table.lower() in lowered
    ]


def sql_rag_chain(question: str) -> str:
    """Spec, Component 4: plain function, natural-language question in, prose answer out.
    Raises ValueError if the LLM produced nothing that cleans to a SELECT (after one retry)."""
    return _run(question)[2]


def answer(question: str, role: str) -> RagResult:
    """The SQL branch of /chat. Role gate first; refusal and error are both ordinary replies."""
    if role not in SQL_RAG_ROLES:
        # Same shape as the document refusal: what is closed, then what is open. The
        # bare "Analytics questions are not available for your role." said only the first
        # half and left the reader to work out what they could still ask.
        return RagResult(answer=refusal(role, RECORDS), retrieval_type="sql_rag", role=role, refusal="role")
    try:
        sql, rows, text = _run(question)
    except ValueError:
        return RagResult(
            answer="I could not form a database query from that question. Please rephrase it.",
            retrieval_type="sql_rag", role=role, refusal="no_query",
        )
    logger.info("sql_rag role=%s rows=%d sql=%s", role, len(rows), sql.replace("\n", " "))
    return RagResult(answer=text, sources=_sources(sql), retrieval_type="sql_rag", role=role, sql=sql)
