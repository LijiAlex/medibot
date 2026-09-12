"""SQL RAG, Component 4. Step 1: read-only DB handle whose table_info carries value hints."""

import pytest
import sqlalchemy.exc

from medibot.retrieval.sql_rag import HINT_COLUMNS, get_db


def schema_context() -> str:
    """Exactly what create_sql_query_chain injects as {table_info}."""
    return get_db().get_table_info()


def test_db_exposes_both_tables():
    assert set(get_db().get_usable_table_names()) == {"claims", "maintenance_tickets"}


def test_db_is_read_only():
    # LLM-written SQL executes against this handle. A write must fail at the driver.
    with pytest.raises(sqlalchemy.exc.OperationalError, match="readonly"):
        get_db().run("CREATE TABLE should_not_exist (x INTEGER)")


def test_schema_context_has_ddl_and_sample_rows():
    ctx = schema_context()
    assert "CREATE TABLE claims" in ctx
    assert "CREATE TABLE maintenance_tickets" in ctx
    assert "3 rows from claims table" in ctx


def test_schema_context_lists_every_status_not_just_sampled_ones():
    # Sample rows show 2 of 5 claim statuses. Hints must show all 5, verbatim case.
    ctx = schema_context()
    for status in ("approved", "pending", "rejected", "escalated", "submitted"):
        assert f"'{status}'" in ctx
    for status in ("resolved", "in_progress", "open"):
        assert f"'{status}'" in ctx


def test_schema_context_covers_every_hint_column():
    ctx = schema_context()
    for table, cols in HINT_COLUMNS.items():
        for col in cols:
            assert f"{table}.{col}" in ctx


# --- Step 2: clean_sql -------------------------------------------------------------
from medibot.retrieval.sql_rag import clean_sql  # noqa: E402

# Both shapes below are verbatim create_sql_query_chain outputs seen on 2026-09-09.
BARE = (
    "Question: How many claims did hdfc ergo approve?  \n"
    "SQLQuery: SELECT COUNT(*) AS \"approved_count\" FROM claims "
    "WHERE insurer = 'HDFC Ergo' AND status = 'approved';"
)
FENCED = (
    "Question: How many tickets are still open at the pune campus?  \n"
    "SQLQuery:  \n```sql\nSELECT COUNT(*) AS open_tickets\nFROM maintenance_tickets\n"
    "WHERE \"campus\" = 'MediAssist Pune Speciality'\n  AND \"status\" = 'open'\nLIMIT 5;\n```"
)


def test_clean_sql_bare_sqlquery_line():
    assert clean_sql(BARE) == (
        "SELECT COUNT(*) AS \"approved_count\" FROM claims "
        "WHERE insurer = 'HDFC Ergo' AND status = 'approved'"
    )


def test_clean_sql_fenced_block_keeps_newlines_drops_fence_and_semicolon():
    out = clean_sql(FENCED)
    assert out.startswith("SELECT COUNT(*) AS open_tickets\nFROM maintenance_tickets")
    assert out.endswith("LIMIT 5")
    assert "```" not in out and ";" not in out


def test_clean_sql_prose_prefix_and_already_clean():
    assert clean_sql("Here is the query:\nSELECT 1") == "SELECT 1"
    assert clean_sql("SELECT 1") == "SELECT 1"
    assert clean_sql("WITH t AS (SELECT 1) SELECT * FROM t") == "WITH t AS (SELECT 1) SELECT * FROM t"


def test_clean_sql_keeps_first_statement_only():
    assert clean_sql("SELECT 1; DROP TABLE claims;") == "SELECT 1"


def test_clean_sql_rejects_non_select():
    with pytest.raises(ValueError, match="SELECT"):
        clean_sql("DROP TABLE claims")


def test_clean_sql_rejects_empty_output():
    # Seen for "escalated last month": the helper returned "" (stop token hit early).
    with pytest.raises(ValueError, match="SELECT"):
        clean_sql("")


# --- Step 3: write_sql -------------------------------------------------------------
import os  # noqa: E402

from medibot.retrieval.sql_rag import write_sql  # noqa: E402

needs_llm = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")


@needs_llm
def test_write_sql_uses_hinted_value_and_explicit_month_and_runs():
    sql = write_sql("How many claims were escalated in March 2024?")
    assert sql.lstrip().upper().startswith("SELECT")
    assert "'escalated'" in sql          # exact hinted value, not 'Escalated'
    assert "2024-03" in sql              # explicit month, not date('now')
    assert ";" not in sql and "```" not in sql
    rows = get_db().run(sql)
    assert rows                          # executes, returns something


@needs_llm
def test_write_sql_maps_lowercase_insurer_to_hinted_spelling():
    sql = write_sql("How many claims did hdfc ergo approve?")
    assert "'HDFC Ergo'" in sql


# --- Step 4: sql_rag_chain + answer -------------------------------------------------
import re  # noqa: E402
import sqlite3  # noqa: E402

from medibot.config import DB_PATH, SQL_RAG_ROLES  # noqa: E402
from medibot.retrieval import sql_rag  # noqa: E402
from medibot.retrieval.hybrid_rag import RagResult  # noqa: E402
from medibot.retrieval.sql_rag import LIMIT_NOTE, answer, sql_rag_chain  # noqa: E402


def truth(sql: str):
    """Ground truth straight from SQLite, independent of any LLM."""
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as c:
        return c.execute(sql).fetchall()


def has_number(text: str, n: int) -> bool:
    """Whole-number match. The trailing guard rejects a following digit, and a decimal
    point only when a digit follows it, so "with 5." counts and "5.2" does not. The
    earlier version rejected any number before a full stop, which failed a correct
    answer ending "Cardiology has the most rejected claims with 5." The model also
    writes U+202F between words, so normalise before matching."""
    flat = re.sub(r"[\u202f\u00a0]", " ", text)
    return re.search(rf"(?<![\d.]){n}(?!\d|\.\d)", flat) is not None


# Graded questions: months are explicit (data is 2024, clock is not), values are hinted.
@needs_llm
def test_graded_count_with_explicit_month():
    (n,), = truth("select count(*) from claims where status='escalated' and submitted_date like '2024-03%'")
    out = sql_rag_chain("How many claims were escalated in March 2024?")
    assert has_number(out, n), out


@needs_llm
def test_graded_group_by_top_category():
    (cat, _), *_ = truth("select category, count(*) n from maintenance_tickets where status='open' group by category order by n desc")
    out = sql_rag_chain("Which equipment category has the most open maintenance tickets?")
    assert cat in out.lower(), out


@needs_llm
def test_graded_hinted_value_lowercase_in_question():
    (n,), = truth("select count(*) from claims where insurer='HDFC Ergo' and status='approved'")
    out = sql_rag_chain("How many claims did hdfc ergo approve?")
    assert has_number(out, n), out


@needs_llm
def test_graded_department_with_most_rejections():
    (dep, n), *_ = truth("select department, count(*) n from claims where status='rejected' group by department order by n desc")
    out = sql_rag_chain("Which department has the most rejected claims, and how many?")
    assert dep in out.lower() and has_number(out, n), out


@needs_llm
def test_every_group_survives_the_row_cap():
    """The cap was the library's default of 5, which cut 8 insurers down to 5 while the
    prose still read as a complete list. Every grouping in this database is well under
    the raised cap: 8 insurers, 7 departments, 6 categories, 5 campuses."""
    out = sql_rag_chain("How many approved claims does each insurer have?")
    # The model writes U+202F between words, so compare on normalised text.
    flat = re.sub(r"[\u202f\u00a0]", " ", out)
    for insurer in ("ICICI Lombard", "United India", "Niva Bupa", "Star Health", "HDFC Ergo"):
        assert insurer in flat, f"{insurer} missing from: {flat}"
    assert LIMIT_NOTE not in flat, "nothing was cut, so the cap should not be mentioned"


@needs_llm
def test_a_single_row_answer_does_not_mention_the_cap():
    # A COUNT returns one row. The cap cut nothing, so saying so would be noise.
    out = sql_rag_chain("How many claims were escalated in March 2024?")
    assert LIMIT_NOTE not in out, out


@needs_llm
def test_answer_shape_for_permitted_role():
    res = answer("How many claims did hdfc ergo approve?", "billing_executive")
    assert isinstance(res, RagResult)
    assert res.retrieval_type == "sql_rag" and res.role == "billing_executive"
    # Spec line 171 shape on every branch; the query itself rides in res.sql (line 168 is a minimum).
    assert len(res.sources) == 1 and set(res.sources[0]) == {"source_document", "section_title", "collection"}
    assert res.sources[0] == {"source_document": "mediassist.db", "section_title": "claims", "collection": "sql"}
    assert res.sql.upper().startswith("SELECT")


def test_answer_refuses_roles_without_analytics_and_never_writes_sql(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("write_sql must not be called for a refused role")
    monkeypatch.setattr(sql_rag, "write_sql", boom)
    for role in ("nurse", "doctor", "technician"):
        assert role not in SQL_RAG_ROLES
        res = answer("How many claims were escalated in March 2024?", role)
        assert res.retrieval_type == "sql_rag" and res.sources == [] and res.role == role
        assert res.sql is None
        assert res.answer.startswith(f"As {'an' if role[0] in 'aeiou' else 'a'} {role.replace('_', ' ')}, you don't have access")


def test_answer_turns_unparseable_sql_into_a_reply_not_a_crash(monkeypatch):
    monkeypatch.setattr(sql_rag, "write_sql", lambda q: (_ for _ in ()).throw(ValueError("no SELECT")))
    res = answer("gibberish", "admin")
    assert res.retrieval_type == "sql_rag" and res.sources == []
    assert "could not" in res.answer.lower()


# --- Data span: relative dates stay real (no anchoring to 2024) ---------------------
from medibot.retrieval.sql_rag import data_span  # noqa: E402


def test_data_span_covers_both_tables():
    lo, hi = data_span()
    (c_lo, c_hi), = truth("select min(submitted_date), max(coalesce(resolved_date, submitted_date)) from claims")
    (t_lo, t_hi), = truth("select min(raised_date), max(coalesce(resolved_date, raised_date)) from maintenance_tickets")
    assert lo == min(c_lo, t_lo) and hi == max(c_hi, t_hi)
    assert f"Data spans {lo} to {hi}" in get_db().get_table_info()


@needs_llm
def test_relative_date_question_is_answered_against_the_real_clock_and_explains_the_span():
    # "last month" is a 2026 window; the data is 2024. Zero rows is the right answer,
    # and the reply must say why, so the user does not read "0" as a finding.
    res = answer("How many billing claims were escalated last month?", "admin")
    assert res.sources[0]["collection"] == "sql"
    assert "2024" in res.answer, res.answer                    # span disclosed
    assert "date('now'" in res.sql or "2026" in res.sql


# --- Money ---------------------------------------------------------------------------
from medibot.retrieval.sql_rag import CURRENCY, _round_money  # noqa: E402


def test_money_is_rounded_before_it_reaches_the_model():
    """SQLite returns the full float, so an average arrived as 55515.90909090909 and the
    model repeated it verbatim, as the prompt told it to. Rounding here means the figure
    is right even if the model ignores the instruction."""
    assert _round_money([{"avg": 55515.90909090909, "n": 12}]) == [{"avg": 55515.91, "n": 12}]
    assert _round_money([(55515.90909090909, "x")]) == [(55515.91, "x")]
    assert _round_money([{"c": 3}]) == [{"c": 3}], "integers are left alone"


@needs_llm
def test_an_amount_is_written_as_rupees_with_two_decimals():
    """MediAssist bills in rupees: the documents use the sign 22 times and no other
    currency appears anywhere in the corpus."""
    out = sql_rag_chain("What is the average approved amount on a claim?")
    flat = re.sub(r"[  ]", " ", out)
    assert CURRENCY in flat, f"no currency sign in: {flat}"
    assert not re.search(r"\d\.\d{3,}", flat), f"unrounded figure in: {flat}"
