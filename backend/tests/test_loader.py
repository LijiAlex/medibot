from medibot.config import COLLECTION_ROLES
from medibot.ingestion.loader import SourceDoc, load_sources


def test_loads_every_pdf_and_md_and_nothing_else():
    docs = load_sources()
    assert len(docs) == 12
    assert all(d.path.suffix in {".pdf", ".md"} for d in docs)
    assert not any("mediassist.db" in str(d.path) for d in docs)


def test_collection_and_roles_come_from_folder():
    by_name = {d.source_document: d for d in load_sources()}

    drug = by_name["drug_formulary.pdf"]
    assert drug.collection == "clinical"
    assert drug.access_roles == COLLECTION_ROLES["clinical"]

    guide = by_name["claim_submission_guide.md"]
    assert guide.collection == "billing"
    assert guide.access_roles == ["billing_executive", "admin"]


def test_every_collection_is_represented():
    seen = {d.collection for d in load_sources()}
    assert seen == set(COLLECTION_ROLES)


def test_source_doc_is_plain_data():
    d = SourceDoc(path=__import__("pathlib").Path("x/general/a.pdf"), collection="general")
    assert d.source_document == "a.pdf"
    assert d.access_roles == COLLECTION_ROLES["general"]
