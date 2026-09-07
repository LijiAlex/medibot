"""Step 1 of ingestion: discover source files and attach collection + access roles.

The folder a file sits in *is* its collection (data/clinical/x.pdf -> "clinical").
Roles come from config.COLLECTION_ROLES, never from the file itself, so a document
can't grant itself wider access. No parsing happens here.

Output of this step feeds three of the five required payload fields:
source_document, collection, access_roles. The other two (section_title,
chunk_type) only exist after chunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from medibot.config import COLLECTION_ROLES, DATA_DIR

# Only these file types are ingested. Anything else in a collection folder is ignored.
SUPPORTED_SUFFIXES = {".pdf", ".md"}


@dataclass(frozen=True)
class SourceDoc:
    """One source file plus the access information it inherits from its folder.

    frozen=True makes the record immutable: once built, nothing downstream can
    change access_roles by accident. That field decides who may retrieve the
    chunks, so it is locked at construction.

    Example:
        SourceDoc(path=Path("data/clinical/drug_formulary.pdf"), collection="clinical")
        -> access_roles == ["doctor", "admin"], source_document == "drug_formulary.pdf"
    """

    path: Path
    collection: str
    access_roles: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Fill access_roles from the role matrix when the caller did not pass them.

        A frozen dataclass forbids normal assignment, so the one-time fill uses
        object.__setattr__. This is the only place roles are ever derived.
        """
        if not self.access_roles:
            object.__setattr__(self, "access_roles", list(COLLECTION_ROLES[self.collection]))

    @property
    def source_document(self) -> str:
        """Filename only, e.g. "drug_formulary.pdf". Used in citations and point ids."""
        return self.path.name


def load_sources(data_dir: Path = DATA_DIR) -> list[SourceDoc]:
    """Return one SourceDoc per supported file under data_dir, in a stable order.

    Iterates the role matrix, not the directory: a folder with no entry in
    COLLECTION_ROLES is skipped entirely (fail closed). That is also why db/
    is never picked up without special-casing it.

    Files are sorted so chunk indices, and therefore point ids, are the same on
    every run.
    """
    docs: list[SourceDoc] = []
    for collection in COLLECTION_ROLES:  # only folders we know roles for
        folder = data_dir / collection
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                docs.append(SourceDoc(path=path, collection=collection))
    return docs
