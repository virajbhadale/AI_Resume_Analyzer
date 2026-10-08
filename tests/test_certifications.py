from src.document_models import DocumentBlock
from src.extraction.certifications import CertificationExtractor


def make_block(text: str) -> DocumentBlock:
    return DocumentBlock(
        id="test_cert_1",
        text=text,
        section="certifications",
        is_heading=False,
    )


def extract(text: str):
    extractor = CertificationExtractor()
    return extractor.extract([make_block(text)])


def test_single_certification_with_issuer_and_date():
    results = extract(
        """AWS Certified Machine Learning - Specialty
Amazon Web Services | 2022"""
    )

    assert len(results) == 1

    cert = results[0]

    assert cert.name == "AWS Certified Machine Learning - Specialty"
    assert cert.issuer == "Amazon Web Services"
    assert cert.date == "2022"


def test_certification_name_only():
    results = extract("AWS Certified Machine Learning - Specialty")

    assert len(results) == 1
    assert results[0].name == "AWS Certified Machine Learning - Specialty"


def test_certification_with_year():
    results = extract("AWS Certified Machine Learning - Specialty 2022")

    assert len(results) == 1
    assert results[0].name == "AWS Certified Machine Learning - Specialty"
    assert results[0].date == "2022"


def test_multiple_certifications():
    results = extract(
        """AWS Certified Machine Learning - Specialty
Amazon Web Services | 2022

Google Professional Data Engineer
Google Cloud | 2023"""
    )

    assert len(results) == 2

    assert results[0].name == "AWS Certified Machine Learning - Specialty"
    assert results[0].issuer == "Amazon Web Services"
    assert results[0].date == "2022"

    assert results[1].name == "Google Professional Data Engineer"
    assert results[1].issuer == "Google Cloud"
    assert results[1].date == "2023"


def test_semicolon_separated_certifications():
    results = extract(
        "AWS Certified Machine Learning - Specialty; Google Professional Data Engineer"
    )

    assert len(results) == 2


def test_bullet_certification():
    results = extract(
        """• AWS Certified Machine Learning - Specialty
Amazon Web Services | 2022"""
    )

    assert len(results) == 1
    assert results[0].name == "AWS Certified Machine Learning - Specialty"
    assert results[0].issuer == "Amazon Web Services"
    assert results[0].date == "2022"