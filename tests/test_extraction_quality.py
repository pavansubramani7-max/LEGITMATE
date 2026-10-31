from __future__ import annotations

import unittest
from unittest.mock import patch

from app.extraction.dates import extract_dates
from app.extraction.entities import extract_entities


class ExtractionQualityTests(unittest.TestCase):
    def entities(self, text: str) -> list[dict[str, object]]:
        with patch("app.extraction.entities.get_nlp", return_value=None):
            return extract_entities("doc", "test.pdf", 1, text)

    def test_blank_template_labels_are_not_people_properties_or_amounts(self) -> None:
        text = """SALE DEED FOR FLAT THIS
        Ward No. ______ Block No. ______ Municipal Corporation
        Gift Settlement; Document No. ______; Volume No. ______
        Revenue Records; Indian Stamp Act; Urban Agglomeration;
        Village Rate Per acre; Market Value; Society; Rs. ______; Rs,
        Stilt Floor; Assigned Lands; Muncipal Corportation; Floor With; Plinth Area.
        """
        entities = self.entities(text)
        self.assertEqual([], entities)
        self.assertEqual([], extract_dates("doc", "test.pdf", 1, text))

    def test_filled_names_organization_and_property_identifiers_remain(self) -> None:
        text = """SELLER: Rajesh Kumar Sharma. BUYER: Priya Anand Verma.
        Witnesses:
        Anil Mehta and Sunita Rao.
        Property: Plot No. 118, Survey No. 44/B, Flat No. 5B.
        Jaipur Development Authority.
        """
        found = {(str(item["kind"]), str(item["value"])) for item in self.entities(text)}
        self.assertIn(("person", "Rajesh Kumar Sharma"), found)
        self.assertIn(("person", "Priya Anand Verma"), found)
        self.assertIn(("person", "Anil Mehta"), found)
        self.assertIn(("person", "Sunita Rao"), found)
        self.assertIn(("org", "Jaipur Development Authority"), found)
        for identifier in ("Plot No. 118", "Survey No. 44/B", "Flat No. 5B"):
            self.assertTrue(any(kind == "property" and value.casefold() == identifier.casefold() for kind, value in found), identifier)

    def test_amount_requires_digits_and_supports_indian_grouping(self) -> None:
        items = extract_dates(
            "doc", "test.pdf", 1,
            "Blank consideration Rs. ______ and Rs,; actual consideration Rs. 1,00,000.",
        )
        amounts = [str(item["value"]) for item in items if item["kind"] == "amount"]
        self.assertEqual(["Rs. 1,00,000"], amounts)

    def test_reference_date_is_labeled_and_invalid_calendar_date_is_dropped(self) -> None:
        items = extract_dates(
            "doc", "test.pdf", 1,
            "GO MS No. 733 Revenue Dept dated 31/10/1988; invalid date 31/02/2024.",
        )
        dates = [item for item in items if item["kind"] == "date"]
        self.assertEqual(1, len(dates))
        self.assertEqual("legal_reference", dates[0]["event_type"])
        self.assertEqual("1988-10-31", dates[0]["event_date"])


if __name__ == "__main__":
    unittest.main()
