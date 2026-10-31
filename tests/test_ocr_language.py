from __future__ import annotations

import unittest
from unittest.mock import patch

from app.ingestion.pdf_reader import detect_ocr_language


class AutomaticOcrLanguageTests(unittest.TestCase):
    @patch("app.ingestion.pdf_reader._installed_tesseract_languages", return_value={"eng", "hin"})
    @patch("app.ingestion.pdf_reader.shutil.which", return_value="tesseract")
    def test_document_script_selects_installed_hindi_pack(self, _which, _languages):
        result = detect_ocr_language(b"pdf-bytes", "यह संपत्ति विक्रय का दस्तावेज है")
        self.assertEqual(result["language"], "hin")
        self.assertFalse(result["fallback"])
        self.assertEqual(result["method"], "document text language estimate")

    @patch("app.ingestion.pdf_reader._installed_tesseract_languages", return_value={"eng", "ara"})
    @patch("app.ingestion.pdf_reader.shutil.which", return_value="tesseract")
    def test_uncertain_arabic_language_uses_detected_arabic_script(self, _which, _languages):
        result = detect_ocr_language(b"pdf-bytes", "هذا مستند قانوني باللغة العربية")
        self.assertEqual(result["language"], "ara")
        self.assertFalse(result["fallback"])
        self.assertEqual(result["method"], "document writing-system estimate")

    @patch("app.ingestion.pdf_reader._installed_tesseract_languages", return_value={"eng"})
    @patch("app.ingestion.pdf_reader.shutil.which", return_value="tesseract")
    def test_missing_detected_pack_uses_reported_installed_fallback(self, _which, _languages):
        result = detect_ocr_language(b"pdf-bytes", "यह संपत्ति विक्रय का दस्तावेज है")
        self.assertEqual(result["language"], "eng")
        self.assertTrue(result["fallback"])

    @patch("app.ingestion.pdf_reader._osd_script", return_value=("Devanagari", "12.4"))
    @patch("app.ingestion.pdf_reader._installed_tesseract_languages", return_value={"eng", "hin"})
    @patch("app.ingestion.pdf_reader.shutil.which", return_value="tesseract")
    def test_scan_only_pdf_uses_local_script_detection(self, _which, _languages, _osd):
        result = detect_ocr_language(b"pdf-bytes")
        self.assertEqual(result["language"], "hin")
        self.assertEqual(result["method"], "local Tesseract script detection")


if __name__ == "__main__":
    unittest.main()
