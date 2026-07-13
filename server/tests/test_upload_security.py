import asyncio
import os
from io import BytesIO

from fastapi import UploadFile

from core import document_processor


def test_uploaded_filename_cannot_escape_temp_directory(monkeypatch, tmp_path):
    safe_dir = tmp_path / "uploaded_files"
    monkeypatch.setattr(document_processor, "TEMPFILE_UPLOAD_DIRECTORY", str(safe_dir))

    malicious_filename = "../../../../evil.txt"
    upload = UploadFile(file=BytesIO(b"malicious content"), filename=malicious_filename)

    saved_paths = asyncio.run(document_processor.save_uploaded_file([upload]))

    assert len(saved_paths) == 1
    saved_path = os.path.abspath(saved_paths[0])
    assert saved_path.startswith(os.path.abspath(str(safe_dir)))
    assert os.path.exists(saved_path)

    # Confirm it truly did not escape upward — no file was written outside the sandbox.
    escaped_path = os.path.abspath(os.path.join(str(tmp_path), "..", "..", "..", "..", "evil.txt"))
    assert not os.path.exists(escaped_path)
