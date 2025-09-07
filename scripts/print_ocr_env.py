#!/usr/bin/env python3
import os
import platform
import shutil
import subprocess


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return e.output


def main():
    print("=== System ===")
    print("Platform:", platform.platform())
    print("Python:", platform.python_version())

    print("\n=== Tesseract (native) ===")
    tess_path = shutil.which("tesseract")
    print("which tesseract:", tess_path or "NOT FOUND")
    print("tesseract --version:\n", sh("tesseract --version"))
    print("tesseract --list-langs:\n", sh("tesseract --list-langs"))
    print("TESSDATA_PREFIX:", os.environ.get("TESSDATA_PREFIX", ""))

    print("\n=== Python packages ===")
    try:
        import fitz  # type: ignore
        print("PyMuPDF:", getattr(fitz, "__version__", getattr(fitz, "VersionBind", None)))
    except Exception as e:
        print("PyMuPDF: NOT INSTALLED", e)

    try:
        from PIL import __version__ as PIL_VERSION  # type: ignore
        print("Pillow:", PIL_VERSION)
    except Exception as e:
        print("Pillow: NOT INSTALLED", e)

    try:
        import pytesseract  # type: ignore
        print("pytesseract:", getattr(pytesseract, "__version__", "unknown"))
        try:
            print("pytesseract get_tesseract_version:", pytesseract.get_tesseract_version())
        except Exception as e:
            print("pytesseract get_tesseract_version failed:", e)
        print("pytesseract tesseract_cmd:", pytesseract.pytesseract.tesseract_cmd)
    except Exception as e:
        print("pytesseract: NOT INSTALLED", e)


if __name__ == "__main__":
    main()
