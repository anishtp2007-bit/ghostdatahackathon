# Optional local OCR setup for Windows

The scanner uses PyMuPDF for PDF extraction and calls the local Tesseract OCR engine for screenshots and image-only PDF pages. OCR runs on your computer; it does not use a paid API.

## Install Tesseract

1. Download the 64-bit Windows installer from the [UB Mannheim Tesseract Windows installation page](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install it in the suggested folder, normally `C:\Program Files\Tesseract-OCR`.
3. Make sure the English language data (`eng.traineddata`) is included.
4. Add the Tesseract install folder to Windows `PATH`, or set `TESSERACT_CMD` to the full path of `tesseract.exe` (normally `C:\Program Files\Tesseract-OCR\tesseract.exe`).
5. Close and reopen VS Code so the terminal sees the updated environment.
6. In the VS Code terminal, check that `tesseract --version` prints a version. Run the evidence scan with `python code/main.py --scan-evidence`; its report shows whether Tesseract was found.

The scanner also checks common Windows install folders and the `TESSERACT_CMD` environment variable. If Tesseract is not installed, it marks image files as partial and explains that their text could not be extracted.
