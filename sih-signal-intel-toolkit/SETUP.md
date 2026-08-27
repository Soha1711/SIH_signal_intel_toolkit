# Environment Setup

Follow this exactly — it should take under 30 minutes. If something fails, post the
exact error in the team chat rather than troubleshooting solo; someone has likely
hit it already.

## 1. Install Python 3

Check if you already have it:
```bash
python3 --version
```
Need 3.9 or newer. If missing, install from https://www.python.org/downloads/
(Windows: check "Add Python to PATH" during install).

## 2. Create a virtual environment (keeps this project's packages isolated)

```bash
python3 -m venv venv

# Activate it:
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows (Command Prompt)
venv\Scripts\Activate.ps1       # Windows (PowerShell)
```

You'll know it worked when your terminal prompt shows `(venv)` at the start.
**Activate this every time** before working on the project.

## 3. Install core Python packages

```bash
pip install numpy scipy soundfile matplotlib pyqtgraph PyQt5
```

## 4. Install GNU Radio

GNU Radio is not a pip package — install it separately:

- **Windows:** download the installer from https://www.gnuradio.org/download/ (Windows builds via Conda/radioconda are easiest — use https://github.com/ryanvolz/radioconda)
- **macOS:** `brew install gnuradio` (requires Homebrew)
- **Linux (Ubuntu/Debian):** `sudo apt install gnuradio`

## 5. Verify it all works

```bash
python3 -c "import numpy, scipy, soundfile, PyQt5; print('Core packages OK')"
gnuradio-companion   # should open the GNU Radio Companion GUI window
```

If `gnuradio-companion` opens a window, you're done. Close it and move on.

## 6. Clone the repo (if you haven't already)

```bash
git clone <repo-url>
cd sih-signal-intel-toolkit
```

## 7. Confirm your module folder

Check `README.md` for which folder is yours, and open `docs/interface_contract.md`
before writing any code — it defines exactly what data your module receives and
must output.

## Troubleshooting

- **PyQt5 install fails on macOS with Apple Silicon:** try `pip install PyQt5 --config-settings --confirm-license= --verbose`, or fall back to `pip install PyQt6` and flag it in team chat (interface calls would need adjusting).
- **GNU Radio not found after install:** restart your terminal, or on Windows use the "Anaconda/Radioconda Prompt" specifically, not the regular terminal.
- **Still stuck after 15 minutes:** post in the team chat, don't lose more time solo.
