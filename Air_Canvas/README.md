# Air Canvas

A Flask-based hand-gesture drawing app.

## ✅ Requirements

- **Python 3.10.15** or **Python 3.11.x** (stable release)
- `pip` installed

> **Important:** Python preview/nightly/alpha builds (e.g., `3.10.0a7`) are not supported and will often break binary packages like NumPy/OpenCV.

## 🚀 Setup

1. **Use a stable Python version** (example using Windows `py` launcher):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. **Install dependencies**

```powershell
pip install -r requirements.txt
```

3. **Run the app**

```powershell
python app.py
```

## 🛠 Troubleshooting

### NumPy / OpenCV import errors (DLL load failed)
If you see errors like:

- `ImportError: DLL load failed while importing mtrand`
- `OpenCV bindings requires "numpy" package`

Then you are likely running a Python build that is incompatible with the installed binary wheels.

**Fix:** Use a stable release (Python 3.11.x or 3.10.15) and recreate the venv.

## ✅ Notes

- The app will show a friendly error page if dependencies are missing.
- The gesture system supports:
  - **Pinch = Draw**
  - **Open Palm = Erase**
  - **Move Hand = Idle** (no drawing)
