# GitHub Upload Notes

Recommended upload target: the entire `electricity_time_series_project/` folder.

Suggested commands:

```bash
cd electricity_time_series_project
git init
git add .
git commit -m "Add electricity time series project"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```

Before pushing, verify:

```bash
python -m py_compile code/main.py code/sarima.py code/var.py code/garch.py
python code/main.py var garch
```

The full `python code/main.py` run includes SARIMA/SARIMAX and may take longer.
