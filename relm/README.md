# ReLM - Regular Expressions for Language Models
This package contains ReLM, a system for testing language models with regular
expressions.

## Manual Install
From this directory, run:
```bash
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip uninstall -y relm-mkuchnik
python3 -m pip install dist/*.whl
```

Alternatively, the commands are in `install.sh`.
