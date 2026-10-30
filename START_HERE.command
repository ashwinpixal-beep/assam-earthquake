#!/bin/bash
# Double-click launcher for macOS: starts the site and opens the browser.
cd "$(dirname "$0")" || exit 1

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python 3 was not found. Install it from https://python.org and run this again."
  read -r -p "Press Enter to close…" _
  exit 1
fi

echo "Starting the damage-risk site…"
"$PY" start.py
status=$?
if [ $status -ne 0 ]; then
  echo
  echo "The site stopped with an error (exit $status). The message above explains why."
  read -r -p "Press Enter to close…" _
fi
exit $status
