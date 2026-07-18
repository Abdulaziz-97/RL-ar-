"""Thin CLI wrapper — prefer `python -m rlvr_sota.audit_coldstart`."""

from rlvr_sota.audit_coldstart import main

if __name__ == "__main__":
    raise SystemExit(main())
