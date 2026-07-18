"""Thin CLI wrapper — prefer `python -m rlvr_pipeline.audit_coldstart`."""

from rlvr_pipeline.audit_coldstart import main

if __name__ == "__main__":
    raise SystemExit(main())
