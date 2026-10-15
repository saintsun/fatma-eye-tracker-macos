#!/usr/bin/env python3
"""Backwards-compatible entry point.

The tracker used to live in this single file; it is now the ``fatma``
package (the original is kept verbatim under ``legacy/`` for reference).
Running this script still works, and so does ``python -m fatma``.
"""

import sys

from fatma.app import main

if __name__ == "__main__":
    sys.exit(main())
