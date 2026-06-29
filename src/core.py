from __future__ import annotations

def format_size(mb):
    if mb < 1:
        return f"{mb * 1024:.1f} KB"
    return f"{mb:.2f} MB"
