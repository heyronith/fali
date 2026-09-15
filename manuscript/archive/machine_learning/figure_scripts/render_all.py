"""Regenerate all FALI main figures and provenance sidecars."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from render_contact_sheet import render as render_contact  # noqa: E402
from render_fig1 import render as render_fig1  # noqa: E402
from render_fig2 import render as render_fig2  # noqa: E402
from render_fig3 import render as render_fig3  # noqa: E402
from render_fig4 import render as render_fig4  # noqa: E402
from render_fig5 import render as render_fig5  # noqa: E402
from render_readability_review import render as render_readability  # noqa: E402
from validate_figure_inputs import main as validate_inputs  # noqa: E402


def main() -> int:
    validate_inputs()
    render_fig1()
    render_fig2()
    render_fig3()
    render_fig4()
    render_fig5()
    contact, printsize = render_contact()
    print(f"Contact sheet: {contact}")
    print(f"Print-size contact sheet: {printsize}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
