"""OpenGlottal Qt GUI: video input, model selection, overlay display, frame range."""

def run_gui():
    """Launch the Qt overlay GUI. Import deferred to avoid double-load when run as __main__."""
    from .main import run_gui as _run_gui
    return _run_gui()

__all__ = ["run_gui"]
