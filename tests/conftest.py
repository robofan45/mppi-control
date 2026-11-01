"""Draw off screen, so tests never open windows or depend on the screen."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
