"""
nb_loader.py — Load notebook-generated assets into Streamlit pages.

All helpers read from FoodDeliveryAI/assets/nb_outputs/ which is populated
by running run_notebooks.py (or the admin page 0_Run_Notebooks.py).
"""

import os
import json
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
_APP_DIR   = Path(__file__).parent.parent
NB_OUT_DIR = _APP_DIR / "assets" / "nb_outputs"
STATUS_FILE= NB_OUT_DIR / "run_status.json"


# ── Status helpers ─────────────────────────────────────────────────────────────

def outputs_ready() -> bool:
    """Return True if notebooks have been run and status file exists."""
    return STATUS_FILE.exists()


@st.cache_data(show_spinner=False)
def load_status() -> dict:
    """Load the run_status.json produced by run_notebooks.py."""
    if not STATUS_FILE.exists():
        return {}
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def notebook_ok(name: str) -> bool:
    """Return True if a specific notebook ran successfully."""
    status = load_status()
    return status.get("notebooks", {}).get(name, {}).get("status") == "success"


# ── Image helpers ──────────────────────────────────────────────────────────────

def nb_image_path(filename: str) -> Path:
    return NB_OUT_DIR / filename


def load_nb_image(filename: str):
    """Load a notebook-output image. Returns PIL Image or None."""
    p = nb_image_path(filename)
    if p.exists():
        return Image.open(p)
    return None


def show_nb_image(filename: str, caption: str = "", use_column_width: bool = True):
    """Display a notebook-output image in Streamlit, or a warning if missing."""
    p = nb_image_path(filename)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=use_column_width)
    else:
        st.warning(f"⚠️ Image not found: `{filename}`. Please run the notebooks first.")


def show_all_inline(nb_name: str, caption_prefix: str = ""):
    """
    Display all inline-extracted images for a notebook (e.g., Part3).
    These are named like `part3_inline_00.png`, `part3_inline_01.png`, …
    """
    prefix = nb_name.lower() + "_inline_"
    images = sorted(NB_OUT_DIR.glob(f"{prefix}*.png"))
    if not images:
        st.warning(f"⚠️ No inline images found for {nb_name}. Please run the notebooks first.")
        return
    for i, img_path in enumerate(images):
        st.image(str(img_path), caption=f"{caption_prefix} Plot {i+1}" if caption_prefix else "", use_container_width=True)


# ── CSV helpers ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_nb_csv(filename: str):
    """Load a notebook-output CSV as a DataFrame. Returns None if missing."""
    import pandas as pd
    p = NB_OUT_DIR / filename
    if p.exists():
        return pd.read_csv(p)
    return None


# ── Metrics helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_part3_metrics() -> dict:
    """Load classification metrics extracted from Part3's text output."""
    status = load_status()
    return status.get("notebooks", {}).get("Part3", {}).get("metrics", {})


# ── "Not ready" banner ─────────────────────────────────────────────────────────

def require_outputs(nb_names: list[str] | None = None) -> bool:
    """
    Display a prominent banner if notebook outputs aren't ready yet.
    Returns True if ready, False otherwise.
    Call at the top of each page.
    """
    if not outputs_ready():
        st.error(
            "### ⚠️ Notebook outputs not found\n\n"
            "Please go to **🚀 Run Notebooks** (first page in sidebar) and "
            "click **Run All Notebooks** to generate results from your Part1–Part7 notebooks."
        )
        st.stop()
        return False

    if nb_names:
        missing = [n for n in nb_names if not notebook_ok(n)]
        if missing:
            st.warning(
                f"⚠️ The following notebooks didn't complete successfully: "
                f"`{'`, `'.join(missing)}`. Some results may be missing."
            )

    return True
