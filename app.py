"""Streamlit prototype application for Kuku Promo Finder.

Allows uploading an episode script zip archive or pointing to a folder,
running the promo finding pipeline with real-time progress, inspecting
ranked promos with event-by-event dialogue breakdowns, and downloading
results in JSON, CSV, or Markdown.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from promo_finder.finder import run_pipeline
from promo_finder.parse import format_timestamp, load_episodes
from promo_finder.rank import rank_and_deduplicate_promos

st.set_page_config(
    page_title="Kuku Promo Finder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 Kuku Promo Finder")
st.caption("AI-Powered Viral Promo Generator for Mobile Microdramas (ReelShort / DramaBox Style)")

# Sidebar Configuration
st.sidebar.header("Configuration")
provider = st.sidebar.selectbox(
    "LLM Provider",
    options=["vertex", "openai_compat"],
    index=0 if os.getenv("LLM_PROVIDER", "vertex") == "vertex" else 1,
)
default_model = os.getenv("LLM_MODEL", "gemini-2.5-flash" if provider == "vertex" else "gpt-4o-mini")
model_name = st.sidebar.text_input("Model Name", value=default_model)
top_n_promos = st.sidebar.slider("Number of Promos to Rank", min_value=5, max_value=25, value=15)
use_cache = st.sidebar.checkbox("Enable Disk Cache", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Architecture**:
    1. Pure-code screenplay parser (ID snapping)
    2. Two-pass candidate extraction (Gemini)
    3. Strict rule validation & repair (20-60s)
    4. Cold-viewer LLM judge (zero-context stranger)
    5. Weighted ranking & diversity filtering
    """
)

# Input Selection
st.subheader("1. Screenplay Input")
input_mode = st.radio("Choose Input Source:", ["Upload Scripts (.zip)", "Local Folder Path"], horizontal=True)

input_source: str | None = None
temp_dir: str | None = None

if input_mode == "Upload Scripts (.zip)":
    uploaded_file = st.file_uploader("Upload Episode Scripts Archive (.zip)", type=["zip"])
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        zip_path = Path(temp_dir) / "uploaded.zip"
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_source = str(zip_path)
        st.success(f"Loaded ZIP file ({uploaded_file.size / 1024:.1f} KB)")
else:
    folder_input = st.text_input("Folder Path to Episodes:", value=str(Path("../data").resolve()))
    if folder_input and Path(folder_input).exists():
        input_source = folder_input
        st.success(f"Using local folder: {folder_input}")
    elif folder_input:
        st.warning(f"Path does not exist: {folder_input}")

max_episodes_to_run = st.number_input("Max Episodes to Process (0 = All)", min_value=0, max_value=100, value=5)

if st.button("🚀 Find Best Promos", type="primary", disabled=input_source is None):
    # Set runtime environment for swappable provider
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model_name

    progress_bar = st.progress(0, text="Loading and parsing episodes...")
    status_text = st.empty()

    try:
        episodes = load_episodes(input_source)
        if max_episodes_to_run > 0:
            episodes = episodes[:max_episodes_to_run]

        st.info(f"Loaded {len(episodes)} episodes ({sum(len(ep.events) for ep in episodes):,} events).")

        # Same pipeline as the CLI; progress lines are shown as they arrive.
        log_lines: list[str] = []

        def show_progress(msg: str) -> None:
            log_lines.append(msg)
            status_text.code("\n".join(log_lines[-12:]))
            progress_bar.progress(min(95, len(log_lines)) / 100, text=msg.strip()[:80])

        result = run_pipeline(episodes, workers=3, use_cache=use_cache, progress=show_progress)
        ranked = rank_and_deduplicate_promos(result["evaluated"], top_n=top_n_promos)

        progress_bar.progress(100, text="Done!")
        st.session_state["ranked_promos"] = ranked
        st.session_state["episodes_by_num"] = {ep.number: ep for ep in episodes}
        st.session_state["rejections_count"] = len(result["rejections"])
        st.session_state["raw_count"] = len(result["raw_candidates"])

    except Exception as exc:
        st.error(f"Pipeline error: {exc}")

# Display Results if available
if "ranked_promos" in st.session_state:
    ranked = st.session_state["ranked_promos"]
    st.subheader(f"🏆 Top {len(ranked)} Ranked Promos")

    col1, col2, col3 = st.columns(3)
    col1.metric("Proposals Evaluated", st.session_state.get("raw_count", len(ranked)))
    col2.metric("Filtered Candidates", len(ranked))
    col3.metric("Invalid Cuts Blocked", st.session_state.get("rejections_count", 0))

    # Summary table
    table_data = []
    for p in ranked:
        j = p.get("judge", {})
        table_data.append({
            "Rank": f"#{p['rank']}",
            "Promo ID": p["promo_id"],
            "Score": f"{p['score']:.2f}",
            "Duration": f"{p['total_duration']}s",
            "Hook (1-5)": j.get("hook_strength", "-"),
            "Cliff (1-5)": j.get("cliffhanger_strength", "-"),
            "Clarity (1-5)": j.get("standalone_clarity", "-"),
            "Cross-Episode": "Yes" if p["is_cross_episode"] else "No",
        })
    st.dataframe(table_data, use_container_width=True)

    # Detailed expandable view per promo
    st.subheader("🔍 Detailed Promo Inspections")
    for p in ranked:
        j = p.get("judge", {})
        with st.expander(f"Rank #{p['rank']}: {p['promo_id']} (Duration: {p['total_duration']}s | Score: {p['score']:.2f})"):
            st.markdown(f"**Hook**: {p.get('hook_text', '')}")
            st.markdown(f"**Cliffhanger**: {p.get('cliffhanger_text', '')}")
            st.markdown(f"**Why it works**: {p['reason']}")
            if p.get("post_first_note"):
                st.info(f"**Why Post First**: {p['post_first_note']}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Hook Strength", f"{j.get('hook_strength')}/5")
            m2.metric("Cliffhanger Tension", f"{j.get('cliffhanger_strength')}/5")
            m3.metric("Standalone Clarity", f"{j.get('standalone_clarity')}/5")

            st.markdown(f"**Cold-Viewer Impression**: *\"{j.get('summary')}\"*")
            st.markdown(f"**Unresolved Burning Question**: *\"{j.get('open_question')}\"*")

            st.markdown("#### Segment Timeline & Script Breakdown")
            for seg in p["segments"]:
                st.markdown(f"**Episode {seg['episode']}** (`{seg['start_ts']}` – `{seg['end_ts']}`, {seg['duration']}s):")
                ep = st.session_state.get("episodes_by_num", {}).get(seg["episode"])
                if ep:
                    for ev in ep.events:
                        if ev.kind != "camera" and seg["start"] - 0.01 <= ev.start < seg["end"]:
                            st.markdown(f"- `[{format_timestamp(ev.start)}]` {ev.line()}")

    # Export downloads
    st.subheader("💾 Export Deliverables")
    d1, d2, d3 = st.columns(3)

    json_str = json.dumps(ranked, indent=2)
    d1.download_button("Download JSON", json_str, file_name="kuku_top_promos.json", mime="application/json")

    # CSV export
    csv_buf = io.StringIO()
    if ranked:
        writer = csv.DictWriter(csv_buf, fieldnames=[
            "rank", "promo_id", "score", "total_duration", "segments", "hook_text", "cliffhanger_text",
            "reason", "is_cross_episode", "post_first_note",
        ])
        writer.writeheader()
        for p in ranked:
            writer.writerow({
                "rank": p["rank"],
                "promo_id": p["promo_id"],
                "score": p["score"],
                "total_duration": p["total_duration"],
                "segments": "; ".join(f"Ep {s['episode']} [{s['start_ts']} - {s['end_ts']}]" for s in p["segments"]),
                "hook_text": p.get("hook_text", ""),
                "cliffhanger_text": p.get("cliffhanger_text", ""),
                "reason": p["reason"],
                "is_cross_episode": p["is_cross_episode"],
                "post_first_note": p.get("post_first_note", ""),
            })
    d2.download_button("Download CSV", csv_buf.getvalue(), file_name="kuku_top_promos.csv", mime="text/csv")

    # Markdown export
    md_buf = ["# Top Ranked Promos\n"]
    for p in ranked:
        md_buf.append(f"## #{p['rank']} - {p['promo_id']} ({p['total_duration']}s, Score {p['score']:.2f})\n")
        for s in p["segments"]:
            md_buf.append(f"- Episode {s['episode']}: {s['start_ts']} - {s['end_ts']}\n")
        md_buf.append(f"- Hook: {p.get('hook_text', '')}\n")
        md_buf.append(f"- Cliffhanger: {p.get('cliffhanger_text', '')}\n")
        md_buf.append(f"- Why it works: {p['reason']}\n")
        md_buf.append(f"- Post first: {p.get('post_first_note', '')}\n")
    d3.download_button("Download Markdown", "\n".join(md_buf), file_name="kuku_top_promos.md", mime="text/markdown")
