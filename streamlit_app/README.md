# Streamlit Model Memory Dashboard 🚀

A lightweight Streamlit UI to explore Hugging Face models, estimate their inference memory footprint and compare models side‑by‑side.

## Features ✨
- 🔍 **Model search**: Type a model ID to fetch suggestions (Hugging Face), set it, and view details.
- 🧮 **Memory estimation**: Calls the underlying estimator to show total memory, parameter counts, and dtype breakdown.
- 📊 **Charts**: Interactive Plotly charts for memory comparison (bar, horizontal bar, pie, line).
- 🧭 **Display options**: Toggle metadata, timestamps, chart type, and color.
- 💾 **Cache management**: View cached models, delete individual entries, or clear all; live count of cached items.
- 📤 **Export**: Download comparison data as CSV, JSON, HTML (interactive), or image guidance.
- 🎯 **Sample data**: One-click sample models to demo the UI.

## Running Locally 🛠️
From this folder:
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app will start at the default Streamlit URL (typically <http://localhost:8501>).

## Usage Flow 🚦
1) Set the number of models to compare in the sidebar.  
2) Enter model IDs (or pick from suggestions) and expand “Memory Requirements” to fetch estimates.  
3) Switch chart type/color and toggle metadata as needed.  
4) Use “Compare Models” to populate comparison data and charts.  
5) Manage cached entries via the Cache Management expander; delete individual models or clear all.  
6) Export results in your preferred format from the Export section.  

## Notes 📝
- The UI relies on the local estimator (`memory_estimator.py`) which wraps `estimate_model_memory` and caching helpers.  
- Cached results live in `st.session_state` and are cleared via the UI controls.  
- Requirements are isolated in `streamlit_app/requirements.txt` to keep the upstream project metadata untouched.  
