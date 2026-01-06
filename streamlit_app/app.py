import streamlit as st
import pandas as pd
from datetime import datetime
import io
import json
import asyncio
import plotly.graph_objects as go
import plotly.express as px
from memory_estimator import estimate_model_memory, search_huggingface_models

# Page configuration
st.set_page_config(
    page_title="Model Comparison Tool",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .model-column {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .model-header {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 10px;
    }
    .model-input {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 5px;
        padding: 10px;
        color: white;
    }
    .stTextInput > label {
        font-size: 14px;
        font-weight: 500;
    }
    .sidebar-section {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 16px;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 12px;
    }
    .header-container {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .header-container h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    .header-container p {
        margin: 5px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .empty-state {
        text-align: center;
        padding: 40px;
        color: #888;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_ids' not in st.session_state:
    st.session_state.model_ids = ['', '', '', '', '']
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = {}
if 'memory_estimates' not in st.session_state:
    st.session_state.memory_estimates = {}


@st.cache_data
def get_cached_memory_estimate(model_id: str):
    """
    Cached function to fetch memory requirements for a model.
    Results are cached so they don't need to be refetched on every interaction.
    """
    return asyncio.run(estimate_model_memory(model_id))


def display_memory_estimate(model_id: str):
    """Fetch and display memory requirements for a model."""
    with st.spinner(f"Estimating memory for {model_id}..."):
        try:
            # Use cached function instead of calling estimate_model_memory directly
            result = get_cached_memory_estimate(model_id)
            
            if "error" in result:
                st.warning(f"⚠️ {result['error']}")
                return None
            
            # Store the result
            st.session_state.memory_estimates[model_id] = result
            
            # Display memory breakdown
            total_gb = result.get('total_gb', 0)
            total_params = result.get('total_params_formatted', 'N/A')
            
            # Create columns for metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Total Memory", f"{total_gb:.2f} GB")
            with metric_col2:
                st.metric("Parameters", total_params)
            
            # Create bar chart for dtype breakdown
            breakdown = result.get('breakdown', {})
            if breakdown:
                st.subheader("Memory Breakdown by Dtype", anchor=None)
                
                # Prepare data for chart
                chart_data = {
                    'Dtype': [],
                    'Memory (GB)': [],
                    'Percentage': []
                }
                for dtype, info in breakdown.items():
                    chart_data['Dtype'].append(dtype)
                    chart_data['Memory (GB)'].append(info['gb'])
                    chart_data['Percentage'].append(info['percentage'])
                
                df_breakdown = pd.DataFrame(chart_data)
                
                # Display bar chart
                st.bar_chart(df_breakdown.set_index('Dtype')['Memory (GB)'])
                
                # Display detailed table
                st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error estimating memory: {str(e)}")
            return None


# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Settings section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Comparison Settings</div>', unsafe_allow_html=True)
    
    num_models = st.slider(
        "Number of Models to Compare",
        min_value=1,
        max_value=5,
        value=2,
        help="Select how many models you want to compare side-by-side (max 5)"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Save format section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    # Export settings in expander
    with st.expander("💾 Export Settings", expanded=False):
        st.markdown("**Select Export Formats:**")
        
        # Vertically aligned toggle switches (single column)
        export_csv = st.toggle("📄 CSV", value=True, key="export_csv")
        export_json = st.toggle("📋 JSON", value=False, key="export_json")
        export_html = st.toggle("🌐 HTML", value=False, key="export_html")
        export_image = st.toggle("📸 Image", value=False, key="export_image")
        
        # Build the save_format list based on toggles
        save_format = []
        if export_csv:
            save_format.append("CSV")
        if export_json:
            save_format.append("JSON")
        if export_html:
            save_format.append("HTML")
        if export_image:
            save_format.append("Image")
    
    # Additional settings
    with st.expander("👁️ Display Options", expanded=False):
        show_metadata = st.checkbox("Show Model Metadata", value=True)
        show_timestamps = st.checkbox("Show Timestamps", value=False)
        
        st.markdown("**Chart Customization:**")
        chart_type = st.selectbox(
            "Chart Type",
            options=["Bar Chart", "Horizontal Bar", "Pie Chart", "Line Chart"],
            key="chart_type"
        )
        chart_color = st.color_picker("Chart Color", value="#667eea", key="chart_color")
    
    # Cache Management
    with st.expander("🗑️ Cache Management", expanded=False):
        # Always read the current state from session state
        if "memory_estimates" not in st.session_state:
            st.session_state.memory_estimates = {}
        
        # Get fresh cache state every render
        cached_models = list(st.session_state.memory_estimates.keys())
        
        st.write(f"**Cached Models: {len(cached_models)}**")
        
        if cached_models:
            st.write("Cached models:")
            
            # Display each cached model with delete button using a cleaner layout
            for i, model in enumerate(cached_models):
                with st.container(border=True):
                    col1, col2 = st.columns([0.80, 0.20], gap="small")
                    
                    with col1:
                        st.markdown(f"**• {model}**")
                    
                    with col2:
                        if st.button("", icon = "🗑️", key=f"delete_cache_{i}_{model}", help="Delete"): # icon = "🗑️", , width="stretch"
                            if model in st.session_state.memory_estimates:
                                del st.session_state.memory_estimates[model]
                            st.success(f"✓ Deleted '{model}' from cache!")
                            st.rerun()
            
            # Clear all button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All Cache", key="clear_all_cache", width="content"):
                    st.session_state.memory_estimates.clear()
                    st.success("✓ Cache cleared!")
                    st.rerun()
            
            with col2:
                st.metric("Total Cached", len(cached_models))
        else:
            st.info("No cached models yet")
    
    # Info section
    st.markdown("---")
    st.markdown("""
    **ℹ️ About**
    
    This tool helps you compare multiple AI models side-by-side. 
    - Add model IDs (HuggingFace, custom, etc.)
    - Compare up to 5 models simultaneously
    - Export results in your preferred format
    """)

# ============ MAIN CONTENT ============
st.markdown('''
<div class="header-container">
    <h1>🤖 Model Comparison Tool</h1>
    <p>Compare multiple AI models side-by-side with ease</p>
</div>
''', unsafe_allow_html=True)

# Display selected number of model columns
st.markdown(f"### Compare {num_models} Model{'s' if num_models > 1 else ''}")

if num_models > 0:
    cols = st.columns(num_models, gap="medium")
    
    for idx in range(num_models):
        with cols[idx]:
            st.markdown(f'''
            <div class="model-column">
                <div class="model-header">
                    Model #{idx + 1}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Model ID input with autocomplete
            model_input = st.text_input(
                f"Model ID #{idx + 1}",
                value=st.session_state.model_ids[idx],
                placeholder=f"e.g., meta-llama/Llama-2-7b",
                key=f"model_input_{idx}",
                label_visibility="collapsed"
            )
            
            # Show autocomplete suggestions
            if model_input and len(model_input) > 2:
                with st.spinner("Searching models..."):
                    suggestions = asyncio.run(search_huggingface_models(model_input, limit=5))
                    if suggestions:
                        selected_model = st.selectbox(
                            "Suggestions",
                            options=suggestions,
                            key=f"suggest_{idx}",
                            label_visibility="collapsed"
                        )
                        model_id = selected_model
                    else:
                        model_id = model_input
            else:
                model_id = model_input
            
            # Store in session state
            st.session_state.model_ids[idx] = model_id
            
            # Display model info if provided
            if model_id:
                st.success(f"✓ Model ID set", icon="✅")
                
                # Create expander for memory estimation
                with st.expander("📊 Memory Requirements", expanded=False):
                    display_memory_estimate(model_id)
                
                if show_metadata:
                    with st.expander("ℹ️ Model Details", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Model #", idx + 1)
                        with col2:
                            st.metric("Status", "Ready")
                        
                        st.text(f"Model: {model_id}")
                        
                        if show_timestamps:
                            st.caption(f"Added: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info(f"Enter model ID #{idx + 1}")

# ============ COMPARISON & EXPORT ============
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🔄 Compare Models", use_container_width=True):
        # Get non-empty model IDs
        active_models = [m for m in st.session_state.model_ids[:num_models] if m]
        
        if len(active_models) == num_models:
            # Fetch memory estimations for all models
            with st.spinner("Fetching memory estimations for all models..."):
                st.session_state.comparison_data = {}
                all_success = True
                
                for i, model_id in enumerate(active_models):
                    try:
                        # Use cached function to avoid refetching on every rerun
                        result = get_cached_memory_estimate(model_id)
                        
                        if "error" in result:
                            st.warning(f"⚠️ Model {i+1} ({model_id}): {result['error']}")
                            all_success = False
                        else:
                            st.session_state.comparison_data[model_id] = result
                            st.session_state.memory_estimates[model_id] = result
                    except Exception as e:
                        st.warning(f"⚠️ Model {i+1} ({model_id}): {str(e)}")
                        all_success = False
                
                if all_success:
                    st.success(f"✓ Loaded and estimated memory for {len(active_models)} model(s)!")
                    st.rerun()  # Rerun to update cache display and charts
                else:
                    st.info("⚠️ Some models could not be loaded, but you can still export the ones that were successful.")
                    st.rerun()  # Rerun to update cache display with partially loaded models
        else:
            st.error(f"❌ Please fill all {num_models} model IDs before comparing")

with col2:
    if st.button("📥 Sample Data", use_container_width=True):
        st.info("Loading sample models for demonstration...")
        sample_models = [
            "meta-llama/Llama-2-7b",
            "mistralai/Mistral-7B-v0.1",
            "NousResearch/Nous-Hermes-2-7b"
        ]
        for idx, model in enumerate(sample_models[:num_models]):
            st.session_state.model_ids[idx] = model
        st.success("✓ Sample models loaded!")
        st.rerun()

with col3:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.model_ids = ['', '', '', '', '']
        st.session_state.comparison_data = {}
        st.success("✓ All cleared!")
        st.rerun()

# ============ COMPARISON VISUALIZATION ============
if st.session_state.comparison_data and len(st.session_state.comparison_data) > 1:
    st.markdown("### 📊 Memory Comparison Chart")
    
    # Prepare data for visualization
    models = []
    memories = []
    for model_id, memory_info in st.session_state.comparison_data.items():
        models.append(model_id)
        memories.append(round(memory_info.get('total_gb', 0), 2))
    
    # Create interactive Plotly visualization (with tooltips on hover)
    if chart_type == "Bar Chart":
        fig = go.Figure(data=[
            go.Bar(x=models, y=memories, marker_color=chart_color,
                   hovertemplate='<b>%{x}</b><br>Memory: %{y} GB<extra></extra>')
        ])
        fig.update_layout(
            title="Model Memory Requirements Comparison",
            xaxis_title="Model",
            yaxis_title="Memory (GB)",
            height=500,
            hovermode='x unified'
        )
    
    elif chart_type == "Horizontal Bar":
        fig = go.Figure(data=[
            go.Bar(y=models, x=memories, orientation='h', marker_color=chart_color,
                   hovertemplate='<b>%{y}</b><br>Memory: %{x} GB<extra></extra>')
        ])
        fig.update_layout(
            title="Model Memory Requirements Comparison",
            xaxis_title="Memory (GB)",
            yaxis_title="Model",
            height=500,
            hovermode='y unified'
        )
    
    elif chart_type == "Pie Chart":
        fig = go.Figure(data=[
            go.Pie(labels=models, values=memories,
                   hovertemplate='<b>%{label}</b><br>Memory: %{value} GB<br>Share: %{percent}<extra></extra>')
        ])
        fig.update_layout(
            title="Memory Distribution Among Models",
            height=500
        )
    
    elif chart_type == "Line Chart":
        fig = go.Figure(data=[
            go.Scatter(x=models, y=memories, mode='lines+markers', 
                      line=dict(color=chart_color, width=3),
                      marker=dict(size=10),
                      hovertemplate='<b>%{x}</b><br>Memory: %{y} GB<extra></extra>')
        ])
        fig.update_layout(
            title="Model Memory Requirements Comparison",
            xaxis_title="Model",
            yaxis_title="Memory (GB)",
            height=500,
            hovermode='x unified',
            showlegend=False
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show comparison table
    st.markdown("#### 📋 Detailed Comparison")
    comparison_data = []
    for model_id, memory_info in st.session_state.comparison_data.items():
        comparison_data.append({
            "Model": model_id,
            "Memory (GB)": round(memory_info.get('total_gb', 0), 2),
            "Parameters": memory_info.get('total_params_formatted', 'N/A'),
            "Type": memory_info.get('model_type', 'unknown'),
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

# ============ EXPORT SECTION ============
if st.session_state.comparison_data:
    st.markdown("### 💾 Export Results")
    
    # Info about export formats
    st.info("""
    **Export Format Guide:**
    - 📄 **CSV**: Raw data table (Excel-compatible, no charts)
    - 📋 **JSON**: Structured data with dtype breakdown (API/integration-ready)
    - 🌐 **HTML**: Interactive chart + data table (best for presentations)
    - 📸 **Image**: Screenshot your chart
    """)
    
    export_cols = st.columns(len(save_format) if save_format else 1)
    
    # Prepare data for export - convert memory estimations to dataframe
    export_data = []
    for model_id, memory_info in st.session_state.comparison_data.items():
        row = {
            "Model ID": model_id,
            "Total Memory (GB)": round(memory_info.get('total_gb', 0), 2),
            "Total Parameters": memory_info.get('total_params_formatted', 'N/A'),
            "Model Type": memory_info.get('model_type', 'unknown'),
        }
        
        # Add dtype breakdown
        breakdown = memory_info.get('breakdown', {})
        for dtype, info in breakdown.items():
            row[f"{dtype} (GB)"] = round(info.get('gb', 0), 2)
            row[f"{dtype} (%)"] = round(info.get('percentage', 0), 2)
        
        export_data.append(row)
    
    df_export = pd.DataFrame(export_data)
    
    if "CSV" in save_format:
        with export_cols[save_format.index("CSV")]:
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    if "JSON" in save_format:
        with export_cols[save_format.index("JSON")]:
            # Convert to nested structure for better JSON readability
            json_export = {}
            for model_id, memory_info in st.session_state.comparison_data.items():
                json_export[model_id] = {
                    "total_memory_gb": round(memory_info.get('total_gb', 0), 2),
                    "total_parameters": memory_info.get('total_params_formatted', 'N/A'),
                    "model_type": memory_info.get('model_type', 'unknown'),
                    "breakdown_by_dtype": {
                        dtype: {
                            "memory_gb": round(info.get('gb', 0), 2),
                            "percentage": round(info.get('percentage', 0), 2),
                            "parameters": info.get('params', 0),
                        }
                        for dtype, info in memory_info.get('breakdown', {}).items()
                    }
                }
            
            json_data = json.dumps(json_export, indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    if "HTML" in save_format:
        with export_cols[save_format.index("HTML")]:
            # Create the interactive chart for embedding
            models = []
            memories = []
            for model_id, memory_info in st.session_state.comparison_data.items():
                models.append(model_id)
                memories.append(round(memory_info.get('total_gb', 0), 2))
            
            # Generate chart based on selected type
            if chart_type == "Bar Chart":
                fig_export = go.Figure(data=[
                    go.Bar(x=models, y=memories, marker_color=chart_color,
                           hovertemplate='<b>%{x}</b><br>Memory: %{y} GB<extra></extra>')
                ])
                fig_export.update_layout(
                    title="Model Memory Requirements Comparison",
                    xaxis_title="Model",
                    yaxis_title="Memory (GB)",
                    height=500,
                    hovermode='x unified'
                )
            
            elif chart_type == "Horizontal Bar":
                fig_export = go.Figure(data=[
                    go.Bar(y=models, x=memories, orientation='h', marker_color=chart_color,
                           hovertemplate='<b>%{y}</b><br>Memory: %{x} GB<extra></extra>')
                ])
                fig_export.update_layout(
                    title="Model Memory Requirements Comparison",
                    xaxis_title="Memory (GB)",
                    yaxis_title="Model",
                    height=500,
                    hovermode='y unified'
                )
            
            elif chart_type == "Pie Chart":
                fig_export = go.Figure(data=[
                    go.Pie(labels=models, values=memories,
                           hovertemplate='<b>%{label}</b><br>Memory: %{value} GB<br>Share: %{percent}<extra></extra>')
                ])
                fig_export.update_layout(
                    title="Memory Distribution Among Models",
                    height=500
                )
            
            else:  # Line Chart
                fig_export = go.Figure(data=[
                    go.Scatter(x=models, y=memories, mode='lines+markers', 
                              line=dict(color=chart_color, width=3),
                              marker=dict(size=10),
                              hovertemplate='<b>%{x}</b><br>Memory: %{y} GB<extra></extra>')
                ])
                fig_export.update_layout(
                    title="Model Memory Requirements Comparison",
                    xaxis_title="Model",
                    yaxis_title="Memory (GB)",
                    height=500,
                    hovermode='x unified',
                    showlegend=False
                )
            
            # Generate HTML with embedded chart
            html_data = df_export.to_html(index=False)
            chart_html = fig_export.to_html(include_plotlyjs='cdn', div_id="plotly_chart")
            
            html_full = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Model Comparison Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    h1 {{ color: #667eea; margin-bottom: 10px; }}
                    .metadata {{ color: #666; font-size: 14px; margin-bottom: 30px; }}
                    .chart-section {{ margin: 30px 0; }}
                    .data-section {{ margin: 30px 0; }}
                    h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #667eea; color: white; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 Model Memory Comparison Report</h1>
                    <div class="metadata">
                        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Models Compared:</strong> {len(models)}</p>
                    </div>
                    
                    <div class="chart-section">
                        <h2>Interactive Comparison Chart</h2>
                        <p style="color: #666; font-size: 13px;">💡 Tip: Hover over the chart for exact values. Use the toolbar to zoom, pan, and download the chart as an image.</p>
                        {chart_html}
                    </div>
                    
                    <div class="data-section">
                        <h2>Detailed Data Table</h2>
                        {html_data}
                    </div>
                </div>
            </body>
            </html>
            """
            
            st.download_button(
                label="🌐 Download HTML",
                data=html_full,
                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
    
    if "Image" in save_format:
        with export_cols[save_format.index("Image")]:
            st.info("📸 Screenshot feature: Use your browser's screenshot tool or Streamlit's built-in export")
    
    # Display the data preview
    st.markdown("### 📊 Data Preview")
    st.dataframe(df_export, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div class="empty-state">
        👆 Add model IDs and click "Compare Models" to get started with exporting results
    </div>
    """, unsafe_allow_html=True)

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px; margin-top: 20px;">
    <p>Model Comparison Tool v1.0 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
