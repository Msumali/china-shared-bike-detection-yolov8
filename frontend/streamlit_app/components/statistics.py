import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..utils.api_client import APIClient
from ..utils.ui_helpers import show_error_message, show_success_message, show_info_message
from ..config.settings import SESSION_KEYS

def render_statistics_component():
    """Render statistics and analytics component."""
    
    st.header("📊 Detection Statistics & Analytics")
    
    # Time range selector
    time_range = render_time_range_selector()
    
    # Load and display statistics
    stats_data = load_statistics_data(time_range)
    
    if not stats_data:
        show_info_message("No statistics data available for the selected time range.")
        return
    
    # Main metrics
    render_main_metrics(stats_data)
    
    # Charts and visualizations
    render_detection_charts(stats_data)
    
    # Detailed analytics
    render_detailed_analytics(stats_data)
    
    # Export options
    render_export_options(stats_data)


def render_time_range_selector():
    """Render time range selector."""
    
    st.subheader("📅 Time Range")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        time_preset = st.selectbox(
            "Quick Select",
            ["Last Hour", "Last 24 Hours", "Last Week", "Last Month", "Custom Range"]
        )
    
    with col2:
        if time_preset == "Custom Range":
            date_range = st.date_input(
                "Select Date Range",
                value=(datetime.now() - timedelta(days=7), datetime.now()),
                max_value=datetime.now()
            )
        else:
            date_range = None
    
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Calculate actual time range
    if time_preset == "Last Hour":
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
    elif time_preset == "Last 24 Hours":
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
    elif time_preset == "Last Week":
        start_time = datetime.now() - timedelta(weeks=1)
        end_time = datetime.now()
    elif time_preset == "Last Month":
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now()
    elif time_preset == "Custom Range" and date_range:
        if len(date_range) == 2:
            start_time = datetime.combine(date_range[0], datetime.min.time())
            end_time = datetime.combine(date_range[1], datetime.max.time())
        else:
            start_time = datetime.now() - timedelta(days=7)
            end_time = datetime.now()
    else:
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
    
    return {"start_time": start_time, "end_time": end_time}


def load_statistics_data(time_range: Dict) -> Optional[Dict[str, Any]]:
    """Load statistics data from backend."""
    
    try:
        api_client = APIClient()
        response = api_client.get_statistics(
            start_time=time_range["start_time"].isoformat(),
            end_time=time_range["end_time"].isoformat()
        )
        
        if response.get("success"):
            return response.get("data", {})
        else:
            show_error_message("Failed to load statistics data.")
            return None
            
    except Exception as e:
        show_error_message(f"Error loading statistics: {str(e)}")
        return None


def render_main_metrics(stats_data: Dict[str, Any]):
    """Render main metrics cards."""
    
    st.subheader("🎯 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_detections = stats_data.get("total_detections", 0)
        st.metric(
            "Total Detections",
            f"{total_detections:,}",
            delta=stats_data.get("detections_change", 0)
        )
    
    with col2:
        total_videos = stats_data.get("total_videos_processed", 0)
        st.metric(
            "Videos Processed",
            f"{total_videos:,}",
            delta=stats_data.get("videos_change", 0)
        )
    
    with col3:
        avg_confidence = stats_data.get("average_confidence", 0)
        st.metric(
            "Avg Confidence",
            f"{avg_confidence:.2f}",
            delta=f"{stats_data.get('confidence_change', 0):.2f}"
        )
    
    with col4:
        processing_time = stats_data.get("total_processing_time", 0)
        st.metric(
            "Processing Time",
            f"{processing_time:.1f}s",
            delta=f"{stats_data.get('processing_time_change', 0):.1f}s"
        )
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        unique_bikes = stats_data.get("unique_bikes_detected", 0)
        st.metric("Unique Bikes", f"{unique_bikes:,}")
    
    with col2:
        fps_avg = stats_data.get("average_fps", 0)
        st.metric("Avg Processing FPS", f"{fps_avg:.1f}")
    
    with col3:
        accuracy = stats_data.get("detection_accuracy", 0)
        st.metric("Detection Accuracy", f"{accuracy:.1f}%")
    
    with col4:
        success_rate = stats_data.get("processing_success_rate", 0)
        st.metric("Success Rate", f"{success_rate:.1f}%")


def render_detection_charts(stats_data: Dict[str, Any]):
    """Render detection charts and visualizations."""
    
    st.subheader("📈 Detection Trends")
    
    # Time series data
    time_series = stats_data.get("time_series", [])
    
    if time_series:
        # Convert to DataFrame
        df = pd.DataFrame(time_series)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Detections over time
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                df, 
                x='timestamp', 
                y='detections',
                title='Detections Over Time',
                labels={'detections': 'Number of Detections', 'timestamp': 'Time'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                df, 
                x='timestamp', 
                y='confidence',
                title='Average Confidence Over Time',
                labels={'confidence': 'Confidence', 'timestamp': 'Time'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detection heatmap by hour
        if 'hour' in df.columns:
            st.subheader("🕐 Detection Patterns by Hour")
            
            hourly_data = df.groupby('hour')['detections'].sum().reset_index()
            
            fig = px.bar(
                hourly_data,
                x='hour',
                y='detections',
                title='Detections by Hour of Day',
                labels={'hour': 'Hour of Day', 'detections': 'Total Detections'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Confidence distribution
    confidence_data = stats_data.get("confidence_distribution", [])
    
    if confidence_data:
        st.subheader("📊 Confidence Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                x=confidence_data,
                nbins=20,
                title='Confidence Score Distribution',
                labels={'x': 'Confidence Score', 'y': 'Frequency'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                y=confidence_data,
                title='Confidence Score Statistics',
                labels={'y': 'Confidence Score'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def render_detailed_analytics(stats_data: Dict[str, Any]):
    """Render detailed analytics section."""
    
    st.subheader("🔍 Detailed Analytics")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Metrics**")
        
        performance_data = stats_data.get("performance_metrics", {})
        
        metrics_df = pd.DataFrame([
            {"Metric": "Average Processing Time per Frame", "Value": f"{performance_data.get('avg_frame_time', 0):.3f}s"},
            {"Metric": "Frames per Second", "Value": f"{performance_data.get('fps', 0):.1f}"},
            {"Metric": "Memory Usage", "Value": f"{performance_data.get('memory_usage', 0):.1f} MB"},
            {"Metric": "CPU Usage", "Value": f"{performance_data.get('cpu_usage', 0):.1f}%"},
            {"Metric": "GPU Usage", "Value": f"{performance_data.get('gpu_usage', 0):.1f}%"}
        ])
        
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.write("**Detection Quality**")
        
        quality_data = stats_data.get("quality_metrics", {})
        
        quality_df = pd.DataFrame([
            {"Metric": "High Confidence (>0.8)", "Value": f"{quality_data.get('high_confidence', 0)}"},
            {"Metric": "Medium Confidence (0.5-0.8)", "Value": f"{quality_data.get('medium_confidence', 0)}"},
            {"Metric": "Low Confidence (<0.5)", "Value": f"{quality_data.get('low_confidence', 0)}"},
            {"Metric": "False Positives", "Value": f"{quality_data.get('false_positives', 0)}"},
            {"Metric": "Missed Detections", "Value": f"{quality_data.get('missed_detections', 0)}"}
        ])
        
        st.dataframe(quality_df, hide_index=True, use_container_width=True)
    
    # Top performing videos
    top_videos = stats_data.get("top_videos", [])
    
    if top_videos:
        st.write("**Top Performing Videos**")
        
        videos_df = pd.DataFrame(top_videos)
        
        st.dataframe(
            videos_df[['filename', 'detections', 'avg_confidence', 'processing_time']],
            hide_index=True,
            use_container_width=True,
            column_config={
                'filename': 'Video File',
                'detections': st.column_config.NumberColumn('Detections', format='%d'),
                'avg_confidence': st.column_config.NumberColumn('Avg Confidence', format='%.2f'),
                'processing_time': st.column_config.NumberColumn('Processing Time (s)', format='%.2f')
            }
        )


def render_export_options(stats_data: Dict[str, Any]):
    """Render export options for statistics."""
    
    st.subheader("💾 Export Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV"):
            export_to_csv(stats_data)
    
    with col2:
        if st.button("📋 Export to JSON"):
            export_to_json(stats_data)
    
    with col3:
        if st.button("📈 Generate Report"):
            generate_report(stats_data)


def export_to_csv(stats_data: Dict[str, Any]):
    """Export statistics to CSV format."""
    
    try:
        # Prepare data for CSV export
        export_data = []
        
        # Add main metrics
        metrics = {
            'total_detections': stats_data.get('total_detections', 0),
            'total_videos_processed': stats_data.get('total_videos_processed', 0),
            'average_confidence': stats_data.get('average_confidence', 0),
            'total_processing_time': stats_data.get('total_processing_time', 0),
        }
        
        for key, value in metrics.items():
            export_data.append({'metric': key, 'value': value})
        
        # Add time series data if available
        time_series = stats_data.get('time_series', [])
        if time_series:
            for entry in time_series:
                export_data.append({
                    'metric': 'time_series',
                    'timestamp': entry.get('timestamp'),
                    'detections': entry.get('detections', 0),
                    'confidence': entry.get('confidence', 0)
                })
        
        df = pd.DataFrame(export_data)
        csv_data = df.to_csv(index=False)
        
        st.download_button(
            "📄 Download CSV",
            data=csv_data,
            file_name=f"detection_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        show_success_message("CSV export prepared for download!")
        
    except Exception as e:
        show_error_message(f"Error exporting to CSV: {str(e)}")


def export_to_json(stats_data: Dict[str, Any]):
    """Export statistics to JSON format."""
    
    try:
        import json
        
        # Add timestamp to the data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'statistics': stats_data
        }
        
        json_data = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            "📋 Download JSON",
            data=json_data,
            file_name=f"detection_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        show_success_message("JSON export prepared for download!")
        
    except Exception as e:
        show_error_message(f"Error exporting to JSON: {str(e)}")


def generate_report(stats_data: Dict[str, Any]):
    """Generate a detailed report."""
    
    try:
        report_content = f"""
# Bike Detection Statistics Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Detections:** {stats_data.get('total_detections', 0):,}
- **Videos Processed:** {stats_data.get('total_videos_processed', 0):,}
- **Average Confidence:** {stats_data.get('average_confidence', 0):.2f}
- **Total Processing Time:** {stats_data.get('total_processing_time', 0):.2f} seconds

## Performance Metrics
- **Average FPS:** {stats_data.get('average_fps', 0):.1f}
- **Detection Accuracy:** {stats_data.get('detection_accuracy', 0):.1f}%
- **Processing Success Rate:** {stats_data.get('processing_success_rate', 0):.1f}%

## Quality Analysis
- **High Confidence Detections (>0.8):** {stats_data.get('quality_metrics', {}).get('high_confidence', 0)}
- **Medium Confidence Detections (0.5-0.8):** {stats_data.get('quality_metrics', {}).get('medium_confidence', 0)}
- **Low Confidence Detections (<0.5):** {stats_data.get('quality_metrics', {}).get('low_confidence', 0)}

---
*This report was automatically generated by the Bike Detection System.*
        """
        
        st.download_button(
            "📈 Download Report",
            data=report_content,
            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
        
        show_success_message("Report generated for download!")
        
    except Exception as e:
        show_error_message(f"Error generating report: {str(e)}")


def render_real_time_stats():
    """Render real-time statistics section."""
    
    st.subheader("⚡ Real-time Statistics")
    
    # Create placeholders for real-time updates
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (5s intervals)")
    
    if auto_refresh:
        # This would typically be implemented with a background task
        # For now, we'll just show a placeholder
        st.info("Real-time monitoring active. Statistics will update automatically.")
        
        # You could implement WebSocket connection or periodic API calls here
        # For this example, we'll keep it simple
        time.sleep(1)  # Simulate delay
        
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Detections", "42", delta="5")
            
            with col2:
                st.metric("Active Streams", "3", delta="1")
            
            with col3:
                st.metric("System Load", "67%", delta="-3%")


def get_system_health():
    """Get system health metrics."""
    
    try:
        api_client = APIClient()
        response = api_client.get_system_health()
        
        if response.get("success"):
            return response.get("data", {})
        else:
            return {}
            
    except Exception as e:
        st.error(f"Error getting system health: {str(e)}")
        return {}


def render_system_health():
    """Render system health section."""
    
    st.subheader("🏥 System Health")
    
    health_data = get_system_health()
    
    if not health_data:
        show_info_message("System health data not available.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = health_data.get('cpu_usage', 0)
        st.metric(
            "CPU Usage",
            f"{cpu_usage:.1f}%",
            delta=None,
            delta_color="inverse" if cpu_usage > 80 else "normal"
        )
    
    with col2:
        memory_usage = health_data.get('memory_usage', 0)
        st.metric(
            "Memory Usage",
            f"{memory_usage:.1f}%",
            delta=None,
            delta_color="inverse" if memory_usage > 80 else "normal"
        )
    
    with col3:
        disk_usage = health_data.get('disk_usage', 0)
        st.metric(
            "Disk Usage",
            f"{disk_usage:.1f}%",
            delta=None,
            delta_color="inverse" if disk_usage > 90 else "normal"
        )
    
    with col4:
        gpu_usage = health_data.get('gpu_usage', 0)
        st.metric(
            "GPU Usage",
            f"{gpu_usage:.1f}%",
            delta=None
        )
    
    # Status indicators
    st.write("**Service Status**")
    
    services = health_data.get('services', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        api_status = services.get('api', 'unknown')
        status_color = "🟢" if api_status == 'healthy' else "🔴"
        st.write(f"{status_color} **API Service:** {api_status.title()}")
    
    with col2:
        detector_status = services.get('detector', 'unknown')
        status_color = "🟢" if detector_status == 'healthy' else "🔴"
        st.write(f"{status_color} **Detector Service:** {detector_status.title()}")
    
    with col3:
        storage_status = services.get('storage', 'unknown')
        status_color = "🟢" if storage_status == 'healthy' else "🔴"
        st.write(f"{status_color} **Storage Service:** {storage_status.title()}")