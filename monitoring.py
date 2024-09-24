import plotly.graph_objs as go
import psutil
import threading
import time

# Assuming clients are separate processes with known PIDs
client_pids = []  # This should be populated with client process PIDs
resource_data = {}

def monitor_client_resources():
    """Monitor CPU and memory usage for each client."""
    while True:
        for pid in client_pids:
            try:
                process = psutil.Process(pid)
                cpu_usage = process.cpu_percent(interval=1)
                memory_usage = process.memory_percent()
                if pid not in resource_data:
                    resource_data[pid] = {"cpu": [], "memory": []}
                resource_data[pid]["cpu"].append(cpu_usage)
                resource_data[pid]["memory"].append(memory_usage)
            except psutil.NoSuchProcess:
                pass
        time.sleep(1)

def create_live_plot():
    """Create a live Plotly graph for each client's resource usage."""
    fig = go.Figure()
    for pid in resource_data:
        fig.add_trace(go.Scatter(x=list(range(len(resource_data[pid]["cpu"]))), y=resource_data[pid]["cpu"], mode='lines', name=f'Client {pid} CPU'))
        fig.add_trace(go.Scatter(x=list(range(len(resource_data[pid]["memory"]))), y=resource_data[pid]["memory"], mode='lines', name=f'Client {pid} Memory'))

    fig.update_layout(
        title="Live Resource Usage per Client",
        xaxis_title="Time (s)",
        yaxis_title="Usage (%)",
        yaxis=dict(range=[0, 100]),
    )
    
    return fig
