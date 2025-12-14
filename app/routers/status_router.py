# app/routers/status_router.py

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def cleanup_dashboard():
    """HTML dashboard to monitor cleanup status"""
    from app.tasks.background_worker import worker
    import datetime

    status = worker.get_status()
    now = datetime.datetime.now()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Context Cleanup Dashboard</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            .status {{ margin: 20px 0; padding: 15px; border-radius: 5px; background: #f8f9fa; }}
            .running {{ border-left: 5px solid #4CAF50; }}
            .stopped {{ border-left: 5px solid #f44336; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .stat-card {{ background: #e8f5e9; padding: 15px; border-radius: 5px; text-align: center; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #2e7d32; }}
            .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
            .btn {{ display: inline-block; padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
            .btn:hover {{ background: #45a049; }}
            .btn-danger {{ background: #f44336; }}
            .btn-danger:hover {{ background: #d32f2f; }}
            .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🗑️ Context Cleanup Dashboard</h1>

            <div class="status {'running' if status['running'] else 'stopped'}">
                <h3>Worker Status: {'🟢 RUNNING' if status['running'] else '🔴 STOPPED'}</h3>
                <p>Last Updated: {now.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{status['run_count']}</div>
                    <div class="stat-label">Total Runs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{status['total_deleted']}</div>
                    <div class="stat-label">Total Deleted</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{status['hours_until_next'] or 'N/A'}</div>
                    <div class="stat-label">Hours Until Next Run</div>
                </div>
            </div>

            <h3>Schedule Information</h3>
            <table border="1" cellpadding="10" cellspacing="0" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Last Run</td>
                    <td><strong>{status['last_run']}</strong></td>
                </tr>
                <tr>
                    <td>Next Run</td>
                    <td><strong>{status['next_run']}</strong></td>
                </tr>
                <tr>
                    <td>Task Active</td>
                    <td>{'✅ Yes' if status['task_active'] else '❌ No'}</td>
                </tr>
                <tr>
                    <td>Server Time</td>
                    <td>{now.strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>

            <h3>Actions</h3>
            <a class="btn" href="/api/admin/cleanup/run-now" onclick="return confirm('Run cleanup now?')">▶️ Run Cleanup Now</a>
            <a class="btn" href="/api/admin/cleanup/status">📊 Refresh Status</a>
            <a class="btn" href="/api/admin/cleanup/stats">📈 Detailed Stats</a>

            <div class="timestamp">
                Auto-refresh every 30 seconds • 
                <a href="/api/admin/cleanup/dashboard">Refresh Now</a>
            </div>
        </div>

        <script>
            // Auto-refresh every 30 seconds
            setTimeout(function() {{
                location.reload();
            }}, 30000);

            // Handle manual cleanup button
            document.querySelector('a[href="/api/admin/cleanup/run-now"]').addEventListener('click', function(e) {{
                if(!confirm('Are you sure you want to run cleanup now?')) {{
                    e.preventDefault();
                    return;
                }}

                // Make API call
                fetch('/api/admin/cleanup/run-now', {{ method: 'POST' }})
                    .then(response => response.json())
                    .then(data => {{
                        alert(data.message || 'Cleanup completed!');
                        location.reload();
                    }})
                    .catch(error => {{
                        alert('Error: ' + error);
                    }});

                e.preventDefault();
            }});
        </script>
    </body>
    </html>
    """
    return html