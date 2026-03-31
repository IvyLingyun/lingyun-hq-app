"""
Lingyun Phone E-Commerce: Hyper-Intelligence Command Center
FastAPI Backend serving the React frontend with real-time Databricks data.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

# ─── Configuration ───────────────────────────────────────────────────────────
CATALOG = os.getenv("CATALOG", "lingyun_demo")
SCHEMA = os.getenv("SCHEMA", "default")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "4b9b953939869799")
SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

logger = logging.getLogger("lingyun-backend")

# ─── Pydantic Models ────────────────────────────────────────────────────────
class OrderCreate(BaseModel):
    customer_name: str
    customer_email: str
    phone_model: str
    variant: str
    quantity: int = 1
    payment_method: str = "Credit Card"
    channel: str = "Online"
    region: str = "Asia-Pacific"
    city: str = ""
    country: str = ""
    shipping_address: str = ""

class TicketCreate(BaseModel):
    order_id: str
    customer_name: str
    phone_model: str
    issue_category: str
    issue_description: str
    priority: str = "MEDIUM"

class ActionUpdate(BaseModel):
    action_id: str
    status: str
    resolution_notes: Optional[str] = None

class GenieQuery(BaseModel):
    question: str

class ChatMessage(BaseModel):
    message: str
    customer_id: Optional[str] = None

# ─── App Lifecycle ───────────────────────────────────────────────────────────
w = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global w
    w = WorkspaceClient()
    logger.info("Databricks WorkspaceClient initialized")
    yield

app = FastAPI(
    title="Lingyun Hyper-Intelligence Command Center",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SQL Helper ──────────────────────────────────────────────────────────────
def execute_sql(query: str, params: dict = None) -> List[Dict[str, Any]]:
    """Execute SQL against the Databricks SQL Warehouse and return results."""
    try:
        if params:
            for key, value in params.items():
                if isinstance(value, str):
                    value = value.replace("'", "''")
                query = query.replace(f":{key}", f"'{value}'" if isinstance(value, str) else str(value))

        response = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=query,
            catalog=CATALOG,
            schema=SCHEMA,
            wait_timeout="50s",
        )

        # Handle PENDING state with polling
        if response.status.state == StatementState.PENDING:
            stmt_id = response.statement_id
            for _ in range(30):
                time.sleep(2)
                response = w.statement_execution.get_statement(stmt_id)
                if response.status.state == StatementState.SUCCEEDED:
                    break
                elif response.status.state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
                    raise HTTPException(status_code=500, detail=f"SQL failed: {response.status.error}")

        if response.status.state != StatementState.SUCCEEDED:
            raise HTTPException(status_code=500, detail=f"SQL failed: {response.status.error}")

        if not response.result or not response.result.data_array:
            return []

        columns = [col.name for col in response.manifest.schema.columns]
        return [dict(zip(columns, row)) for row in response.result.data_array]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── LLM Helper ─────────────────────────────────────────────────────────────
def _get_auth_token() -> str:
    """Get a valid auth token from the WorkspaceClient."""
    # Try direct token first
    if w.config.token:
        return w.config.token
    # Try authenticate() which works for OAuth/SP auth in Databricks Apps
    try:
        auth_func = w.config.authenticate
        if callable(auth_func):
            headers = auth_func()
            if isinstance(headers, dict):
                return headers.get("Authorization", "").replace("Bearer ", "")
    except Exception:
        pass
    # Fallback: use DATABRICKS_TOKEN env var
    return os.environ.get("DATABRICKS_TOKEN", "")

def call_llm(system_prompt: str, user_message: str) -> str:
    """Call Foundation Model API via REST (bypasses SDK serialization issues)."""
    try:
        import urllib.request
        import ssl
        host = w.config.host
        if host and not host.startswith("http"):
            host = f"https://{host}"
        host = host.rstrip("/")
        token = _get_auth_token()

        payload = json.dumps({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
        }).encode()

        req = urllib.request.Request(
            f"{host}/serving-endpoints/{SERVING_ENDPOINT}/invocations",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            result = json.loads(resp.read())

        choices = result.get("choices", [])
        if not choices:
            return "No response from AI model."
        msg = choices[0].get("message", {})
        return msg.get("content", "") if isinstance(msg, dict) else str(msg)
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        return f"AI service temporarily unavailable: {str(e)}"

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 ENDPOINTS: Global Executive Insights
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/metrics/hero")
def get_hero_metrics():
    """Hero KPIs: GMV, Order Count, Avg Sentiment, Active Tickets."""
    gmv_data = execute_sql("""
        SELECT
            COALESCE(SUM(total_amount), 0) AS total_gmv,
            COUNT(*) AS total_orders,
            COALESCE(AVG(total_amount), 0) AS avg_order_value,
            COUNT(DISTINCT customer_id) AS unique_customers
        FROM ops_orders
    """)

    sentiment_data = execute_sql("""
        SELECT
            COALESCE(AVG(sentiment_score), 0) AS avg_sentiment,
            COUNT(*) AS total_tickets,
            COUNT(CASE WHEN status != 'RESOLVED' THEN 1 END) AS open_tickets
        FROM ops_tickets
    """)

    # Sparkline data (last 7 days)
    sparkline = execute_sql("""
        SELECT
            DATE(placed_at) AS date,
            SUM(total_amount) AS daily_gmv
        FROM ops_orders
        WHERE placed_at >= CURRENT_DATE() - INTERVAL 7 DAYS
        GROUP BY DATE(placed_at)
        ORDER BY date
    """)

    return {
        "gmv": gmv_data[0] if gmv_data else {},
        "sentiment": sentiment_data[0] if sentiment_data else {},
        "sparkline": sparkline,
    }

@app.get("/api/metrics/regional")
def get_regional_metrics():
    """Regional sales heatmap data."""
    return execute_sql("""
        SELECT
            region,
            country,
            COUNT(*) AS orders,
            SUM(total_amount) AS revenue,
            AVG(total_amount) AS avg_order_value,
            COUNT(DISTINCT customer_id) AS customers
        FROM ops_orders
        GROUP BY region, country
        ORDER BY revenue DESC
    """)

@app.get("/api/metrics/funnel")
def get_order_funnel():
    """Sankey diagram data: Order lifecycle flow."""
    return execute_sql("""
        SELECT
            order_status,
            COUNT(*) AS count,
            SUM(total_amount) AS value
        FROM ops_orders
        GROUP BY order_status
    """)

@app.get("/api/metrics/tickets-by-status")
def get_ticket_funnel():
    """Ticket lifecycle for Sankey."""
    orders_with_tickets = execute_sql("""
        SELECT
            'DELIVERED' AS source,
            'TICKET_RAISED' AS target,
            COUNT(DISTINCT t.order_id) AS count
        FROM ops_tickets t
        JOIN ops_orders o ON t.order_id = o.order_id
        WHERE o.order_status = 'DELIVERED'
        UNION ALL
        SELECT
            'DELIVERED' AS source,
            'NO_ISSUE' AS target,
            COUNT(*) - (SELECT COUNT(DISTINCT order_id) FROM ops_tickets) AS count
        FROM ops_orders
        WHERE order_status = 'DELIVERED'
    """)
    return orders_with_tickets

@app.get("/api/metrics/model-comparison")
def get_model_comparison():
    """Radar chart: Compare phone models on Quality, Price, CSAT."""
    return execute_sql("""
        SELECT
            o.phone_model,
            AVG(o.total_amount) AS avg_price,
            COUNT(DISTINCT o.order_id) AS sales_volume,
            COALESCE(AVG(t.sentiment_score), 0) AS avg_csat,
            COUNT(DISTINCT t.ticket_id) AS ticket_count,
            ROUND(COUNT(DISTINCT t.ticket_id) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS defect_rate,
            100 - ROUND(COUNT(DISTINCT t.ticket_id) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS quality_score
        FROM ops_orders o
        LEFT JOIN ops_tickets t ON o.order_id = t.order_id
        GROUP BY o.phone_model
    """)

@app.get("/api/metrics/sentiment-trends")
def get_sentiment_trends():
    """Word cloud / sentiment trend data."""
    categories = execute_sql("""
        SELECT
            issue_category,
            COUNT(*) AS count,
            AVG(sentiment_score) AS avg_sentiment
        FROM ops_tickets
        GROUP BY issue_category
        ORDER BY count DESC
    """)

    models = execute_sql("""
        SELECT
            phone_model,
            DATE(created_at) AS date,
            COUNT(*) AS tickets,
            AVG(sentiment_score) AS sentiment
        FROM ops_tickets
        GROUP BY phone_model, DATE(created_at)
        ORDER BY date
    """)

    return {"categories": categories, "trends": models}

@app.get("/api/metrics/by-channel")
def get_channel_metrics():
    """Sales breakdown by channel."""
    return execute_sql("""
        SELECT
            channel,
            COUNT(*) AS orders,
            SUM(total_amount) AS revenue,
            AVG(total_amount) AS avg_order_value
        FROM ops_orders
        GROUP BY channel
        ORDER BY revenue DESC
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 ENDPOINTS: Staff AI Decision Assistant
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/actions")
def get_action_items(status: Optional[str] = None):
    """Get all action items, optionally filtered by status."""
    where = f"WHERE status = '{status}'" if status else ""
    return execute_sql(f"""
        SELECT * FROM ops_action_items
        {where}
        ORDER BY
            CASE priority WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END,
            created_at DESC
    """)

@app.get("/api/actions/{action_id}/context")
def get_action_context(action_id: str):
    """Get full context for an action item including AI intelligence."""
    # Get action details
    action = execute_sql(f"""
        SELECT * FROM ops_action_items WHERE action_id = '{action_id.replace("'", "''")}'
    """)
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")

    action_item = action[0]
    customer_id = action_item.get("customer_id", "")

    # Get customer history
    customer_orders = execute_sql(f"""
        SELECT order_id, phone_model, total_amount, order_status, placed_at
        FROM ops_orders
        WHERE customer_id = '{customer_id.replace("'", "''")}'
        ORDER BY placed_at DESC
    """)

    customer_tickets = execute_sql(f"""
        SELECT ticket_id, issue_category, issue_description, priority, status, created_at
        FROM ops_tickets
        WHERE customer_id = '{customer_id.replace("'", "''")}'
        ORDER BY created_at DESC
    """)

    # Generate AI recommendation — truncate context to avoid 400 errors
    context_summary = (
        f"Customer: {action_item.get('customer_name')} (LTV: ${action_item.get('customer_ltv')})\n"
        f"Phone: {action_item.get('phone_model')}\n"
        f"Action Type: {action_item.get('action_type')}\n"
        f"Priority: {action_item.get('priority')}\n"
        f"Description: {action_item.get('action_description', '')[:200]}\n"
        f"Trigger: {action_item.get('trigger_reason', '')[:200]}\n"
        f"Orders: {len(customer_orders)} total\n"
        f"Tickets: {len(customer_tickets)} total\n"
    )
    if customer_tickets:
        t = customer_tickets[0]
        context_summary += f"Latest ticket: {t.get('issue_category')} - {t.get('issue_description', '')[:150]}\n"

    ai_recommendation = call_llm(
        system_prompt="You are a senior customer success AI for Lingyun Phones. Given a VIP customer situation, provide: 1) Risk assessment 2) Recommended action 3) Relationship strategy 4) Estimated cost. Be concise, use bullet points.",
        user_message=f"Analyze and recommend:\n{context_summary}"
    )

    return {
        "action": action_item,
        "customer_orders": customer_orders,
        "customer_tickets": customer_tickets,
        "ai_recommendation": ai_recommendation,
    }

@app.put("/api/actions/{action_id}")
def update_action(action_id: str, update: ActionUpdate):
    """Update action item status."""
    completed = ", completed_at = CURRENT_TIMESTAMP()" if update.status == "COMPLETED" else ""
    notes = f", ai_recommendation = CONCAT(COALESCE(ai_recommendation, ''), '\n[Staff Note]: {update.resolution_notes.replace(chr(39), chr(39)+chr(39))}')" if update.resolution_notes else ""

    execute_sql(f"""
        UPDATE ops_action_items
        SET status = '{update.status}',
            updated_at = CURRENT_TIMESTAMP()
            {completed}
            {notes}
        WHERE action_id = '{action_id.replace("'", "''")}'
    """)
    return {"status": "updated", "action_id": action_id}

@app.post("/api/genie/query")
def genie_query(query: GenieQuery):
    """Genie-style natural language query interface."""
    # Use LLM to generate SQL from natural language
    schema_context = """
    Tables in lingyun_demo.default:
    - ops_orders: order_id, customer_id, customer_name, customer_ltv, phone_model, variant, quantity, unit_price, total_amount, discount_amount, payment_method, order_status, channel, region, city, country, placed_at, shipped_at, delivered_at
    - ops_tickets: ticket_id, order_id, customer_id, customer_name, customer_ltv, phone_model, issue_category, issue_description, priority, status, sentiment_score, sentiment_label, created_at, resolved_at
    - ops_action_items: action_id, ticket_id, order_id, customer_id, customer_name, customer_ltv, phone_model, action_type, action_description, priority, status, ai_recommendation, created_at

    Phone models: Lingyun X1 (entry $499), Lingyun Pro (mid $899), Lingyun Ultra (premium $1299-$1599)
    Regions: Asia-Pacific, Europe-West, Europe-East, North America, Latin America, Middle East, Africa
    """

    sql_response = call_llm(
        system_prompt=f"""You are a SQL expert for a phone e-commerce database.
        Generate a single Databricks SQL query to answer the user's question.
        {schema_context}
        IMPORTANT: Return ONLY the SQL query, no explanation, no markdown fences.""",
        user_message=query.question,
    )

    # Clean SQL
    sql = sql_response.strip().strip('`').replace('```sql', '').replace('```', '').strip()

    try:
        results = execute_sql(sql)
        # Generate natural language answer with truncated results
        results_text = json.dumps(results[:10], default=str)
        if len(results_text) > 1500:
            results_text = results_text[:1500] + "..."
        try:
            answer = call_llm(
                system_prompt="You are a business analyst. Answer the question based on the query results. Be specific with numbers. Keep it under 3 sentences.",
                user_message=f"Question: {query.question}\nResults: {results_text}",
            )
        except Exception as llm_err:
            # If LLM fails, generate a simple answer from the data
            logger.error(f"LLM answer error: {llm_err}")
            if results:
                answer = f"Query returned {len(results)} row(s). " + ", ".join(
                    f"{k}: {v}" for k, v in list(results[0].items())[:5]
                )
            else:
                answer = "Query returned no results."
        return {"question": query.question, "sql": sql, "results": results, "answer": answer}
    except Exception as e:
        return {"question": query.question, "sql": sql, "results": [], "answer": f"Query error: {str(e)}"}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 ENDPOINTS: Consumer Portal
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/products")
def get_products():
    """Get product catalog for e-shop."""
    return [
        {
            "id": "lingyun-x1",
            "name": "Lingyun X1",
            "tagline": "Essential Excellence",
            "price": 499.99,
            "variants": ["128GB Black", "128GB White", "256GB Black"],
            "features": ["6.1\" OLED Display", "48MP Camera", "5000mAh Battery", "5G Ready"],
            "image": "/static/images/x1.png",
            "rating": 4.3,
        },
        {
            "id": "lingyun-pro",
            "name": "Lingyun Pro",
            "tagline": "Professional Power",
            "price": 899.99,
            "variants": ["256GB Silver", "256GB Blue", "512GB Emerald"],
            "features": ["6.5\" AMOLED 120Hz", "108MP Triple Camera", "4800mAh + 65W Charge", "AI Photography"],
            "image": "/static/images/pro.png",
            "rating": 4.6,
        },
        {
            "id": "lingyun-ultra",
            "name": "Lingyun Ultra",
            "tagline": "Ultimate Innovation",
            "price": 1299.99,
            "variants": ["512GB Midnight", "512GB Pearl", "1TB Gold"],
            "features": ["6.8\" LTPO AMOLED 144Hz", "200MP Quad Camera + Periscope", "5500mAh + 120W + Wireless", "Lingyun AI Engine 3.0"],
            "image": "/static/images/ultra.png",
            "rating": 4.8,
        },
    ]

@app.post("/api/orders")
def create_order(order: OrderCreate):
    """Place a new order (writes to Lakebase)."""
    price_map = {"Lingyun X1": 499.99, "Lingyun Pro": 899.99, "Lingyun Ultra": 1299.99}
    if "1TB" in order.variant:
        price_map["Lingyun Ultra"] = 1599.99
    if "512GB" in order.variant and order.phone_model == "Lingyun Pro":
        price_map["Lingyun Pro"] = 999.99

    unit_price = price_map.get(order.phone_model, 499.99)
    total = unit_price * order.quantity
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    customer_id = f"CUST-WEB-{datetime.now().strftime('%H%M%S')}"

    execute_sql(f"""
        INSERT INTO ops_orders (
            order_id, customer_id, customer_name, customer_email, customer_ltv,
            phone_model, variant, quantity, unit_price, total_amount,
            payment_method, order_status, channel, region, city, country, shipping_address,
            placed_at, updated_at
        ) VALUES (
            '{order_id}', '{customer_id}', '{order.customer_name.replace("'", "''")}',
            '{order.customer_email.replace("'", "''")}', {total},
            '{order.phone_model}', '{order.variant}', {order.quantity}, {unit_price}, {total},
            '{order.payment_method}', 'PLACED', '{order.channel}', '{order.region}',
            '{order.city.replace("'", "''")}', '{order.country.replace("'", "''")}',
            '{order.shipping_address.replace("'", "''")}',
            CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()
        )
    """)

    return {"order_id": order_id, "total": total, "status": "PLACED"}

@app.post("/api/tickets")
def create_ticket(ticket: TicketCreate):
    """Submit a support ticket (writes to Lakebase)."""
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    customer_id = f"CUST-WEB-{datetime.now().strftime('%H%M%S')}"

    execute_sql(f"""
        INSERT INTO ops_tickets (
            ticket_id, order_id, customer_id, customer_name, customer_ltv,
            phone_model, issue_category, issue_description, priority, status,
            created_at, updated_at
        ) VALUES (
            '{ticket_id}', '{ticket.order_id.replace("'", "''")}', '{customer_id}',
            '{ticket.customer_name.replace("'", "''")}', 0,
            '{ticket.phone_model}', '{ticket.issue_category}',
            '{ticket.issue_description.replace("'", "''")}', '{ticket.priority}', 'OPEN',
            CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()
        )
    """)

    return {"ticket_id": ticket_id, "status": "OPEN"}

@app.get("/api/orders/track/{order_id}")
def track_order(order_id: str):
    """Track an order by ID."""
    results = execute_sql(f"""
        SELECT order_id, customer_name, phone_model, variant, total_amount,
               order_status, placed_at, shipped_at, delivered_at
        FROM ops_orders
        WHERE order_id = '{order_id.replace("'", "''")}'
    """)
    if not results:
        raise HTTPException(status_code=404, detail="Order not found")
    return results[0]

@app.post("/api/chat")
def ai_concierge(chat: ChatMessage):
    """AI Concierge chatbot for consumer portal."""
    # Get relevant context if customer_id provided
    context = ""
    if chat.customer_id:
        orders = execute_sql(f"""
            SELECT order_id, phone_model, order_status, placed_at
            FROM ops_orders WHERE customer_id = '{chat.customer_id.replace("'", "''")}'
            ORDER BY placed_at DESC LIMIT 5
        """)
        context = f"\nCustomer's recent orders: {json.dumps(orders, default=str)}"

    response = call_llm(
        system_prompt=f"""You are the Lingyun AI Concierge — a friendly, helpful assistant for Lingyun Phone customers.
        You help with: order tracking, technical support, product recommendations, and general inquiries.
        Product lineup: Lingyun X1 ($499, entry), Lingyun Pro ($899, mid-range), Lingyun Ultra ($1299+, premium).
        Be warm, professional, and concise. Use emojis sparingly.{context}""",
        user_message=chat.message,
    )

    return {"response": response}

# ─── Alerts Dashboard Endpoint ───────────────────────────────────────────────
@app.get("/api/alerts")
def get_alerts():
    """Get current alert statuses."""
    revenue = execute_sql("SELECT * FROM v_alert_revenue_drop")
    quality = execute_sql("SELECT * FROM v_alert_quality_crisis")
    vip_pending = execute_sql("SELECT * FROM v_vip_action_trigger")

    return {
        "revenue_alert": revenue[0] if revenue else {"revenue_alert_flag": "OK"},
        "quality_alerts": quality,
        "vip_pending_actions": vip_pending,
    }

# ─── Debug Endpoint ──────────────────────────────────────────────────────────
@app.get("/api/debug/auth")
def debug_auth():
    """Debug auth status."""
    host = w.config.host
    has_token = bool(w.config.token)
    try:
        auth_token = _get_auth_token()
        token_prefix = auth_token[:20] + "..." if auth_token else "NONE"
    except Exception as e:
        token_prefix = f"ERROR: {e}"
    return {
        "host": host,
        "has_direct_token": has_token,
        "auth_token_prefix": token_prefix,
        "env_databricks_host": os.environ.get("DATABRICKS_HOST", "not set"),
        "env_databricks_token": "set" if os.environ.get("DATABRICKS_TOKEN") else "not set",
        "is_app": bool(os.environ.get("DATABRICKS_APP_NAME")),
    }

# ─── Serve Static Frontend ──────────────────────────────────────────────────
import pathlib

# Try React build first, then fall back to static single-page HTML
frontend_build = pathlib.Path(__file__).parent.parent / "frontend" / "build"
static_dir = pathlib.Path(__file__).parent.parent / "static"

if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="frontend")
elif static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
