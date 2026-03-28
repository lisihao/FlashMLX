#!/usr/bin/env python3
"""
Adaptive Recent Window Benchmark - Workload-Driven Optimization

Validates the hypothesis: Different workloads have different redundancy,
requiring different recent_window_size for optimal memory-quality trade-off.

Test Workloads:
1. Long Document Summarization (High Redundancy) → Expect small window (128-256)
2. QA Dialogue (Low Redundancy) → Expect large window (512)
3. Coding Task (Medium Redundancy) → Expect medium window (256-384)

Comparison:
- Fixed Window (512) vs Adaptive Window (auto-detected)
- Memory usage, speed, quality

Usage:
    python benchmark_adaptive_window.py --calibration-dir /tmp/am_calibrations_ultra_dense
"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
import argparse
from datetime import datetime
import time

from mlx_lm.models.cache import KVCache
from mlx_lm.models.double_layer_cache import DoubleLayerKVCache

def log(msg, end='\n'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, end=end)

# ====================================================================
# Test Workload Corpus
# ====================================================================

# Workload 1: Long Document Summarization (High Redundancy)
# Repeated themes, redundant phrasing - EXTENDED VERSION
WORKLOAD_SUMMARIZATION = """
The annual technology conference took place in San Francisco last week. The technology conference brought together industry leaders from around the world. Technology companies showcased their latest innovations at the conference.

Artificial intelligence was a major theme at the technology conference. Many speakers discussed artificial intelligence applications. Artificial intelligence development continues to accelerate in various industries. Machine learning algorithms are transforming business operations. Machine learning enables predictive analytics and automation.

Cloud computing received significant attention at the conference. Cloud computing platforms are becoming increasingly important. Cloud computing enables businesses to scale operations efficiently. Cloud infrastructure reduces capital expenditure. Cloud services provide flexibility and reliability.

Cybersecurity concerns were raised multiple times during the conference. Cybersecurity threats continue to evolve. Companies must invest in cybersecurity measures to protect data. Cybersecurity professionals emphasized the importance of proactive defense. Cybersecurity training programs help employees recognize threats.

Data analytics was another recurring theme at the conference. Data analytics tools help businesses make informed decisions. Data analytics capabilities are essential for competitive advantage. Data visualization techniques make insights accessible. Data governance frameworks ensure data quality and compliance.

Digital transformation initiatives were widely discussed. Digital transformation requires organizational change. Digital transformation strategies align technology with business goals. Digital transformation projects often face cultural resistance. Digital transformation success depends on leadership commitment.

The Internet of Things received considerable attention. IoT devices are proliferating across industries. IoT networks generate massive amounts of data. IoT security remains a significant challenge. IoT applications range from smart homes to industrial automation.

Blockchain technology sparked interesting debates. Blockchain promises transparency and immutability. Blockchain applications extend beyond cryptocurrency. Blockchain adoption faces regulatory hurdles. Blockchain scalability issues require innovative solutions.

5G networks were highlighted as enabling technologies. 5G connectivity supports edge computing. 5G bandwidth enables real-time applications. 5G deployment is accelerating globally. 5G infrastructure investments are substantial.

The conference concluded with discussions about future technology trends. Technology innovation will continue to shape business operations. Future technology conferences will build upon these discussions. Technology leaders emphasized the need for ethical considerations. Technology education must prepare workers for rapid change.

Question: What were the main themes discussed at the technology conference?
Answer:"""

# Workload 2: QA Dialogue (Low Redundancy)
# Unique question requiring precise context - EXTENDED VERSION
WORKLOAD_QA = """
Dr. Sarah Chen founded the Quantum Dynamics Research Lab at Stanford University in 2019 with $5 million from the National Science Foundation. Her team aimed to develop room-temperature quantum computers.

The initial phase involved assembling a diverse team of experts. Chen recruited Dr. Robert Kim from MIT's quantum computing division, Dr. Elena Rodriguez from Caltech's physics department, and Dr. Yuki Tanaka from the University of Tokyo. They started with theoretical modeling of quantum coherence at elevated temperatures.

In 2020, they built their first prototype in the basement laboratory of the Gates Computer Science Building. Early tests were disappointing - quantum coherence lasted only 12 milliseconds at room temperature, far short of the minimum 1 second needed for practical computation.

The team persevered through 2021, making incremental improvements to the superconducting circuits. They experimented with different materials including niobium-titanium alloys and high-temperature superconductors. By December 2021, they had extended coherence to 3.7 seconds at 280 Kelvin using a novel error correction scheme.

The breakthrough came on July 15, 2022, at precisely 3:47 AM Pacific Time. The quantum processor achieved stable coherence at 294 Kelvin (room temperature, 21 Celsius) for an unprecedented 47 seconds. They ran 127 experiments that day with an 89% success rate across all qubit configurations.

The results were initially met with skepticism. Professor Marcus Blackwell from Oxford University published a harsh critique in Nature, questioning the calibration methods and suggesting systematic errors in the temperature measurements.

However, within six months, five independent teams successfully replicated the results. The Tokyo Institute of Technology team achieved 45 seconds at 293 Kelvin. Cambridge University reported 42 seconds at 295 Kelvin. ETH Zurich documented 49 seconds at 294 Kelvin. Tsinghua University in Beijing reached 44 seconds at 293 Kelvin. Caltech, using an entirely different qubit design, achieved 46 seconds at 294.5 Kelvin. The overall replication success rate across all teams was 84%.

Professor Blackwell later admitted his laboratory's thermometer had a calibration error of 12 Kelvin, which invalidated his criticism. He published a retraction and apology in the same journal.

Dr. Chen received the Nobel Prize in Physics in 2024, shared equally with Dr. Robert Kim and Dr. Elena Rodriguez. In her acceptance speech at Stockholm City Hall on December 10, 2024, she announced the decision to open-source the basic quantum processor designs under a Creative Commons license, enabling researchers worldwide to build upon their work.

The technology spread rapidly across the global scientific community. By January 2025, room-temperature quantum computers were deployed in major research centers across 50 countries spanning all continents except Antarctica. Applications expanded beyond the original drug discovery focus to include climate modeling, financial optimization, cryptography research, and materials science simulations.

Question: What was the exact time and temperature of the quantum breakthrough in 2022?
Answer:"""

# Workload 3: Coding Task (Medium Redundancy)
# Code structure, but with local dependencies - EXTENDED VERSION
WORKLOAD_CODING = """
class UserDataProcessor:
    def __init__(self, database_connection: DatabaseConnection):
        self.db = database_connection
        self.cache = {}
        self.validation_rules = self._init_validation_rules()

    def _init_validation_rules(self) -> dict:
        return {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'name': r'^[a-zA-Z\s]{2,50}$',
            'age': lambda x: 0 < x < 150,
            'phone': r'^\+?1?\d{9,15}$'
        }

    def get_user(self, user_id: int) -> User:
        if user_id in self.cache:
            return self.cache[user_id]

        user = self.db.query('SELECT * FROM users WHERE id = ?', [user_id])
        if user:
            self.cache[user_id] = user
        return user

    def validate_field(self, field_name: str, value: any) -> bool:
        if field_name not in self.validation_rules:
            return False

        rule = self.validation_rules[field_name]
        if callable(rule):
            return rule(value)
        else:
            import re
            return bool(re.match(rule, str(value)))

    def process_user_data(self, user_id: int, options: dict) -> dict:
        # Fetch user profile from database
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found in database")

        # Validate required fields
        required_fields = ['email', 'name', 'age']
        for field in required_fields:
            if field not in options:
                raise ValueError(f"Missing required field: {field}")

            if not self.validate_field(field, options[field]):
                raise ValueError(f"Invalid value for field: {field}")

        # Check for optional fields
        optional_fields = ['phone', 'address', 'company']
        for field in optional_fields:
            if field in options and not self.validate_field(field, options[field]):
                raise ValueError(f"Invalid value for optional field: {field}")

        # Process user data with transformations
        processed_data = {
            'user_id': user_id,
            'email': options['email'].lower().strip(),
            'name': options['name'].strip().title(),
            'age': int(options['age']),
            'verified': user.is_verified(),
            'created_at': user.created_at,
            'updated_at': datetime.now()
        }

        # Add optional fields if present
        if 'phone' in options:
            processed_data['phone'] = self._normalize_phone(options['phone'])
        if 'address' in options:
            processed_data['address'] = options['address']
        if 'company' in options:
            processed_data['company'] = options['company']

        # Update database with transaction
        try:
            self.db.begin_transaction()
            self.db.execute(
                'UPDATE users SET email=?, name=?, age=?, updated_at=? WHERE id=?',
                [processed_data['email'], processed_data['name'],
                 processed_data['age'], processed_data['updated_at'], user_id]
            )
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise RuntimeError(f"Database update failed: {e}")

        # Invalidate cache
        if user_id in self.cache:
            del self.cache[user_id]

        return processed_data

    def _normalize_phone(self, phone: str) -> str:
        # Remove all non-digit characters
        digits = ''.join(filter(str.isdigit, phone))

        # Format as international number
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+{digits}"
        else:
            return f"+{digits}"

Question: Write a function to validate user email format in the above code.
Answer:"""

# Workload 4: Agent Execution (Low Redundancy)
# Extended real agent debugging session - comprehensive investigation
WORKLOAD_AGENT = """
User: My API endpoint is returning 500 errors intermittently. Can you help debug this?

Agent: I'll help you debug the 500 errors. Let me start by gathering information about the issue.

Tool Call: search_logs(service="api-server", level="error", time_range="24h")
Result: Found 247 error entries. Most recent: "Database connection timeout after 30s" at 2024-03-15 14:23:17 UTC. Error spike started at 08:30 UTC with 3 errors/min, escalated to 18 errors/min by 14:00 UTC.

Agent: The logs show database connection timeouts with an escalating pattern. Let me check the database configuration and current connections.

Tool Call: read_file(path="/config/database.yml")
Result:
```yaml
production:
  adapter: postgresql
  pool: 10
  timeout: 30000
  host: db.example.com
  port: 5432
  database: production_db
  username: app_user
  # Connection pool settings
  pool_timeout: 5
  checkout_timeout: 5
  reaping_frequency: 10
```

Agent: The pool size is only 10 connections with a 30-second timeout. Let me check the current database connection usage and status.

Tool Call: execute_command(cmd="psql -h db.example.com -U app_user -d production_db -c 'SELECT count(*), state, wait_event_type FROM pg_stat_activity WHERE datname = \\'production_db\\' GROUP BY state, wait_event_type'")
Result:
| count | state  | wait_event_type |
|-------|--------|-----------------|
| 94    | active | Client          |
| 3     | idle   | NULL            |
| 2     | idle in transaction | Client |

Agent: Critical finding - you have 94 active connections, plus 5 idle/stuck connections. Pool size is only 10. This explains the timeouts. The application is creating connections outside the pool. Let me search for database connection patterns in the codebase.

Tool Call: grep_code(pattern="psycopg2.connect|DatabaseConnection|create_engine", directory="src/", recursive=true)
Result: Found 7 locations:
- src/api/handlers.py:15 - Direct psycopg2.connect() call
- src/api/handlers.py:87 - Direct psycopg2.connect() call
- src/background/worker.py:8 - Direct psycopg2.connect() call
- src/background/scheduler.py:12 - Direct psycopg2.connect() call
- src/database/pool.py:23 - Proper pool usage
- src/migrations/migrate.py:5 - Direct connection (OK for migrations)
- tests/test_db.py:10 - Test connection (OK for tests)

Agent: Found the root cause. Four production modules are creating direct connections bypassing the pool. Let me check the handler code first.

Tool Call: read_file(path="src/api/handlers.py", lines="1-100")
Result:
```python
import psycopg2
import json
from datetime import datetime

def process_request(request_data):
    # BUG: Direct connection bypassing pool
    conn = psycopg2.connect(
        host='db.example.com',
        database='production_db',
        user='app_user',
        password=os.getenv('DB_PASSWORD')
    )
    cursor = conn.cursor()

    # Extract user data
    user_id = request_data.get('user_id')
    action = request_data.get('action')

    # Query user info
    cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
    user = cursor.fetchone()

    if not user:
        return {'error': 'User not found'}, 404

    # Log the action
    cursor.execute(
        'INSERT INTO activity_log (user_id, action, timestamp) VALUES (%s, %s, %s)',
        (user_id, action, datetime.now())
    )
    conn.commit()

    # Missing conn.close()!
    return {'status': 'success'}, 200

def process_batch_request(batch_data):
    # BUG: Another direct connection!
    conn = psycopg2.connect(
        host='db.example.com',
        database='production_db',
        user='app_user',
        password=os.getenv('DB_PASSWORD')
    )
    cursor = conn.cursor()

    results = []
    for item in batch_data:
        cursor.execute(
            'INSERT INTO batch_jobs (data, status) VALUES (%s, %s) RETURNING id',
            (json.dumps(item), 'pending')
        )
        job_id = cursor.fetchone()[0]
        results.append(job_id)

    conn.commit()
    # Missing conn.close()!

    return {'job_ids': results}, 200
```

Agent: Both handlers create connections without closing them. Each request leaks a connection. Let me check the background worker too.

Tool Call: read_file(path="src/background/worker.py")
Result:
```python
import psycopg2
import time
import logging

def background_task():
    logging.info("Starting background worker")

    # BUG: Direct connection in infinite loop
    conn = psycopg2.connect(
        host='db.example.com',
        database='production_db',
        user='app_user',
        password=os.getenv('DB_PASSWORD')
    )

    while True:
        cursor = conn.cursor()
        cursor.execute('SELECT id, data FROM batch_jobs WHERE status = \\'pending\\' LIMIT 10')
        jobs = cursor.fetchall()

        for job_id, data in jobs:
            try:
                # Process job
                result = process_job(data)
                cursor.execute(
                    'UPDATE batch_jobs SET status = \\'completed\\', result = %s WHERE id = %s',
                    (result, job_id)
                )
                conn.commit()
            except Exception as e:
                logging.error(f"Job {job_id} failed: {e}")
                cursor.execute(
                    'UPDATE batch_jobs SET status = \\'failed\\' WHERE id = %s',
                    (job_id,)
                )
                conn.commit()

        time.sleep(5)
```

Agent: The worker never closes the connection and runs in an infinite loop. Let me check the scheduler too.

Tool Call: read_file(path="src/background/scheduler.py")
Result:
```python
import psycopg2
import schedule
import time

def cleanup_old_jobs():
    # BUG: Creates new connection on every run
    conn = psycopg2.connect(
        host='db.example.com',
        database='production_db',
        user='app_user',
        password=os.getenv('DB_PASSWORD')
    )
    cursor = conn.cursor()
    cursor.execute('DELETE FROM batch_jobs WHERE created_at < NOW() - INTERVAL \\'7 days\\'')
    conn.commit()
    # Missing conn.close()!

schedule.every().day.at("02:00").do(cleanup_old_jobs)

while True:
    schedule.run_pending()
    time.sleep(60)
```

Agent: The scheduler creates a new leaked connection every day at 2 AM. Now let me check when this code was introduced.

Tool Call: execute_command(cmd="git log --oneline --all --since='2024-03-01' -- src/api/handlers.py src/background/")
Result:
- a3f2b91 (2024-03-10) Add batch request handler for bulk operations
- f7e2c44 (2024-03-08) Add background scheduler for job cleanup
- 9d1a823 (2024-03-05) Implement background worker for async job processing
- 7e1d832 (2024-02-15) Initial API implementation

Agent: Multiple problematic commits. Let me check the deployment timeline and traffic patterns.

Tool Call: search_logs(service="deployment", pattern="deployed|rollback", time_range="15d")
Result:
- 2024-03-05 10:30 UTC - Deployed version v2.2.0 (commit: 9d1a823) - Background worker
- 2024-03-08 11:45 UTC - Deployed version v2.2.5 (commit: f7e2c44) - Scheduler added
- 2024-03-11 09:15 UTC - Deployed version v2.3.1 (commit: a3f2b91) - Batch handler
- 2024-03-15 08:30 UTC - Error spike began (no deployment)

Agent: Interesting timeline. The bad code was deployed March 11th but errors started March 15th. Let me check if there was a traffic increase.

Tool Call: execute_command(cmd="grep -A 5 '2024-03-15 08:' /var/log/nginx/access.log | wc -l")
Result: 18473 requests

Tool Call: execute_command(cmd="grep -A 5 '2024-03-14 08:' /var/log/nginx/access.log | wc -l")
Result: 3241 requests

Agent: Traffic increased 5.7x on March 15th! That explains the delayed error onset. Let me verify the connection pool implementation to ensure it's correct.

Tool Call: read_file(path="src/database/pool.py")
Result:
```python
from psycopg2.pool import ThreadedConnectionPool
import os

# Create connection pool
pool = ThreadedConnectionPool(
    minconn=5,
    maxconn=10,
    host='db.example.com',
    database='production_db',
    user='app_user',
    password=os.getenv('DB_PASSWORD'),
    port=5432
)

def get_connection():
    # Get a connection from the pool.
    return pool.getconn()

def release_connection(conn):
    # Return a connection to the pool.
    pool.putconn(conn)

class DatabaseContext:
    # Context manager for safe connection handling.
    def __enter__(self):
        self.conn = get_connection()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        release_connection(self.conn)
```

Agent: The pool implementation is correct and even provides a context manager. The problem is that it's not being used. Let me check how many worker processes are running.

Tool Call: execute_command(cmd="ps aux | grep 'background_task\\|scheduler' | grep -v grep")
Result:
- worker.py: 12 processes running
- scheduler.py: 1 process running

Agent: 12 worker processes! Each creates one leaked connection. Combined with leaked connections from API requests and the scheduler, this explains the 94 active connections.

Here's my complete analysis:

Root Cause Analysis:
1. **API handlers** (handlers.py): Two functions create direct connections on every request
   - process_request: Leaks 1 connection per request
   - process_batch_request: Leaks 1 connection per batch
   - With 18,473 requests in one hour = potential for massive leak

2. **Background workers** (worker.py): 12 processes × 1 connection each = 12 leaked connections
   - Each worker creates one connection and holds it forever

3. **Scheduler** (scheduler.py): Creates 1 leaked connection per day
   - Minor but accumulates over time

4. **Traffic spike**: 5.7x increase on March 15th triggered the crisis
   - Low traffic (3,241 req/hour) masked the leak before
   - High traffic (18,473 req/hour) exhausted connection pool

5. **Database limit**: PostgreSQL default max_connections = 100
   - 94 active + 5 idle/stuck = 99/100 slots used
   - New requests timeout waiting for available connections

Timeline:
- March 5: Background worker deployed (12 leaked connections)
- March 8: Scheduler deployed (1 more leaked connection)
- March 11: Batch handler deployed (per-request leaks begin)
- March 11-14: Low traffic, leaks accumulate slowly
- March 15 08:30: Traffic spike 5.7x → rapid connection exhaustion → 500 errors

Immediate Fix (Critical - Apply Now):
1. **Restart all services** to clear leaked connections
2. **Emergency patch** - Apply these fixes:

handlers.py:
```python
from database.pool import DatabaseContext

def process_request(request_data):
    with DatabaseContext() as conn:
        cursor = conn.cursor()
        user_id = request_data.get('user_id')
        action = request_data.get('action')

        cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
        user = cursor.fetchone()

        if not user:
            return {'error': 'User not found'}, 404

        cursor.execute(
            'INSERT INTO activity_log (user_id, action, timestamp) VALUES (%s, %s, %s)',
            (user_id, action, datetime.now())
        )
        conn.commit()

    return {'status': 'success'}, 200

def process_batch_request(batch_data):
    with DatabaseContext() as conn:
        cursor = conn.cursor()
        results = []
        for item in batch_data:
            cursor.execute(
                'INSERT INTO batch_jobs (data, status) VALUES (%s, %s) RETURNING id',
                (json.dumps(item), 'pending')
            )
            job_id = cursor.fetchone()[0]
            results.append(job_id)
        conn.commit()

    return {'job_ids': results}, 200
```

worker.py:
```python
from database.pool import get_connection, release_connection

def background_task():
    logging.info("Starting background worker")
    conn = get_connection()
    try:
        while True:
            cursor = conn.cursor()
            cursor.execute('SELECT id, data FROM batch_jobs WHERE status = \\'pending\\' LIMIT 10')
            jobs = cursor.fetchall()

            for job_id, data in jobs:
                try:
                    result = process_job(data)
                    cursor.execute(
                        'UPDATE batch_jobs SET status = \\'completed\\', result = %s WHERE id = %s',
                        (result, job_id)
                    )
                    conn.commit()
                except Exception as e:
                    logging.error(f"Job {job_id} failed: {e}")
                    cursor.execute(
                        'UPDATE batch_jobs SET status = \\'failed\\' WHERE id = %s',
                        (job_id,)
                    )
                    conn.commit()

            time.sleep(5)
    finally:
        release_connection(conn)
```

scheduler.py:
```python
from database.pool import DatabaseContext

def cleanup_old_jobs():
    with DatabaseContext() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM batch_jobs WHERE created_at < NOW() - INTERVAL \\'7 days\\'')
        conn.commit()
```

3. **Increase pool size** (config/database.yml):
```yaml
production:
  pool: 50  # Increased from 10
  timeout: 30000
  pool_timeout: 10  # Increased from 5
```

Long-term Improvements:
1. **Connection leak monitoring**: Add Prometheus metrics for pool usage
2. **Circuit breaker**: Implement timeout protection for database calls
3. **Linting rule**: Prevent direct psycopg2.connect() in src/ directory
4. **Load testing**: Simulate high traffic before production deployment
5. **Database limit increase**: Raise PostgreSQL max_connections to 200
6. **Connection pooler**: Consider PgBouncer for better connection management

Would you like me to create a pull request with these fixes?

Question: What was the root cause of the 500 errors?
Answer:"""

WORKLOAD_CORPUS = {
    'summarization': WORKLOAD_SUMMARIZATION,
    'qa': WORKLOAD_QA,
    'coding': WORKLOAD_CODING,
    'agent': WORKLOAD_AGENT
}

# ====================================================================
# Benchmark Function
# ====================================================================

def benchmark_workload(
    name: str,
    corpus: str,
    model,
    tokenizer,
    cache_factory,
    num_generate: int = 50
):
    """
    Benchmark a specific workload with given cache.

    Returns
    -------
    dict : Performance metrics
    """
    log(f"\n{'='*70}")
    log(f"Benchmarking: {name}")
    log(f"{'='*70}")

    # Tokenize
    tokens = tokenizer.encode(corpus)
    prompt_len = len(tokens)
    log(f"Prompt length: {prompt_len} tokens")

    # Create caches
    num_layers = len(model.model.layers)
    cache_list = cache_factory(num_layers)

    # Prefill
    log("Step 1: Prefill...")
    y = mx.array([tokens])
    mx.eval(y)
    mx.clear_cache()

    # Inject tokens for adaptive window analysis
    for cache in cache_list:
        if hasattr(cache, 'set_tokens_for_analysis'):
            cache.set_tokens_for_analysis(tokens)

    prefill_start = time.time()
    logits = model(y[:, :-1], cache=cache_list)
    mx.eval(logits)
    prefill_time = time.time() - prefill_start

    prefill_tps = prompt_len / prefill_time
    log(f"  Prefill: {prefill_tps:.2f} tokens/sec ({prefill_time:.3f}s)")

    # Decode
    log(f"Step 2: Generate {num_generate} tokens...")
    y = mx.array([[tokens[-1]]])

    generate_start = time.time()
    generated_tokens = []

    for i in range(num_generate):
        logits = model(y, cache=cache_list)
        y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)

        token_id = y[0, 0].item()
        generated_tokens.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

    generate_time = time.time() - generate_start
    generate_tps = len(generated_tokens) / generate_time

    log(f"  Generated {len(generated_tokens)} tokens")
    log(f"  TG speed: {generate_tps:.2f} tokens/sec ({generate_time:.3f}s)")

    # Memory usage
    total_memory = 0
    for cache in cache_list:
        if hasattr(cache, 'nbytes'):
            total_memory += cache.nbytes
        elif hasattr(cache, 'keys') and hasattr(cache, 'values'):
            total_memory += cache.keys.nbytes + cache.values.nbytes

    total_memory_mb = total_memory / (1024 ** 2)
    log(f"  Memory: {total_memory_mb:.2f} MB")

    # Output text
    output_text = tokenizer.decode(generated_tokens[:50])
    log(f"  Output: {output_text[:100]}...")

    # Stats
    compression_stats = None
    if hasattr(cache_list[0], 'get_stats'):
        compression_stats = cache_list[0].get_stats()
        log(f"  Compression stats:")
        log(f"    - compressions: {compression_stats['num_compressions']}")
        if compression_stats.get('adaptive_window_enabled'):
            log(f"    - adaptive window: ENABLED")
            log(f"    - detected redundancy: {compression_stats['detected_redundancy']:.2%}")
            log(f"    - configured window: {compression_stats['configured_window']}")
            log(f"    - initial window: {compression_stats['initial_window']}")
        else:
            log(f"    - adaptive window: DISABLED")
            log(f"    - configured window: {compression_stats.get('recent_window_size', 'N/A')}")

    return {
        'name': name,
        'prompt_tokens': prompt_len,
        'generated_tokens': len(generated_tokens),
        'prefill_tps': prefill_tps,
        'generate_tps': generate_tps,
        'memory_mb': total_memory_mb,
        'output_text': output_text,
        'compression_stats': compression_stats
    }

# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Adaptive Recent Window Benchmark')
    parser.add_argument('--model-path', default='/Volumes/toshiba/models/qwen3-8b-mlx',
                        help='Path to model')
    parser.add_argument('--calibration-dir', required=True,
                        help='Calibration directory')
    parser.add_argument('--num-generate', type=int, default=50,
                        help='Number of tokens to generate')
    parser.add_argument('--memory-budget', type=float, default=2.0,
                        help='Memory budget in MB per layer')
    args = parser.parse_args()

    log("=" * 70)
    log("🔬 Adaptive Recent Window Benchmark")
    log("=" * 70)
    log(f"Model: {args.model_path}")
    log(f"Calibration: {args.calibration_dir}")
    log(f"Memory Budget: {args.memory_budget} MB/layer")

    # Load model
    log("\nLoading model...")
    model, tokenizer = load(args.model_path)
    num_layers = len(model.model.layers)
    log(f"✓ Model loaded: {num_layers} layers")

    # Test each workload
    results = {}

    for workload_name, corpus in WORKLOAD_CORPUS.items():
        log(f"\n\n{'#'*70}")
        log(f"# Workload: {workload_name.upper()}")
        log(f"{'#'*70}")

        workload_results = []

        # Test 1: Baseline (Full KVCache)
        result_baseline = benchmark_workload(
            name=f"{workload_name} - Baseline",
            corpus=corpus,
            model=model,
            tokenizer=tokenizer,
            cache_factory=lambda n: [KVCache() for _ in range(n)],
            num_generate=args.num_generate
        )
        workload_results.append(result_baseline)

        # Test 2: Fixed Window (512)
        result_fixed = benchmark_workload(
            name=f"{workload_name} - Fixed Window (512)",
            corpus=corpus,
            model=model,
            tokenizer=tokenizer,
            cache_factory=lambda n: [
                DoubleLayerKVCache(
                    memory_budget_mb=args.memory_budget,
                    recent_window_size=512,
                    compression_ratio=1.5,
                    calibration_dir=args.calibration_dir,
                    layer_idx=i,
                    enable_compression=True,
                    enable_adaptive_window=False
                )
                for i in range(n)
            ],
            num_generate=args.num_generate
        )
        workload_results.append(result_fixed)

        # Test 3: Adaptive Window (auto-detected)
        result_adaptive = benchmark_workload(
            name=f"{workload_name} - Adaptive Window",
            corpus=corpus,
            model=model,
            tokenizer=tokenizer,
            cache_factory=lambda n: [
                DoubleLayerKVCache(
                    memory_budget_mb=args.memory_budget,
                    recent_window_size=512,  # Initial value
                    compression_ratio=1.5,
                    calibration_dir=args.calibration_dir,
                    layer_idx=i,
                    enable_compression=True,
                    enable_adaptive_window=True,
                    workload_hint=workload_name  # Explicit hint
                )
                for i in range(n)
            ],
            num_generate=args.num_generate
        )
        workload_results.append(result_adaptive)

        results[workload_name] = workload_results

    # Final Summary
    log("\n\n" + "=" * 70)
    log("📊 Summary: Fixed vs Adaptive Window")
    log("=" * 70)

    for workload_name, workload_results in results.items():
        log(f"\n{workload_name.upper()}:")
        baseline = workload_results[0]
        fixed = workload_results[1]
        adaptive = workload_results[2]

        log(f"  Baseline:  {baseline['memory_mb']:.2f} MB")
        log(f"  Fixed512:  {fixed['memory_mb']:.2f} MB ({fixed['memory_mb']/baseline['memory_mb']:.1%})")
        log(f"  Adaptive:  {adaptive['memory_mb']:.2f} MB ({adaptive['memory_mb']/baseline['memory_mb']:.1%})")

        if adaptive['compression_stats'] and adaptive['compression_stats'].get('adaptive_window_enabled'):
            detected = adaptive['compression_stats']['detected_redundancy']
            window = adaptive['compression_stats']['configured_window']
            log(f"  → Detected redundancy: {detected:.2%}, Window: {window}")

        # Memory improvement: Adaptive vs Fixed
        improvement = (fixed['memory_mb'] - adaptive['memory_mb']) / fixed['memory_mb']
        log(f"  → Adaptive improvement over Fixed: {improvement:+.1%}")

    log("\n" + "=" * 70)

if __name__ == '__main__':
    main()
