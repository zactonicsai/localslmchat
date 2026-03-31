"""Temporal worker with upload + query task queues.

Best practices:
- Two workers on separate task queues (upload vs query) for independent scaling
- Query worker limited to max_concurrent_activities=1 (LLM is GPU-bound)
- Graceful shutdown via asyncio signal handling
- Connection params injected via env vars
"""

import asyncio
import logging
import os
import signal

from temporalio.client import Client
from temporalio.worker import Worker

from workflows import DocumentUploadWorkflow, QueryWorkflow
from activities import (
    extract_text_activity,
    chunk_text_activity,
    embed_and_store_activity,
    execute_query_activity,
)
from shared.models import TASK_QUEUE, QUERY_TASK_QUEUE

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("temporal-worker")


async def main():
    """Start both upload and query workers."""
    temporal_address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")

    logger.info(f"Connecting to Temporal at {temporal_address} (namespace={namespace})")
    client = await Client.connect(temporal_address, namespace=namespace)
    logger.info(
        f"Connected. Upload queue='{TASK_QUEUE}', Query queue='{QUERY_TASK_QUEUE}'"
    )

    # Upload worker: handles extract -> chunk -> embed pipeline
    upload_worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DocumentUploadWorkflow],
        activities=[
            extract_text_activity,
            chunk_text_activity,
            embed_and_store_activity,
        ],
    )

    # Query worker: handles RAG queries (limited concurrency — LLM is GPU-bound)
    query_worker = Worker(
        client,
        task_queue=QUERY_TASK_QUEUE,
        workflows=[QueryWorkflow],
        activities=[execute_query_activity],
        max_concurrent_activities=1,
    )

    # Graceful shutdown on SIGINT/SIGTERM
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received, draining workers...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Run both workers concurrently; cancel on shutdown signal
    worker_tasks = [
        asyncio.create_task(upload_worker.run(), name="upload-worker"),
        asyncio.create_task(query_worker.run(), name="query-worker"),
    ]

    # Wait for shutdown signal OR a worker to crash
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    done, _ = await asyncio.wait(
        [*worker_tasks, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # If shutdown was signaled, cancel workers gracefully
    for task in worker_tasks:
        if not task.done():
            task.cancel()

    # Wait for workers to finish draining
    await asyncio.gather(*worker_tasks, return_exceptions=True)

    # Check if a worker crashed (not a clean shutdown)
    for task in done:
        if task != shutdown_task and task.exception():
            logger.error(f"Worker crashed: {task.get_name()}", exc_info=task.exception())
            raise task.exception()

    logger.info("Workers shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
