"""Temporal worker entrypoint."""
import asyncio
import os
from temporalio.client import Client
from temporalio.worker import Worker
from workflows import DocumentUploadWorkflow
from activities import extract_text_activity, chunk_text_activity, embed_and_store_activity
from shared.models import TASK_QUEUE

TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")

async def main():
    print(f"[WORKER] Connecting to Temporal at {TEMPORAL_ADDRESS}")
    client = await Client.connect(TEMPORAL_ADDRESS)
    print(f"[WORKER] Starting worker on queue '{TASK_QUEUE}'")
    worker = Worker(
        client, task_queue=TASK_QUEUE,
        workflows=[DocumentUploadWorkflow],
        activities=[extract_text_activity, chunk_text_activity, embed_and_store_activity],
    )
    print("[WORKER] Running. Waiting for tasks...")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
