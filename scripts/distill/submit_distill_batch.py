"""
OpenAI Batch API 제출

prepare_distill_batch.py 가 만든 jsonl을 업로드하고 batch를 생성한다.
batch_id를 stdout과 metadata 파일에 기록한다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _lib import PROJECT_ROOT, ensure_dotenv, get_openai_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="jsonl batch input file")
    parser.add_argument("--meta-file", required=True, help="저장될 batch metadata 경로")
    parser.add_argument("--description", default="distill batch")
    args = parser.parse_args()

    ensure_dotenv()
    client = get_openai_client()

    in_path = PROJECT_ROOT / args.input_file
    meta_path = PROJECT_ROOT / args.meta_file
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[upload] {in_path}  ({in_path.stat().st_size:,} bytes)")
    file_obj = client.files.create(file=open(in_path, "rb"), purpose="batch")
    print(f"[upload] file_id={file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": args.description},
    )
    print(f"[batch ] batch_id={batch.id}  status={batch.status}")

    meta = {
        "input_file": str(in_path),
        "input_file_id": file_obj.id,
        "batch_id": batch.id,
        "status": batch.status,
        "description": args.description,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] meta written to {meta_path}")
    print(f"[hint] check status: openai api batches.retrieve -i {batch.id}")


if __name__ == "__main__":
    main()
