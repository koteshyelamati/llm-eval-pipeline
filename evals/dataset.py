from dataclasses import dataclass
import json
import os
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


@dataclass
class EvalSample:
    id: str
    question: str
    ground_truth: str
    contexts: list[str]
    answer: str = ""


def _parse_samples(data: list[dict]) -> list[EvalSample]:
    return [
        EvalSample(
            id=item["id"],
            question=item["question"],
            ground_truth=item["ground_truth"],
            contexts=item["contexts"],
            answer=item.get("answer", ""),
        )
        for item in data
    ]


class DatasetLoader:
    def load_local(self, path: str) -> list[EvalSample]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Golden set not found at path: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _parse_samples(data)

    def load_s3(self, s3_path: str) -> list[EvalSample]:
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        client = boto3.client("s3")
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("NoSuchBucket", "NoSuchKey", "404"):
                raise FileNotFoundError(
                    f"S3 object not found — bucket: '{bucket}', key: '{key}'"
                ) from e
            raise
        data = json.loads(response["Body"].read().decode("utf-8"))
        return _parse_samples(data)

    def load(self, local_path: str, s3_path: str | None = None) -> list[EvalSample]:
        if s3_path:
            try:
                return self.load_s3(s3_path)
            except Exception:
                pass
        return self.load_local(local_path)
