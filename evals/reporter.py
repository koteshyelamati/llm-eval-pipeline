import json
import logging
import os
from datetime import datetime, timezone

import requests

from evals.metrics import MetricResult

logger = logging.getLogger(__name__)

_STATUS_PASS = "PASSED"
_STATUS_FAIL = "FAILED"


class EvalReporter:
    def format_markdown_table(
        self,
        results: dict[str, list[MetricResult]],
        overall_passed: bool,
    ) -> str:
        banner_icon = "✅" if overall_passed else "❌"
        banner_label = _STATUS_PASS if overall_passed else _STATUS_FAIL
        lines = [
            f"## RAG Evaluation — {banner_icon} {banner_label}",
            "",
            f"_Run timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
            "| Sample ID | Metric | Score | Threshold | Status |",
            "| --- | --- | --- | --- | --- |",
        ]

        total_score = 0.0
        total_count = 0

        for sample_id, metric_results in results.items():
            for mr in metric_results:
                status = "✅ Pass" if mr.passed else "❌ Fail"
                lines.append(
                    f"| `{sample_id}` | {mr.name} | {mr.score:.4f} | {mr.threshold:.2f} | {status} |"
                )
                total_score += mr.score
                total_count += 1

        if total_count:
            avg = total_score / total_count
            lines.append(f"| — | **Average** | **{avg:.4f}** | — | — |")

        lines += [
            "",
            "<details><summary>Cost summary</summary>",
            "",
            f"- Total metrics scored: {total_count}",
            f"- Samples evaluated: {len(results)}",
            f"- Judge model calls: ~{total_count * 2} (decomposition + verification per metric)",
            "",
            "</details>",
        ]

        return "\n".join(lines)

    def post_pr_comment(
        self,
        markdown: str,
        github_token: str,
        repo: str,
        pr_number: int,
    ) -> None:
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        payload = {"body": markdown}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            if response.ok:
                logger.info(
                    "PR comment posted to %s#%d (comment id: %s)",
                    repo,
                    pr_number,
                    response.json().get("id"),
                )
            else:
                logger.warning(
                    "Failed to post PR comment: %d %s", response.status_code, response.text
                )
        except Exception as exc:
            logger.warning("PR comment request raised an exception: %s", exc)

    def write_json_results(self, results: dict, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results written to %s", output_path)
