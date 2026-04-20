import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

from evals.dataset import DatasetLoader, EvalSample
from evals.metrics import EvalMetrics, MetricResult
from evals.reporter import EvalReporter

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

console = Console()


class EvalRunner:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Export it before running the eval suite."
            )

        judge_model = self._cfg["models"]["judge"]
        self._loader = DatasetLoader()
        self._metrics = EvalMetrics(judge_model=judge_model, gemini_api_key=gemini_api_key)
        self._reporter = EvalReporter()
        self._config_path = config_path

    def run(self, answers: list[dict] | None = None) -> bool:
        settings = self._cfg["settings"]
        golden_path = settings["golden_set_path"]
        s3_path = os.environ.get("GOLDEN_SET_S3_PATH")
        output_path = settings["results_output"]

        samples: list[EvalSample] = self._loader.load(
            local_path=golden_path, s3_path=s3_path or None
        )

        if answers:
            answer_map = {a["id"]: a["answer"] for a in answers}
            for sample in samples:
                if sample.id in answer_map:
                    sample.answer = answer_map[sample.id]
        else:
            # Baseline self-score: use ground_truth as the answer
            for sample in samples:
                if not sample.answer:
                    sample.answer = sample.ground_truth

        per_sample: dict[str, list[MetricResult]] = {}
        for sample in samples:
            per_sample[sample.id] = self._metrics.score_all(sample)

        # Aggregate: mean score per metric across all samples
        metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
        aggregated: dict[str, float] = {}
        for mname in metric_names:
            scores = [
                mr.score
                for results in per_sample.values()
                for mr in results
                if mr.name == mname
            ]
            aggregated[mname] = sum(scores) / len(scores) if scores else 0.0

        thresholds = self._cfg["thresholds"]
        failed_metrics = [
            mname
            for mname in metric_names
            if aggregated.get(mname, 0.0) < thresholds.get(mname, 0.0)
        ]
        overall_passed = len(failed_metrics) == 0

        results = {
            "overall_passed": overall_passed,
            "total_samples": len(samples),
            "failed_metrics": failed_metrics,
            "aggregated": aggregated,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "cost_summary": {
                "total_metric_calls": sum(len(v) for v in per_sample.values()),
                "judge_model": self._cfg["models"]["judge"],
                "estimated_llm_calls": sum(len(v) for v in per_sample.values()) * 2,
            },
            "per_sample": {
                sid: [asdict(mr) for mr in mrs] for sid, mrs in per_sample.items()
            },
        }

        self._reporter.write_json_results(results, output_path)

        markdown = self._reporter.format_markdown_table(per_sample, overall_passed)

        # Post PR comment if running in CI with a PR number
        github_token = os.environ.get("GITHUB_TOKEN")
        pr_number_str = os.environ.get("PR_NUMBER")
        repo = os.environ.get("GITHUB_REPOSITORY")
        if github_token and pr_number_str and repo:
            try:
                self._reporter.post_pr_comment(
                    markdown=markdown,
                    github_token=github_token,
                    repo=repo,
                    pr_number=int(pr_number_str),
                )
            except Exception as exc:
                logger.warning("Could not post PR comment: %s", exc)

        self._print_rich_table(aggregated, thresholds, overall_passed)

        return overall_passed

    def _print_rich_table(
        self,
        aggregated: dict[str, float],
        thresholds: dict[str, float],
        overall_passed: bool,
    ) -> None:
        status_str = "[bold green]PASSED[/bold green]" if overall_passed else "[bold red]FAILED[/bold red]"
        console.print(f"\n[bold]RAG Evaluation Result:[/bold] {status_str}\n")

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim", width=22)
        table.add_column("Avg Score", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Status", justify="center")

        for mname, score in aggregated.items():
            threshold = thresholds.get(mname, 0.0)
            passed = score >= threshold
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            table.add_row(mname, f"{score:.4f}", f"{threshold:.2f}", status)

        console.print(table)

    def run_ci(self) -> None:
        passed = self.run()
        if not passed:
            console.print("\n[bold red]EVAL FAILED[/bold red] — the following metrics are below threshold:\n")
            with open(self._cfg["settings"]["results_output"], "r") as f:
                import json
                data = json.load(f)
            thresholds = self._cfg["thresholds"]
            for mname in data.get("failed_metrics", []):
                score = data["aggregated"].get(mname, 0.0)
                threshold = thresholds.get(mname, 0.0)
                console.print(f"  [red]{mname}[/red]: score={score:.4f}  threshold={threshold:.2f}")
            sys.exit(1)
        console.print("\n[bold green]EVAL PASSED[/bold green] — all metrics above threshold.\n")
        sys.exit(0)


if __name__ == "__main__":
    runner = EvalRunner()
    runner.run_ci()
