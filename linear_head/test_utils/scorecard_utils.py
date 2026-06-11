from pathlib import Path
from typing import Any


######################## #
# 마크다운 보고서 생성 함수
######################## #
def generate_markdown_report(
    model_name: str,
    dataset_name: str,
    checkpoint_path: Path,
    class_names: list[str],
    eval_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """추론 결과 리스트를 기반으로 Markdown 평가 보고서를 작성하여 저장합니다.
    Args:
        model_name (str): 모델 이름
        dataset_name (str): 데이터셋 이름
        checkpoint_path (Path): 사용된 가중치 경로
        class_names (list[str]): 클래스 이름 리스트
        eval_results (list[dict[str, Any]]): 각 샘플의 추론 정보 딕셔너리 리스트
        output_dir (Path): 보고서가 저장될 폴더 경로
    """
    correct_count = sum(1 for res in eval_results if res["match"])
    total_count = len(eval_results)
    accuracy = (correct_count / total_count) * 100.0 if total_count > 0 else 0.0

    # 1. 헤더 및 메타데이터 작성
    md_lines = [
        "# DINOv3 Inference Evaluation Report",
        "",
        "## 📋 Run Metadata",
        f"- **Model Backbone**: `{model_name}`",
        f"- **Dataset Split**: `{dataset_name}`",
        f"- **Checkpoint Path**: `{checkpoint_path}`",
        f"- **Total Samples Evaluated**: `{total_count}`",
        f"- **Overall Accuracy**: `{accuracy:.2f}%` ({correct_count}/{total_count})",
        "",
        "## 🔍 Detailed Inference Results",
        "",
        "| Index | File Name | True Class | Predicted Class | Probabilities | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :---: |",
    ]

    # 2. 결과 테이블 행 작성
    for idx, res in enumerate(eval_results):
        # 확률 문자열 결합 (예: cut: 98.2% | danger: 1.5%)
        prob_strs = [
            f"{class_names[c_idx]}: {prob * 100.0:.2f}%"
            for c_idx, prob in enumerate(res["probs"])
        ]
        prob_summary = " | ".join(prob_strs)
        status_icon = "✅" if res["match"] else "❌"

        md_lines.append(
            f"| {idx+1} | `{res['filename']}` | {res['true_class']} | {res['pred_class']} | {prob_summary} | {status_icon} |"
        )

    md_content = "\n".join(md_lines) + "\n"

    # 3. 마크다운 파일 저장
    report_path = output_dir / "test_report.md"
    with open(file=report_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[*] Inference report successfully saved to: {report_path.resolve()}")
