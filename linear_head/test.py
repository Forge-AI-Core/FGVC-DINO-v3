from pathlib import Path
import random
import torch
from torch import nn

from linear_head.get_data_loaders import get_data_loader, SquarePad
from linear_head.test_utils.metrics_utils import generate_and_save_pr_curve
from linear_head.test_utils.attn_heatemap_utils import (
    patch_attention_module,
    generate_attention_heatmap,
    save_visualization_plot,
)
from linear_head.test_utils.scorecard_utils import generate_markdown_report


from PIL import Image


############ #
# 테스트 함수
############ #
def test_model(
    checkpoint_path: Path,
    device: torch.device,
    model: nn.Module,
    model_name: str,
    dataset_dir: Path,
    dataset_name: str,
    image_size: int,
) -> None:
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Attention 몽키 패치 적용
    patch_attention_module(model)

    # 데이터로더 정의
    _, _, test_loader = get_data_loader(
        dataset_dir=dataset_dir,
        batch_size=1,
        image_size=image_size,
    )
    test_dataset = test_loader.dataset
    class_names = test_dataset.classes

    print(
        f"[*] Test dataset loaded. Size: {len(test_dataset)} (Classes: {class_names})"
    )

    # 랜덤 10개 추출
    sample_indices = random.sample(
        range(len(test_dataset)), k=min(10, len(test_dataset))
    )

    print(f"[*] Selected indices for evaluation: {sample_indices}")

    # 결과 디렉토리 생성
    output_dir = Path(f"results/test_outputs/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 결과 디렉토리 초기화
    for p in output_dir.glob("attention_sample_*.png"):
        p.unlink()

    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    patch_size = base_vit.patch_size

    global captured_attn_weights

    eval_results = []
    for index, sample_index in enumerate(sample_indices):
        img_tensor, true_label = test_dataset[sample_index]
        img_path, _ = test_dataset.samples[sample_index]

        img_input = img_tensor.unsqueeze(dim=0).to(device)

        with torch.inference_mode():
            logits = model(img_input)
            probs = torch.softmax(input=logits, dim=1).squeeze().cpu()

        pred_label = int(probs.argmax())
        confidence = float(probs[pred_label])

        true_name = class_names[true_label]
        pred_name = class_names[pred_label]

        match_status = true_label == pred_label

        print(
            f"[{index+1}/10] Evaluated {Path(img_path).name} | "
            f"True: {true_name} | Pred: {pred_name} ({confidence * 100.0:.2f}%)"
        )
        print(
            f"[{index+1}/10] Visualizing {Path(img_path).name} | "
            f"True: {true_name} | Pred: {pred_name} ({confidence * 100.0:.2f}%)"
        )

        eval_results.append(
            {
                "filename": Path(img_path).name,
                "true_class": true_name,
                "pred_class": pred_name,
                "probs": probs.tolist(),
                "match": match_status,
            }
        )

        heatmap = generate_attention_heatmap(
            model=model,
            img_tensor=img_input,
            image_size=image_size,
            patch_size=patch_size,
        )

        # 다음 루프 반복을 위해 전역 캐시 초기화
        captured_attn_weights = None

        raw_img = Image.open(img_path).convert("RGB")
        orig_img = SquarePad()(raw_img)

        save_path = (
            output_dir
            / f"attention_sample_{index+1}_true_{true_name}_pred_{pred_name}.png"
        )
        save_visualization_plot(
            orig_img=orig_img,
            heatmap=heatmap,
            true_label_name=true_name,
            pred_label_name=pred_name,
            confidence=confidence,
            save_path=save_path,
        )

    # 마크다운 리포트 생성 함수 호출
    generate_markdown_report(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        eval_results=eval_results,
        output_dir=output_dir,
    )

    # 전체 테스트 데이터셋에 대한 PR 커브 생성 및 플롯 저장
    pr_save_path = output_dir / "pr_curve.png"
    generate_and_save_pr_curve(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_path=pr_save_path,
    )

    print(f"[*] Done! Attention overlays and PR curve saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    test_model()
