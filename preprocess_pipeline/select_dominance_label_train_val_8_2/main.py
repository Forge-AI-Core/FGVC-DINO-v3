"""Main pipeline orchestrator for Group-Level Splitting Architecture (25%, 10%, 0% padding).

This script executes the steps sequentially for 25%, 10%, and 0% crop margins:
1. Cropping (Group Creation)
2. Label Synchronization (Cascade Consensus majority voting)
3. Clustering
4. Folder Split (8:2)
5. Train/Val Sampler (Unique)
6. Reunion Splitter (Vanilla)
"""

import sys
from pathlib import Path

# 임포트를 위해 현재 디렉토리를 임시로 sys.path에 추가하지만 차후 초기화를 위해 added_to_path 카운터를 같이 활성합니다.
current_dir = Path(__file__).resolve().parent
added_to_path = False
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    added_to_path = True

try:
    # 25% Pipeline
    from base_25pct import cropper_25pct
    from base_25pct import label_synchronizer as label_synchronizer_25
    from base_25pct import cluster_25pct
    from base_25pct import folder_splitter_25pct
    from base_25pct import train_val_sampler_25pct
    from base_25pct import reunion_splitter_25pct

    # 10% Pipeline
    from base_50pct import cropper_50pct
    from base_50pct import label_synchronizer as label_synchronizer_10
    from base_50pct import cluster_50pct
    from base_50pct import folder_splitter_50pct
    from base_50pct import train_val_sampler_50pct
    from base_50pct import reunion_splitter_50pct

    # 0% Pipeline
    from base_0pct import cropper_0pct
    from base_0pct import label_synchronizer as label_synchronizer_0
    from base_0pct import cluster_0pct
    from base_0pct import folder_splitter_0pct
    from base_0pct import train_val_sampler_0pct
    from base_0pct import reunion_splitter_0pct
finally:
    # 임포트가 완료되든 트라이문이 끊기든 상관없이 sys.path의 초기화를 보장합니다.
    if added_to_path:
        sys.path.remove(str(current_dir))


def run_pipeline() -> None:
    print("=" * 70)
    print(" 🚀 Starting Preprocessing Pipeline Orchestration (25% -> 50% -> 0%) ")
    print("=" * 70)

    # ------------------ 25% Pipeline ------------------
    print("\n" + "#" * 60)
    print(" 🌟 RUNNING 25% PADDING PIPELINE 🌟 ")
    print("#" * 60)
    
    print("\n" + "=" * 50)
    print(" [25%] Step 1: Cropping ")
    print("=" * 50)
    cropper_25pct.main()

    print("\n" + "=" * 50)
    print(" [25%] Step 2: Label Synchronization (Cascade Consensus) ")
    print("=" * 50)
    label_synchronizer_25.main()

    print("\n" + "=" * 50)
    print(" [25%] Step 3: Clustering ")
    print("=" * 50)
    cluster_25pct.main()

    print("\n" + "=" * 50)
    print(" [25%] Step 4: Folder Split (8:2) ")
    print("=" * 50)
    folder_splitter_25pct.main()


    print("\n" + "=" * 50)
    print(" [25%] Step 5: Train/Val Sampler (Unique) ")
    print("=" * 50)
    train_val_sampler_25pct.main()

    print("\n" + "=" * 50)
    print(" [25%] Step 6: Reunion Splitter (Vanilla) ")
    print("=" * 50)
    reunion_splitter_25pct.main()

    # ------------------ 10% Pipeline ------------------
    print("\n" + "#" * 60)
    print(" 🌟 RUNNING 50% PADDING PIPELINE 🌟 ")
    print("#" * 60)
    
    print("\n" + "=" * 50)
    print(" [50%] Step 1: Cropping ")
    print("=" * 50)
    cropper_50pct.main()

    print("\n" + "=" * 50)
    print(" [50%] Step 2: Label Synchronization (Cascade Consensus) ")
    print("=" * 50)
    label_synchronizer_10.main()

    print("\n" + "=" * 50)
    print(" [50%] Step 3: Clustering ")
    print("=" * 50)
    cluster_50pct.main()

    print("\n" + "=" * 50)
    print(" [50%] Step 4: Folder Split (8:2) ")
    print("=" * 50)
    folder_splitter_50pct.main()


    print("\n" + "=" * 50)
    print(" [50%] Step 5: Train/Val Sampler (Unique) ")
    print("=" * 50)
    train_val_sampler_50pct.main()

    print("\n" + "=" * 50)
    print(" [50%] Step 6: Reunion Splitter (Vanilla) ")
    print("=" * 50)
    reunion_splitter_50pct.main()

    # ------------------ 0% Pipeline ------------------
    print("\n" + "#" * 60)
    print(" 🌟 RUNNING 0% PADDING PIPELINE 🌟 ")
    print("#" * 60)
    
    print("\n" + "=" * 50)
    print(" [0%] Step 1: Cropping ")
    print("=" * 50)
    cropper_0pct.main()

    print("\n" + "=" * 50)
    print(" [0%] Step 2: Label Synchronization (Cascade Consensus) ")
    print("=" * 50)
    label_synchronizer_0.main()

    print("\n" + "=" * 50)
    print(" [0%] Step 3: Clustering ")
    print("=" * 50)
    cluster_0pct.main()

    print("\n" + "=" * 50)
    print(" [0%] Step 4: Folder Split (8:2) ")
    print("=" * 50)
    folder_splitter_0pct.main()


    print("\n" + "=" * 50)
    print(" [0%] Step 5: Train/Val Sampler (Unique) ")
    print("=" * 50)
    train_val_sampler_0pct.main()

    print("\n" + "=" * 50)
    print(" [0%] Step 6: Reunion Splitter (Vanilla) ")
    print("=" * 50)
    reunion_splitter_0pct.main()

    print("\n🎉 모든 파이프라인 처리가 성공적으로 완료되었습니다!")
    print("이제 완벽하게 공정한 비교 평가를 진행하실 수 있습니다.")


if __name__ == "__main__":
    run_pipeline()
