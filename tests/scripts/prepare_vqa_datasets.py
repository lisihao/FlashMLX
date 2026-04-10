#!/usr/bin/env python3
"""
准备 VQA 测试数据集

Usage:
    python tests/scripts/prepare_vqa_datasets.py
"""

import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
import random

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_synthetic_vqa_samples():
    """
    创建合成 VQA 样本（用于测试）

    由于真实 VQA 数据集需要授权和大量下载，这里创建合成样本用于测试。
    实际生产环境需要下载真实数据集。
    """

    datasets_dir = project_root / "tests" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic VQA datasets in {datasets_dir}...")

    # VQAV2 样本
    create_vqav2_samples(datasets_dir, num_samples=500)

    # GQA 样本
    create_gqa_samples(datasets_dir, num_samples=200)

    # TextVQA 样本
    create_textvqa_samples(datasets_dir, num_samples=200)

    # POPE 样本
    create_pope_samples(datasets_dir, num_samples=200)

    print("✅ Created all synthetic VQA datasets")


def create_vqav2_samples(datasets_dir: Path, num_samples: int):
    """创建 VQAV2 样本"""

    vqav2_dir = datasets_dir / "vqav2"
    vqav2_dir.mkdir(exist_ok=True)

    questions = [
        "What color is the cat?",
        "How many dogs are in the image?",
        "What is the bird doing?",
        "Is this a car or a truck?",
        "What type of tree is shown?",
        "What is the main subject of the image?",
        "How many objects are visible?",
        "What is the background color?",
        "Is this indoors or outdoors?",
        "What time of day does this appear to be?",
    ]

    answers = [
        "orange", "brown", "blue", "red", "green",
        "one", "two", "three", "four", "five",
        "yes", "no", "maybe",
        "sitting", "standing", "flying", "running",
    ]

    samples = []
    for i in range(num_samples):
        # 使用 fixtures 中的图像
        image_id = i % 15
        if image_id < 10:
            image_file = f"fixture_{image_id}.jpg"
        else:
            image_names = ["cat.jpg", "dog.jpg", "bird.jpg", "car.jpg", "tree.jpg"]
            image_file = image_names[image_id - 10]

        sample = {
            "question_id": f"vqav2_{i}",
            "image_id": f"img_{image_id}",
            "image": f"tests/fixtures/images/{image_file}",
            "question": random.choice(questions),
            "answers": [
                {"answer": random.choice(answers), "confidence": 1.0}
            ],
            "answer_type": "other",
            "question_type": "what color",
        }
        samples.append(sample)

    # 保存
    output_file = vqav2_dir / "val_samples.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"  ✅ VQAV2: {num_samples} samples → {output_file}")


def create_gqa_samples(datasets_dir: Path, num_samples: int):
    """创建 GQA 样本"""

    gqa_dir = datasets_dir / "gqa"
    gqa_dir.mkdir(exist_ok=True)

    questions = [
        "Is the cat on the left or right side?",
        "What is between the dog and the tree?",
        "Which object is larger, the car or the bird?",
        "Are there more cats or dogs?",
        "What is the spatial relationship between objects?",
    ]

    answers = ["left", "right", "center", "top", "bottom", "yes", "no"]

    samples = []
    for i in range(num_samples):
        image_id = i % 15
        if image_id < 10:
            image_file = f"fixture_{image_id}.jpg"
        else:
            image_names = ["cat.jpg", "dog.jpg", "bird.jpg", "car.jpg", "tree.jpg"]
            image_file = image_names[image_id - 10]

        sample = {
            "question_id": f"gqa_{i}",
            "image_id": f"img_{image_id}",
            "image": f"tests/fixtures/images/{image_file}",
            "question": random.choice(questions),
            "answer": random.choice(answers),
            "full_answer": random.choice(answers),
            "semantic": ["spatial", "logical"],
        }
        samples.append(sample)

    output_file = gqa_dir / "val_samples.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"  ✅ GQA: {num_samples} samples → {output_file}")


def create_textvqa_samples(datasets_dir: Path, num_samples: int):
    """创建 TextVQA 样本"""

    textvqa_dir = datasets_dir / "textvqa"
    textvqa_dir.mkdir(exist_ok=True)

    questions = [
        "What text is visible in the image?",
        "What does the sign say?",
        "What number is shown?",
        "What word appears on the object?",
        "What is written on the surface?",
    ]

    # TextVQA 的答案通常是图像中的文本
    text_answers = [
        "STOP", "EXIT", "OPEN", "CLOSED", "123",
        "ABC", "HELLO", "WELCOME", "CAT", "DOG",
    ]

    samples = []
    for i in range(num_samples):
        image_id = i % 15
        if image_id < 10:
            image_file = f"fixture_{image_id}.jpg"
        else:
            image_names = ["cat.jpg", "dog.jpg", "bird.jpg", "car.jpg", "tree.jpg"]
            image_file = image_names[image_id - 10]

        sample = {
            "question_id": f"textvqa_{i}",
            "image_id": f"img_{image_id}",
            "image": f"tests/fixtures/images/{image_file}",
            "question": random.choice(questions),
            "answers": [random.choice(text_answers) for _ in range(3)],
            "ocr_tokens": random.sample(text_answers, k=min(5, len(text_answers))),
        }
        samples.append(sample)

    output_file = textvqa_dir / "val_samples.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"  ✅ TextVQA: {num_samples} samples → {output_file}")


def create_pope_samples(datasets_dir: Path, num_samples: int):
    """创建 POPE (Polling-based Object Probing Evaluation) 样本"""

    pope_dir = datasets_dir / "pope"
    pope_dir.mkdir(exist_ok=True)

    # POPE 专注于检测幻觉 (hallucination)
    # 问题形式: "Is there a [object] in the image?"
    objects = [
        "cat", "dog", "bird", "car", "tree",
        "person", "chair", "table", "book", "phone",
        "bicycle", "umbrella", "bag", "hat", "shoe",
    ]

    samples = []
    for i in range(num_samples):
        image_id = i % 15
        if image_id < 10:
            image_file = f"fixture_{image_id}.jpg"
        else:
            image_names = ["cat.jpg", "dog.jpg", "bird.jpg", "car.jpg", "tree.jpg"]
            image_file = image_names[image_id - 10]

        obj = random.choice(objects)

        # 50% 的样本有对象 (yes), 50% 没有 (no)
        # 用于测试模型是否会产生幻觉
        has_object = i % 2 == 0

        sample = {
            "question_id": f"pope_{i}",
            "image_id": f"img_{image_id}",
            "image": f"tests/fixtures/images/{image_file}",
            "question": f"Is there a {obj} in the image?",
            "answer": "yes" if has_object else "no",
            "label": has_object,
            "object": obj,
        }
        samples.append(sample)

    output_file = pope_dir / "val_samples.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"  ✅ POPE: {num_samples} samples → {output_file}")


def download_real_datasets():
    """
    下载真实 VQA 数据集（可选）

    注意:
    1. VQAV2: https://visualqa.org/download.html
    2. GQA: https://cs.stanford.edu/people/dorarad/gqa/download.html
    3. TextVQA: https://textvqa.org/dataset
    4. POPE: https://github.com/AoiDragon/POPE

    这些数据集需要授权和大量存储空间（10GB+），实际使用时需要手动下载。
    """

    print("⚠️  Real dataset download not implemented")
    print("   For production use, please download datasets manually:")
    print("   - VQAV2: https://visualqa.org/download.html")
    print("   - GQA: https://cs.stanford.edu/people/dorarad/gqa/download.html")
    print("   - TextVQA: https://textvqa.org/dataset")
    print("   - POPE: https://github.com/AoiDragon/POPE")
    print()
    print("   Using synthetic samples for testing...")
    return False


def verify_datasets():
    """验证数据集可用性"""

    datasets_dir = project_root / "tests" / "datasets"

    if not datasets_dir.exists():
        print("❌ Datasets directory not found")
        return False

    required_datasets = ["vqav2", "gqa", "textvqa", "pope"]

    for dataset_name in required_datasets:
        dataset_dir = datasets_dir / dataset_name
        if not dataset_dir.exists():
            print(f"❌ Missing dataset: {dataset_name}")
            return False

        sample_file = dataset_dir / "val_samples.json"
        if not sample_file.exists():
            print(f"❌ Missing samples file: {sample_file}")
            return False

        # 验证 JSON 格式
        try:
            with open(sample_file, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                print(f"❌ Invalid samples file: {sample_file}")
                return False
        except Exception as e:
            print(f"❌ Failed to load {sample_file}: {e}")
            return False

    print(f"✅ Verified all datasets")
    return True


def create_dataset_index():
    """创建数据集索引文件"""

    datasets_dir = project_root / "tests" / "datasets"
    index_path = datasets_dir / "dataset_index.json"

    datasets = {}
    for dataset_name in ["vqav2", "gqa", "textvqa", "pope"]:
        dataset_dir = datasets_dir / dataset_name
        sample_file = dataset_dir / "val_samples.json"

        with open(sample_file, 'r') as f:
            data = json.load(f)

        datasets[dataset_name] = {
            "name": dataset_name,
            "num_samples": len(data),
            "sample_file": str(sample_file.relative_to(project_root)),
            "format": "json",
        }

    index = {
        "total_datasets": len(datasets),
        "datasets": datasets,
        "version": "1.0",
        "created_for": "FlashMLX VLM Testing",
    }

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"✅ Created index: {index_path}")
    print(f"   Total datasets: {index['total_datasets']}")
    for name, info in datasets.items():
        print(f"   - {name}: {info['num_samples']} samples")


def main():
    print("=" * 60)
    print("FlashMLX VQA Dataset Preparation")
    print("=" * 60)
    print()

    # 尝试下载真实数据集
    if not download_real_datasets():
        # 回退到创建合成样本
        create_synthetic_vqa_samples()

    # 验证数据集
    if verify_datasets():
        print()
        print("✅ VQA datasets ready!")

        # 创建索引
        create_dataset_index()

        datasets_dir = project_root / "tests" / "datasets"
        print()
        print(f"Datasets location: {datasets_dir}")
        print("Next steps:")
        print("  1. Run download_fixtures.py to create test images")
        print("  2. Use these datasets for VLM testing")
    else:
        print()
        print("❌ Failed to prepare VQA datasets")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
