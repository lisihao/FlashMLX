#!/usr/bin/env python3
"""
下载 VLM 测试所需的测试图像

Usage:
    python tests/scripts/download_fixtures.py
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from PIL import Image
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def create_test_images():
    """创建测试图像（如果无法下载真实图像）"""

    fixtures_dir = project_root / "tests" / "fixtures" / "images"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating test images in {fixtures_dir}...")

    # 创建简单的测试图像
    test_images = {
        "cat.jpg": (255, 200, 180),      # 暖色调
        "dog.jpg": (180, 160, 140),      # 棕色调
        "bird.jpg": (100, 150, 255),     # 蓝色调
        "car.jpg": (200, 50, 50),        # 红色调
        "tree.jpg": (50, 200, 100),      # 绿色调
    }

    for i in range(10):
        # 创建 fixture_N.jpg
        colors = [(200, 100, 50), (50, 200, 100), (100, 50, 200)]
        color = colors[i % 3]
        create_colored_image(
            fixtures_dir / f"fixture_{i}.jpg",
            size=(336, 336),
            color=color
        )

    for name, color in test_images.items():
        create_colored_image(
            fixtures_dir / name,
            size=(336, 336),
            color=color
        )

    print(f"✅ Created {len(test_images) + 10} test images")
    return fixtures_dir


def create_colored_image(path, size=(336, 336), color=(128, 128, 128)):
    """创建纯色测试图像"""

    # 创建渐变图像（更真实）
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # 添加渐变效果
            factor_x = x / width
            factor_y = y / height

            r = int(color[0] * (0.7 + 0.3 * factor_x))
            g = int(color[1] * (0.7 + 0.3 * factor_y))
            b = int(color[2] * (0.7 + 0.3 * (factor_x + factor_y) / 2))

            img_array[y, x] = [
                min(255, r),
                min(255, g),
                min(255, b)
            ]

    # 添加一些噪声（更真实）
    noise = np.random.randint(-10, 10, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    img.save(path, quality=95)


def download_real_images():
    """
    下载真实测试图像（可选）

    注意: 这里使用占位符 URL。实际使用时需要替换为真实的公开图像 URL。
    """

    fixtures_dir = project_root / "tests" / "fixtures" / "images"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # 公开的测试图像 URL（示例）
    # 实际使用时需要替换为可用的 URL
    real_image_urls = {
        # "cat.jpg": "https://example.com/cat.jpg",
        # "dog.jpg": "https://example.com/dog.jpg",
    }

    if not real_image_urls:
        print("⚠️  No real image URLs configured, using synthetic images")
        return False

    print(f"Downloading real images to {fixtures_dir}...")

    for filename, url in real_image_urls.items():
        output_path = fixtures_dir / filename
        try:
            print(f"  Downloading {filename}...")
            urlretrieve(url, output_path)
            print(f"  ✅ {filename}")
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")
            return False

    return True


def verify_images():
    """验证图像可用性"""

    fixtures_dir = project_root / "tests" / "fixtures" / "images"

    if not fixtures_dir.exists():
        print("❌ Fixtures directory not found")
        return False

    image_files = list(fixtures_dir.glob("*.jpg")) + list(fixtures_dir.glob("*.png"))

    if len(image_files) < 10:
        print(f"❌ Not enough images: {len(image_files)} < 10")
        return False

    # 验证图像可加载
    for img_path in image_files[:3]:  # 验证前 3 个
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            print(f"❌ Invalid image {img_path}: {e}")
            return False

    print(f"✅ Verified {len(image_files)} images")
    return True


def create_fixture_index():
    """创建测试图像索引文件"""

    fixtures_dir = project_root / "tests" / "fixtures" / "images"
    index_path = project_root / "tests" / "fixtures" / "image_index.json"

    image_files = sorted(fixtures_dir.glob("*.jpg"))

    import json

    index = {
        "total_images": len(image_files),
        "images": [
            {
                "filename": img.name,
                "path": str(img.relative_to(project_root)),
                "size": img.stat().st_size
            }
            for img in image_files
        ]
    }

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"✅ Created index: {index_path}")
    print(f"   Total images: {index['total_images']}")


def main():
    print("=" * 60)
    print("FlashMLX VLM Test Fixtures Preparation")
    print("=" * 60)
    print()

    # 尝试下载真实图像
    if not download_real_images():
        # 回退到创建合成图像
        create_test_images()

    # 验证图像
    if verify_images():
        print()
        print("✅ Test fixtures ready!")

        # 创建索引
        create_fixture_index()

        fixtures_dir = project_root / "tests" / "fixtures" / "images"
        print()
        print(f"Fixtures location: {fixtures_dir}")
        print(f"Total images: {len(list(fixtures_dir.glob('*.jpg')))}")
    else:
        print()
        print("❌ Failed to prepare test fixtures")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
