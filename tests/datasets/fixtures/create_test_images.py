"""
Create test images for VLM testing

Creates simple, recognizable images to verify VLM understanding.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path


def create_simple_shapes_image(output_path):
    """Create image with simple geometric shapes."""
    # Create white background
    img = Image.new('RGB', (448, 448), color='white')
    draw = ImageDraw.Draw(img)

    # Draw shapes
    # Red circle
    draw.ellipse([50, 50, 150, 150], fill='red', outline='darkred', width=3)

    # Blue square
    draw.rectangle([200, 50, 300, 150], fill='blue', outline='darkblue', width=3)

    # Green triangle
    draw.polygon([(350, 150), (400, 50), (450, 150)], fill='green', outline='darkgreen')

    # Yellow star (approximation)
    center = (125, 300)
    points = []
    for i in range(10):
        angle = i * 36  # 360/10
        radius = 50 if i % 2 == 0 else 25
        x = center[0] + radius * np.cos(np.radians(angle - 90))
        y = center[1] + radius * np.sin(np.radians(angle - 90))
        points.append((x, y))
    draw.polygon(points, fill='yellow', outline='orange')

    # Add text
    try:
        # Try to use a default font
        draw.text((50, 400), "Shapes: Circle, Square, Triangle", fill='black')
    except:
        pass

    img.save(output_path)
    print(f"✅ Created: {output_path}")
    return img


def create_text_image(output_path):
    """Create image with large text."""
    img = Image.new('RGB', (448, 448), color='lightblue')
    draw = ImageDraw.Draw(img)

    # Draw large text
    try:
        # Try to get a larger font
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
        except:
            font = ImageFont.load_default()
    except:
        font = None

    # Main text
    text = "HELLO\nWORLD"
    draw.multiline_text((100, 150), text, fill='darkblue', font=font, align='center')

    # Border
    draw.rectangle([10, 10, 438, 438], outline='navy', width=5)

    img.save(output_path)
    print(f"✅ Created: {output_path}")
    return img


def create_color_grid(output_path):
    """Create a grid of colors."""
    colors = [
        ('red', (255, 0, 0)), ('green', (0, 255, 0)), ('blue', (0, 0, 255)),
        ('yellow', (255, 255, 0)), ('cyan', (0, 255, 255)), ('magenta', (255, 0, 255)),
        ('orange', (255, 165, 0)), ('purple', (128, 0, 128)), ('pink', (255, 192, 203)),
    ]

    img = Image.new('RGB', (448, 448), color='white')
    draw = ImageDraw.Draw(img)

    cell_size = 448 // 3
    for i, (name, color) in enumerate(colors):
        row = i // 3
        col = i % 3
        x1 = col * cell_size
        y1 = row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size

        draw.rectangle([x1, y1, x2, y2], fill=color, outline='white', width=3)

    img.save(output_path)
    print(f"✅ Created: {output_path}")
    return img


def create_pattern_image(output_path):
    """Create an image with recognizable patterns."""
    img = Image.new('RGB', (448, 448), color='white')
    draw = ImageDraw.Draw(img)

    # Create checkerboard pattern
    cell_size = 56  # 448 / 8 = 56
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                color = 'black'
            else:
                color = 'white'

            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            draw.rectangle([x1, y1, x2, y2], fill=color)

    img.save(output_path)
    print(f"✅ Created: {output_path}")
    return img


def main():
    """Create all test images."""
    print("Creating test images for VLM testing...")
    print("=" * 60)

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create images
    create_simple_shapes_image(output_dir / "shapes.jpg")
    create_text_image(output_dir / "text.jpg")
    create_color_grid(output_dir / "colors.jpg")
    create_pattern_image(output_dir / "checkerboard.jpg")

    print("=" * 60)
    print("✅ All test images created!")
    print(f"Location: {output_dir}")
    print("\nImages:")
    print("  - shapes.jpg: Geometric shapes (circle, square, triangle, star)")
    print("  - text.jpg: Large text 'HELLO WORLD'")
    print("  - colors.jpg: 3x3 grid of colors")
    print("  - checkerboard.jpg: 8x8 checkerboard pattern")
    print("\nUse these to test VLM understanding:")
    print("  - Can it count shapes?")
    print("  - Can it read text?")
    print("  - Can it identify colors?")
    print("  - Can it recognize patterns?")


if __name__ == "__main__":
    main()
