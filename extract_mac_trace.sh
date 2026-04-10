#!/bin/bash
# MAC-Attention Metal Trace 自动提取脚本

set -e

TRACE_FILE="/tmp/mac_trace.gputrace"
OUTPUT_DIR="/tmp/mac_trace_analysis"
EXPORT_FILE="$OUTPUT_DIR/trace_export.xml"

echo "=========================================="
echo "MAC-Attention Metal Trace 分析"
echo "=========================================="
echo ""

# 检查 trace 文件
if [ ! -d "$TRACE_FILE" ]; then
    echo "❌ Trace 文件不存在: $TRACE_FILE"
    exit 1
fi

echo "✅ Trace 文件: $(du -sh $TRACE_FILE | cut -f1)"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# Step 1: 导出 trace 到 XML
echo "Step 1: 导出 trace 到 XML (可能需要几分钟)..."
if [ ! -f "$EXPORT_FILE" ]; then
    xctrace export --input "$TRACE_FILE" --output "$EXPORT_FILE" 2>&1 | head -20
    if [ $? -eq 0 ]; then
        echo "✅ 导出完成: $(du -sh $EXPORT_FILE | cut -f1)"
    else
        echo "⚠️  xctrace export 失败，尝试备用方案..."
        # 备用：直接分析 .trace 文件
        TRACE_BINARY=$(find "$TRACE_FILE" -name "*.trace" | head -1)
        if [ -n "$TRACE_BINARY" ]; then
            echo "   找到 trace 文件: $TRACE_BINARY"
            EXPORT_FILE="$TRACE_BINARY"
        else
            echo "❌ 无法找到 trace 数据文件"
            exit 1
        fi
    fi
else
    echo "✅ 使用已存在的导出文件"
fi
echo ""

# Step 2: 提取 Attention Kernel 信息
echo "Step 2: 提取 Attention Kernel 统计..."
cat > "$OUTPUT_DIR/extract_attention.py" << 'PYTHON_EOF'
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

xml_file = sys.argv[1]
print(f"解析 XML: {xml_file}")

try:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 查找所有 GPU kernel 调用
    attention_kernels = []
    for elem in root.iter():
        if 'kernel' in elem.tag.lower() or 'gpu' in elem.tag.lower():
            name = elem.get('name', '')
            if 'attention' in name.lower() or 'sdpa' in name.lower() or 'matmul' in name.lower():
                duration = elem.get('duration', '0')
                buffers = elem.findall('.//buffer')
                attention_kernels.append({
                    'name': name,
                    'duration': float(duration) if duration else 0,
                    'buffers': [(b.get('id'), b.get('size', '0')) for b in buffers]
                })

    if attention_kernels:
        print(f"\n✅ 找到 {len(attention_kernels)} 个 attention kernel 调用")
        print("\n前 10 个调用:")
        for i, k in enumerate(attention_kernels[:10]):
            print(f"  {i+1}. {k['name']}: {k['duration']:.3f} ms")
            if k['buffers']:
                print(f"     Buffers: {len(k['buffers'])} 个")
                for buf_id, buf_size in k['buffers'][:3]:
                    size_mb = int(buf_size) / (1024**2) if buf_size.isdigit() else 0
                    print(f"       - Buffer {buf_id}: {size_mb:.2f} MB")

        # 统计
        total_duration = sum(k['duration'] for k in attention_kernels)
        print(f"\n总耗时: {total_duration:.3f} ms")
        print(f"平均耗时: {total_duration/len(attention_kernels):.3f} ms/call")
    else:
        print("⚠️  未找到 attention kernel（XML 结构可能不同）")
        print("   尝试列出前 20 个元素:")
        for i, elem in enumerate(root.iter()):
            if i >= 20:
                break
            print(f"  {elem.tag}: {elem.attrib}")

except Exception as e:
    print(f"❌ 解析失败: {e}")
    print("   XML 文件可能太大或格式不支持")
    import traceback
    traceback.print_exc()
PYTHON_EOF

if [ -f "$EXPORT_FILE" ]; then
    python3 "$OUTPUT_DIR/extract_attention.py" "$EXPORT_FILE" > "$OUTPUT_DIR/attention_kernels.txt" 2>&1
    cat "$OUTPUT_DIR/attention_kernels.txt"
else
    echo "⚠️  跳过（无导出文件）"
fi
echo ""

# Step 3: 查找 buffer 大小（关键证据）
echo "Step 3: 查找 K/V buffer 大小..."
if [ -f "$EXPORT_FILE" ]; then
    grep -i "buffer" "$EXPORT_FILE" | grep -i "size" | head -20 > "$OUTPUT_DIR/buffers.txt" 2>/dev/null || echo "  无法提取 buffer 信息（可能需要手动分析）"
    if [ -s "$OUTPUT_DIR/buffers.txt" ]; then
        cat "$OUTPUT_DIR/buffers.txt"
    fi
fi
echo ""

# Step 4: 总结
echo "=========================================="
echo "分析完成"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  - $OUTPUT_DIR/attention_kernels.txt"
echo "  - $OUTPUT_DIR/buffers.txt"
echo ""
echo "下一步:"
echo "  1. 手动打开 Instruments: open $TRACE_FILE"
echo "  2. 查看 Buffer Bindings 确认 K/V buffer size"
echo "  3. 更新 MAC_ATTENTION_EXPERIMENTAL.md"
echo ""
