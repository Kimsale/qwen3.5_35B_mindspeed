#!/bin/bash
# 检查权重转换进度的快速脚本

echo "=== 权重转换进度检查 ==="
echo

# 1. 检查进程是否还在运行
if ps -p 1861720 > /dev/null 2>&1; then
    echo "✅ 转换进程运行中 (PID 1861720)"
    ps -p 1861720 -o pid,etime,%cpu,rss,cmd --no-headers | awk '{
        printf "   运行时间: %s\n", $2
        printf "   CPU占用: %.1f%%\n", $3
        printf "   内存: %.1f GB\n", $4/1024/1024
    }'
else
    echo "❌ 转换进程已结束"
fi

echo

# 2. 检查输出目录内容
OUTPUT_DIR="/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8"
echo "📂 输出目录: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    file_count=$(find "$OUTPUT_DIR" -type f | wc -l)
    total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "   文件数量: $file_count"
    echo "   总大小: $total_size"
    echo

    # 3. 检查关键标志（iter_0000000 目录表示转换完成）
    if [ -d "$OUTPUT_DIR/iter_0000000" ]; then
        echo "✅ 转换完成！发现 iter_0000000/ 目录"
        echo "   内容预览:"
        ls -lh "$OUTPUT_DIR/iter_0000000/" | head -10 | sed 's/^/     /'
        echo
        echo "🚀 可以启动训练:"
        echo "   bash /data/sejin/baseline_26/scripts/train_hulk_aligned_ready.sh"
    else
        echo "⏳ 转换进行中..."
        echo "   当前文件:"
        ls -lh "$OUTPUT_DIR/" 2>/dev/null | head -10 | sed 's/^/     /'
    fi
else
    echo "❌ 输出目录不存在"
fi

echo
echo "=== 检查完成 ==="
