#!/bin/bash

# Ferrum Metal vs CPU Performance Benchmark Script
# Apple GPU优化性能测试

echo "🍎 Ferrum Apple GPU 优化性能测试"
echo "=================================="

# 确保已编译
echo "📦 编译项目..."
cargo build --release --features metal

# 测试配置
HOST="127.0.0.1"
PORT_CPU=8001
PORT_METAL=8002
MODEL="dummy"
WARMUP_REQUESTS=3
BENCHMARK_REQUESTS=10

# 性能测试函数
run_benchmark() {
    local backend=$1
    local port=$2
    local log_file="benchmark_${backend}.log"
    
    echo ""
    echo "🔧 测试 ${backend} 后端:"
    echo "Port: ${port}"
    
    # 启动服务器
    echo "启动服务器..."
    ./target/release/ferrum serve --backend $backend --model $MODEL --port $port > $log_file 2>&1 &
    local server_pid=$!
    
    # 等待服务器启动
    echo "等待服务器初始化..."
    for i in {1..30}; do
        if curl -s http://${HOST}:${port}/health > /dev/null 2>&1; then
            echo "✅ 服务器启动成功 (${i}秒)"
            break
        fi
        sleep 1
    done
    
    # Warmup
    echo "🔥 预热 ($WARMUP_REQUESTS 次请求)..."
    for i in $(seq 1 $WARMUP_REQUESTS); do
        curl -s -X POST http://${HOST}:${port}/v1/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"'$MODEL'","prompt":"Hello","max_tokens":10}' > /dev/null
    done
    
    # 实际测试
    echo "⚡ 开始性能测试 ($BENCHMARK_REQUESTS 次请求)..."
    local start_time=$(date +%s.%3N)
    
    for i in $(seq 1 $BENCHMARK_REQUESTS); do
        local req_start=$(date +%s.%3N)
        curl -s -X POST http://${HOST}:${port}/v1/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"'$MODEL'","prompt":"Hello world, this is a test","max_tokens":50}' \
            > /dev/null
        local req_end=$(date +%s.%3N)
        local req_time=$(echo "$req_end - $req_start" | bc)
        echo "  请求 $i: ${req_time}s"
    done
    
    local end_time=$(date +%s.%3N)
    local total_time=$(echo "$end_time - $start_time" | bc)
    local avg_time=$(echo "scale=3; $total_time / $BENCHMARK_REQUESTS" | bc)
    local rps=$(echo "scale=2; $BENCHMARK_REQUESTS / $total_time" | bc)
    
    echo "📊 ${backend} 后端性能结果:"
    echo "  总时间: ${total_time}s"
    echo "  平均延迟: ${avg_time}s"
    echo "  吞吐量: ${rps} RPS"
    
    # 停止服务器
    kill $server_pid
    wait $server_pid 2>/dev/null
    
    # 返回结果
    echo "$backend,$total_time,$avg_time,$rps" >> benchmark_results.csv
}

# 清理之前的结果
rm -f benchmark_*.log benchmark_results.csv
echo "backend,total_time,avg_latency,rps" > benchmark_results.csv

# 检查是否有Metal支持
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 检测到 macOS，支持 Metal 加速"
    
    # 测试CPU后端
    run_benchmark "cpu" $PORT_CPU
    
    # 测试Metal后端  
    run_benchmark "metal" $PORT_METAL
    
    echo ""
    echo "📈 性能对比结果:"
    echo "=================================="
    cat benchmark_results.csv | column -t -s ','
    
    echo ""
    echo "📋 详细日志："
    echo "CPU 后端日志: benchmark_cpu.log"
    echo "Metal 后端日志: benchmark_metal.log"
    
else
    echo "❌ 非 macOS 系统，只测试 CPU 后端"
    run_benchmark "cpu" $PORT_CPU
fi

echo ""
echo "✅ 性能测试完成！"
