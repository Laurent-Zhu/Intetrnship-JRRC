from safetensors.torch import safe_open

# 替换为你的 safetensors 文件路径
file_path = "model-00001-of-00118.safetensors"

# 读取权重信息
weights_info = []
with safe_open(file_path, framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        # 展示前10个元素（多维张量会展平成一维）
        flat_values = tensor.flatten()[:10].tolist()
        weights_info.append({
            "name": key,
            "shape": str(tensor.shape),
            "dtype": str(tensor.dtype),
            "values": flat_values
        })

# 生成 HTML
html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>权重信息展示</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            min-height: 100vh;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: transparent;
            padding: 40px 0 32px 0;
            max-width: 1200px;
            width: 98vw;
            margin: 40px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #1e293b;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 10px;
            letter-spacing: 2px;
            text-align: center;
        }
        .subtitle {
            color: #64748b;
            font-size: 1.1rem;
            margin-bottom: 24px;
            text-align: center;
        }
        .card-list {
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            justify-content: center;
            width: 100%;
        }
        .card {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.10);
            padding: 24px 22px 18px 22px;
            min-width: 260px;
            max-width: 340px;
            flex: 1 1 300px;
            display: flex;
            flex-direction: column;
            margin-bottom: 0;
            transition: box-shadow 0.2s, transform 0.2s;
        }
        .card:hover {
            box-shadow: 0 8px 32px #bae6fd;
            transform: translateY(-4px) scale(1.03);
        }
        .card-title {
            font-weight: 700;
            font-size: 1.08rem;
            color: #2563eb;
            margin-bottom: 8px;
            word-break: break-all;
        }
        .card-meta {
            color: #64748b;
            font-size: 0.98rem;
            margin-bottom: 8px;
        }
        .card-label {
            color: #334155;
            font-weight: 600;
            margin-right: 4px;
        }
        .weight-values {
            background: #f1f5f9;
            border-radius: 6px;
            padding: 8px 10px;
            margin-top: 6px;
            font-family: 'JetBrains Mono', 'Consolas', 'Menlo', monospace;
            font-size: 0.98rem;
            color: #0f172a;
            overflow-x: auto;
            word-break: break-all;
        }
        @media (max-width: 900px) {
            .card-list { gap: 12px; }
            .card { min-width: 180px; max-width: 98vw; padding: 16px 8px 12px 8px; }
        }
        @media (max-width: 700px) {
            .container { padding: 12px 2vw; }
            h1 { font-size: 1.3rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Safetensors 权重信息展示</h1>
        <div class="subtitle">每个权重以卡片形式展示，含前10项权重值</div>
        <div class="card-list">
'''

for info in weights_info:
    html += f'''
            <div class="card">
                <div class="card-title">{info["name"]}</div>
                <div class="card-meta"><span class="card-label">形状:</span>{info["shape"]}</div>
                <div class="card-meta"><span class="card-label">数据类型:</span>{info["dtype"]}</div>
                <div class="card-meta"><span class="card-label">前10项权重值:</span></div>
                <div class="weight-values">{info["values"]}</div>
            </div>
'''

html += '''
        </div>
    </div>
</body>
</html>
'''

# 保存为 html 文件
with open("demo_html/safetensors_weights.html", "w", encoding="utf-8") as f:
    f.write(html)

print("HTML 文件已生成：safetensors_weights.html")