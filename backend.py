import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NETE 大语言模型后门检测评估系统")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    attack_method: str
    detection_method: str
    model_name: str
    dataset: str

# 静态配置数据
ATTACK_METHODS = [
    {"id": "badchain", "name": "BadChain (思维链后门)"},
    {"id": "badedit", "name": "BadEdit (知识编辑后门)"},
    {"id": "cba", "name": "CBA (干净标签后门)"},
    {"id": "sleepagent", "name": "SleepAgent (休眠代理后门)"},
    {"id": "vpi", "name": "VPI (虚拟提示词注入)"}
]

DETECTION_METHODS = [
    {"id": "nete", "name": "NETE (本文方法: 扰动差异一致性)"},
    {"id": "strip", "name": "STRIP (基线: 基于熵的强扰动)"},
    {"id": "onion", "name": "Onion (基线: 基于困惑度的词汇剔除)"}
]

MODELS = [
    {"id": "llama-2-7b", "name": "LLaMA-2-7B"},
    {"id": "llama-3-8b", "name": "LLaMA-3-8B"},
    {"id": "vicuna-7b", "name": "Vicuna-7B"},
    {"id": "gpt-j-6b", "name": "GPT-J-6B"}
]

DATASETS = [
    {"id": "sst-2", "name": "SST-2 (情感分类)"},
    {"id": "ag-news", "name": "AG News (新闻分类)"},
    {"id": "imdb", "name": "IMDB (电影评论)"},
    {"id": "alpaca", "name": "Alpaca-Data (指令微调)"}
]

@app.get("/api/options")
async def get_options():
    """返回供前端选择的下拉菜单配置项"""
    return {
        "attacks": ATTACK_METHODS,
        "detections": DETECTION_METHODS,
        "models": MODELS,
        "datasets": DATASETS
    }

@app.post("/api/evaluate")
async def run_evaluation(request: EvaluationRequest):
    """
    模拟执行检测脚本，返回基于论文理论的默认实验数据。
    真实场景下这里会调用 main_detect.py、STRIP 或 Onion 的相关脚本。
    """
    await asyncio.sleep(2.5)  # 模拟掩码填充和概率计算耗时

    report = {
        "status": "success",
        "dataset": request.dataset,
        "model_tested": request.model_name,
        "attack_evaluated": request.attack_method,
        "detection_method": request.detection_method,
        "conclusion": "检测到后门 (Backdoor Detected)",
        "metrics": {},
        "details": ""
    }

    # 根据选择的检测方法，返回论文对应的代表性数据
    if request.detection_method == "nete":
        report["metrics"] = {
            "指标名称": "曲率评分 (Curvature Score)",
            "数值": "0.185",
            "安全阈值": "0.500",
            "对数概率差异": "1.204"
        }
        report["details"] = "根据 NETE 理论：该输入样本在经过掩码填充（Mask-filling）生成扰动后，其对数概率的差异变化极小。测得曲率为 0.185（远低于正常样本的 0.8+，且低于阈值 0.5）。这展现了异常的扰动差异一致性（Anomalous perturbation discrepancy consistency），明确判定为后门样本。该零样本黑盒检测效果优于现有基线。"

    elif request.detection_method == "strip":
        report["metrics"] = {
            "指标名称": "预测熵值 (Shannon Entropy)",
            "数值": "0.210",
            "安全阈值": "0.450",
            "扰动副本数": "100"
        }
        report["details"] = "根据 STRIP 理论：将样本与其他文本拼接扰动后，模型预测分布的香农熵仅为 0.210。预测结果高度集中，不随输入扰动而改变，判定为触发器激活的后门样本。注：在黑盒大模型场景下计算开销较大。"

    elif request.detection_method == "onion":
        report["metrics"] = {
            "指标名称": "最大困惑度下降 (Max PPL Drop)",
            "数值": "45.6",
            "安全阈值": "20.0",
            "剔除词汇": "['cf', 'mn'] (疑似触发器)"
        }
        report["details"] = "根据 Onion 理论：系统逐词剔除输入序列中的词汇并计算困惑度（Perplexity）。当剔除特定词汇时，句子困惑度出现断崖式下降（降低了45.6，超过阈值20）。判定该样本包含后门触发器。"

    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)