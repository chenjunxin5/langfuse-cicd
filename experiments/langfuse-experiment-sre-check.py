"""
Langfuse experiment gate — 多模型对比评估 sre-check 二分类任务
Expected by experiment-action: defines experiment(context: RunnerContext).

数据集格式：每个 item 需要在 metadata 中指定 model 名称
{
    "input": {"question": "..."},
    "expected_output": "是",
    "metadata": {"model": "glm-5.2"}
}
"""
import os
from collections import defaultdict
from langfuse import RegressionError, RunnerContext, Langfuse, Evaluation
from langfuse.langchain import CallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# --- 初始化 Langfuse ---
langfuse = Langfuse()

prompt_obj = langfuse.get_prompt("sre-check", type="text")
prompt = ChatPromptTemplate.from_template(
    prompt_obj.get_langchain_prompt(),
    metadata={"langfuse_prompt": prompt_obj},
)
langfuse_handler = CallbackHandler()


# --- 模型列表配置 ---
# 格式：逗号分隔的模型名，如 "glm-5.2,deepseek-v3.2,qwen-plus"
EVAL_MODELS = os.environ.get("EVAL_MODELS", "deepseek-v3.2").split(",")


def get_model(model_name):
    """动态创建模型实例"""
    return ChatOpenAI(
        model=model_name.strip(),
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        temperature=0,
        seed=42,
    )


# --- 评估器 ---
def sre_classification_evaluator(*, input, output, expected_output, metadata, **kwargs):
    """
    直接代码判断输出是否正确分类
    """
    try:
        expected = expected_output.strip()
        model_output = output.strip()

        # 检查输出是否包含期望的分类
        if expected == "是":
            is_correct = 1.0 if "是" in model_output and "否" not in model_output else 0.0
        elif expected == "否":
            is_correct = 1.0 if "否" in model_output else 0.0
        else:
            is_correct = 0.0

        model_name = metadata.get("model", "default") if metadata else "default"

        return Evaluation(
            name="sre_classification",
            value=is_correct,
            comment=f"model={model_name}, expected={expected}, output={model_output}",
        )
    except Exception as e:
        return Evaluation(
            name="sre_classification",
            value=0.0,
            comment=f"eval error: {str(e)}",
        )


def average_accuracy_by_model(*, item_results, **kwargs):
    """
    按模型分组计算准确率，并输出对比表格
    """
    # 按模型分组
    model_scores = defaultdict(list)
    for result in item_results:
        for evaluation in result.evaluations:
            if evaluation.name == "sre_classification":
                # 从 comment 中解析 model 名称
                comment = evaluation.comment or ""
                if "model=" in comment:
                    model_name = comment.split("model=")[1].split(",")[0]
                else:
                    model_name = "default"
                model_scores[model_name].append(evaluation.value)

    # 计算每个模型的平均准确率
    model_avgs = {}
    for model_name, scores in model_scores.items():
        avg = sum(scores) / len(scores) if scores else 0
        model_avgs[model_name] = avg

    # 输出对比结果（per-model evaluator 不打印，只返回指标）
    # 最终汇总在 experiment 主函数统一打印

    # 返回所有模型的平均准确率（用于阈值判断）
    all_avgs = list(model_avgs.values())
    overall_avg = sum(all_avgs) / len(all_avgs) if all_avgs else None

    return {"name": "avg_accuracy", "value": overall_avg}


def model_comparison_summary(*, item_results, **kwargs):
    """
    输出每个模型的详细评估结果
    """
    model_scores = defaultdict(list)
    for result in item_results:
        for evaluation in result.evaluations:
            if evaluation.name == "sre_classification":
                comment = evaluation.comment or ""
                if "model=" in comment:
                    model_name = comment.split("model=")[1].split(",")[0]
                else:
                    model_name = "default"
                model_scores[model_name].append(evaluation.value)

    # 按准确率排序
    sorted_models = sorted(model_scores.items(), key=lambda x: -sum(x[1])/len(x[1]) if x[1] else 0)

    best_model = sorted_models[0][0] if sorted_models else None
    best_accuracy = sum(sorted_models[0][1]) / len(sorted_models[0][1]) if sorted_models and sorted_models[0][1] else 0

    print(f"\n🏆 最佳模型: {best_model} (准确率: {best_accuracy:.2%})\n")

    return {"name": "best_model", "value": best_model, "comment": f"accuracy: {best_accuracy:.2%}"}


# --- Experiment entry point (required by experiment-action) ---
def experiment(context: RunnerContext):
    """
    Receives RunnerContext from langfuse/experiment-action.
    Uses context.dataset_name (from workflow `dataset_name` input) automatically.
    """
    # 存储所有模型的评估结果
    all_model_results = {}

    for model_name in EVAL_MODELS:
        model_name = model_name.strip()

        def process_item(*, item, **kwargs):
            model = get_model(model_name)
            llm_app = prompt | model | StrOutputParser()
            return llm_app.invoke(item.input, config={"callbacks": [langfuse_handler]})

        result = context.run_experiment(
            name=f"langfuse-experiment-sre-check-{model_name}",
            description=f"Multi-model comparison for sre-check ({model_name})",
            task=process_item,
            evaluators=[sre_classification_evaluator],
            run_evaluators=[average_accuracy_by_model],
        )

        avg_accuracy = next(
            (e.value for e in result.run_evaluations if e.name == "avg_accuracy"),
            None,
        )
        all_model_results[model_name] = avg_accuracy

    # 输出最终对比
    print(f"\n{'='*50}")
    print(" 多模型准确率对比汇总")
    print(f"{'='*50}")
    for model_name, avg in sorted(all_model_results.items(), key=lambda x: -x[1] if x[1] else 0):
        avg_str = f"{avg:.2%}" if avg is not None else "N/A"
        print(f"  {model_name}: {avg_str}")
    print(f"{'='*50}\n")

    # 用最低准确率的模型做阈值判断
    valid_avgs = [v for v in all_model_results.values() if v is not None]
    worst_avg = min(valid_avgs) if valid_avgs else None

    if worst_avg is not None and worst_avg < 0.8:
        raise RegressionError(
            result=result,
            message=f"最低准确率 {worst_avg:.2%} < 0.8 threshold",
        )

    return result


if __name__ == "__main__":
    # 测试单个 item
    test_input = {"question": "2021_03_04 哪些服务请求较慢，超过500ms？"}
    for model_name in EVAL_MODELS:
        model = get_model(model_name.strip())
        llm_app = prompt | model | StrOutputParser()
        result = llm_app.invoke(test_input, config={"callbacks": [langfuse_handler]})
        print(f"[{model_name.strip()}] Result: {result}")
