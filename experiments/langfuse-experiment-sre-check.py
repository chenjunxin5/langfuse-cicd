"""
Langfuse experiment gate — 使用 LLM-as-Judge 评估 sre-check 二分类任务
Expected by experiment-action: defines experiment(context: RunnerContext).
"""
import os
from langfuse import RegressionError, RunnerContext, Langfuse, Evaluation
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# --- 初始化 Langfuse ---
# 本地运行时设置环境变量：LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
langfuse = Langfuse(
    public_key="pk-lf-d2f237a1-0533-44c0-a8fc-8bd8e79db7d5",
    secret_key="sk-lf-7cce40b5-e96a-4d20-bc5c-bcd8449d70cf",
    host="https://jp.cloud.langfuse.com",
)

prompt_obj = langfuse.get_prompt("sre-check")
prompt_text = prompt_obj.prompt

prompt = PromptTemplate.from_template(prompt_text)


# --- LLM chain ---
api_key = os.environ.get("dashscope_api_key","sk-ws-H.EMHMELD.LKzU.MEUCIEgTw-viM44ZPCbWXa54wZKzIRijTPumuAhFlZQtTTgQAiEAgkY3x_K42UzLqFVdX7kxI9CC9tcxMs9c8Hqo0p-Sumk")
model = ChatOpenAI(
    model=os.environ.get("LLM_MODEL", "deepseek-v3.2"),
    api_key=api_key,
    base_url=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0,
    seed=42,
)

parser = StrOutputParser()
llm_application = prompt | model | parser


# --- 评估器 ---
def sre_classification_evaluator(*, input, output, expected_output, **kwargs):
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

        return Evaluation(
            name="sre_classification",
            value=is_correct,
            comment=f"expected={expected}, output={model_output}",
        )
    except Exception as e:
        return Evaluation(
            name="sre_classification",
            value=0.0,
            comment=f"eval error: {str(e)}",
        )


def average_classification_accuracy(*, item_results, **kwargs):
    """
    计算分类准确率
    """
    accuracies = [
        evaluation.value
        for result in item_results
        for evaluation in result.evaluations
        if evaluation.name == "sre_classification"
    ]
    if not accuracies:
        return {"name": "avg_accuracy", "value": None}
    avg = sum(accuracies) / len(accuracies)
    return {"name": "avg_accuracy", "value": avg}


# --- Experiment entry point (required by experiment-action) ---
def experiment(context: RunnerContext):
    """
    Receives RunnerContext from langfuse/experiment-action.
    Uses context.dataset_name (from workflow `dataset_name` input) automatically.
    """
    def process_item(*, item, **kwargs):
        return llm_application.invoke(item.input)

    result = context.run_experiment(
        name="langfuse-experiment-sre-check",
        description="LLM-as-Judge evaluation for sre-check classification",
        task=process_item,
        evaluators=[sre_classification_evaluator],
        run_evaluators=[average_classification_accuracy],
    )

    avg_accuracy = next(
        (e.value for e in result.run_evaluations if e.name == "avg_accuracy"),
        None,
    )

    # Threshold gate — fail CI if accuracy drops below 0.85
    if avg_accuracy is not None and avg_accuracy < 0.85:
        raise RegressionError(
            result=result,
            message=f"avg_accuracy {avg_accuracy:.2%} < 0.85 threshold",
        )

    return result


if __name__ == "__main__":
    result = llm_application.invoke({"question": "2021_03_04 哪些服务请求较慢，超过500ms？"})
    print(f"Result: {result}")
