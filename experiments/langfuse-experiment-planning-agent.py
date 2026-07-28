"""
Langfuse experiment gate for the planning-agent prompt.

Expected by experiment-action: defines experiment(context: RunnerContext).
Dataset item shape after CSV import:
{
    "input": {
        "input_round": "middle",
        "input_question": "...",
        "input_executed_tasks": "..."
    },
    "expected_output": {
        "expected_stage": "...",
        "must_cover": "...",
        "must_not_have": "...",
        "expected_agents": "...",
        "parallel_expectation": "..."
    },
    "metadata": {
        "case_type": "...",
        "difficulty": "...",
        "focus": "..."
    }
}
"""
import json
import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langfuse import Evaluation, Langfuse, RegressionError, RunnerContext
from langfuse.langchain import CallbackHandler


langfuse = Langfuse()

planning_prompt_obj = langfuse.get_prompt(
    "planning-agent",
    type="text",
    label="latest",
    cache_ttl_seconds=0,
)
planning_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=planning_prompt_obj.get_langchain_prompt()),
        ("user", "{planning_user_message}"),
    ]
)
planning_prompt.metadata = {"langfuse_prompt": planning_prompt_obj}

langfuse_handler = CallbackHandler()
_judge_cache = {}


def get_model(model_name, temperature=0):
    return ChatOpenAI(
        model=model_name.strip(),
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=os.environ.get(
            "OPENAI_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=temperature,
        seed=42,
    )


def format_planning_user_message(item_input):
    question = item_input.get("input_question", "")
    executed_tasks = item_input.get("input_executed_tasks", "")
    round_name = item_input.get("input_round", "")

    return "\n".join(
        [
            f"当前轮次：{round_name}",
            f"用户问题：{question}",
            f"已执行任务和任务结果：{executed_tasks or '无'}",
        ]
    )


def extract_json_object(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("judge output does not contain a JSON object")

    return json.loads(match.group(0))


def clamp_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, score))


def judge_planning_output(input, output, expected_output, metadata):
    cache_key = json.dumps(
        {
            "input": input,
            "output": output,
            "expected_output": expected_output,
            "metadata": metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]

    judge_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 planning-agent prompt 的严格评估器。"
                "请只输出 JSON，不要输出解释性前后缀。"
                "所有分数必须是 0 到 1 的数字。",
            ),
            (
                "user",
                """
请评估 planning-agent 的输出是否满足故障诊断规划要求。

评分维度：
1. planning_reasonableness：规划合理性。检查是否遵循预处理 -> 异常检测 -> 故障识别 -> 根因定位，以及阈值计算 -> 数据提取 -> 指标分析 -> 轨迹分析 -> 日志分析；是否避免过早调用 report_assistant；是否基于已执行任务继续推理；证据充分时是否停止。
2. task_decomposition_quality：任务分解质量。检查 task 是否原子、明确、可执行；agent 是否正确；parallel/depends_on 是否合理；slots 是否包含关键参数；每轮是否不超过 10 个步骤；是否避免让 planning_agent 自己写代码、画图、保存文件或跑 bash。
3. format_compliance：格式合规性。检查是否为合法 JSON，是否包含 analysis、plans、plans_brief_description；plans 内是否包含 task_id、parent_id、task_goal、agent、info_sufficient_for_report、parallel、depends_on、slots。

数据集输入：
{input_json}

期望输出约束：
{expected_json}

metadata：
{metadata_json}

planning-agent 实际输出：
{actual_output}

请输出如下 JSON：
{{
  "planning_reasonableness": 0.0,
  "task_decomposition_quality": 0.0,
  "format_compliance": 0.0,
  "overall": 0.0,
  "pass": false,
  "reason": "不超过80字"
}}
""",
            ),
        ]
    )

    judge_model_name = os.environ.get("PLANNING_JUDGE_MODEL", "qwen-plus")
    judge_chain = judge_prompt | get_model(judge_model_name) | StrOutputParser()
    raw_judge_output = judge_chain.invoke(
        {
            "input_json": json.dumps(input, ensure_ascii=False),
            "expected_json": json.dumps(expected_output, ensure_ascii=False),
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
            "actual_output": output,
        }
    )

    try:
        parsed = extract_json_object(raw_judge_output)
    except Exception as exc:
        parsed = {
            "planning_reasonableness": 0.0,
            "task_decomposition_quality": 0.0,
            "format_compliance": 0.0,
            "overall": 0.0,
            "pass": False,
            "reason": f"judge parse error: {exc}",
        }

    planning_reasonableness = clamp_score(parsed.get("planning_reasonableness"))
    task_decomposition_quality = clamp_score(parsed.get("task_decomposition_quality"))
    format_compliance = clamp_score(parsed.get("format_compliance"))
    overall = clamp_score(
        parsed.get(
            "overall",
            0.45 * planning_reasonableness
            + 0.45 * task_decomposition_quality
            + 0.10 * format_compliance,
        )
    )

    result = {
        "planning_reasonableness": planning_reasonableness,
        "task_decomposition_quality": task_decomposition_quality,
        "format_compliance": format_compliance,
        "overall": overall,
        "pass": bool(parsed.get("pass", overall >= 0.75)),
        "reason": str(parsed.get("reason", ""))[:300],
    }
    _judge_cache[cache_key] = result
    return result


def planning_reasonableness_evaluator(
    *, input, output, expected_output, metadata, **kwargs
):
    result = judge_planning_output(input, output, expected_output, metadata)
    return Evaluation(
        name="planning_reasonableness",
        value=result["planning_reasonableness"],
        comment=result["reason"],
    )


def task_decomposition_quality_evaluator(
    *, input, output, expected_output, metadata, **kwargs
):
    result = judge_planning_output(input, output, expected_output, metadata)
    return Evaluation(
        name="task_decomposition_quality",
        value=result["task_decomposition_quality"],
        comment=result["reason"],
    )


def format_compliance_evaluator(*, input, output, expected_output, metadata, **kwargs):
    result = judge_planning_output(input, output, expected_output, metadata)
    return Evaluation(
        name="format_compliance",
        value=result["format_compliance"],
        comment=result["reason"],
    )


def planning_overall_evaluator(*, input, output, expected_output, metadata, **kwargs):
    result = judge_planning_output(input, output, expected_output, metadata)
    return Evaluation(
        name="planning_overall",
        value=result["overall"],
        comment=f"pass={result['pass']}; {result['reason']}",
    )


def average_score(item_results, score_name):
    scores = [
        evaluation.value
        for item_result in item_results
        for evaluation in item_result.evaluations
        if evaluation.name == score_name and evaluation.value is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def average_planning_reasonableness(*, item_results, **kwargs):
    value = average_score(item_results, "planning_reasonableness")
    return Evaluation(
        name="planning_reasonableness_avg",
        value=value,
        comment=f"Average planning reasonableness: {value}",
    )


def average_task_decomposition_quality(*, item_results, **kwargs):
    value = average_score(item_results, "task_decomposition_quality")
    return Evaluation(
        name="task_decomposition_quality_avg",
        value=value,
        comment=f"Average task decomposition quality: {value}",
    )


def average_format_compliance(*, item_results, **kwargs):
    value = average_score(item_results, "format_compliance")
    return Evaluation(
        name="format_compliance_avg",
        value=value,
        comment=f"Average format compliance: {value}",
    )


def average_planning_overall(*, item_results, **kwargs):
    value = average_score(item_results, "planning_overall")
    return Evaluation(
        name="planning_overall_avg",
        value=value,
        comment=f"Average planning overall: {value}",
    )


def experiment(context: RunnerContext):
    model_name = os.environ.get("PLANNING_AGENT_MODEL", "deepseek-v3.2")

    def process_item(*, item, **kwargs):
        planning_chain = planning_prompt | get_model(model_name) | StrOutputParser()
        return planning_chain.invoke(
            {"planning_user_message": format_planning_user_message(item.input)},
            config={"callbacks": [langfuse_handler]},
        )

    result = context.run_experiment(
        name=f"langfuse-experiment-planning-agent-{model_name}",
        description="Evaluate planning-agent prompt for incident diagnosis planning.",
        task=process_item,
        evaluators=[
            planning_reasonableness_evaluator,
            task_decomposition_quality_evaluator,
            format_compliance_evaluator,
            planning_overall_evaluator,
        ],
        run_evaluators=[
            average_planning_reasonableness,
            average_task_decomposition_quality,
            average_format_compliance,
            average_planning_overall,
        ],
    )

    run_scores = {
        evaluation.name: evaluation.value for evaluation in result.run_evaluations
    }

    planning_avg = run_scores.get("planning_reasonableness_avg")
    decomposition_avg = run_scores.get("task_decomposition_quality_avg")
    overall_avg = run_scores.get("planning_overall_avg")

    print(
        "\n"
        f"planning-agent planning avg: {planning_avg}\n"
        f"planning-agent decomposition avg: {decomposition_avg}\n"
        f"planning-agent overall avg: {overall_avg}\n"
    )

    threshold = float(os.environ.get("PLANNING_OVERALL_THRESHOLD", "0.75"))
    failed_scores = {
        "planning_reasonableness_avg": planning_avg,
        "task_decomposition_quality_avg": decomposition_avg,
        "planning_overall_avg": overall_avg,
    }
    failed_scores = {
        name: value
        for name, value in failed_scores.items()
        if value is not None and value < threshold
    }

    if failed_scores:
        raise RegressionError(
            result=result,
            message=f"planning-agent scores below {threshold:.0%}: {failed_scores}",
        )

    return result


if __name__ == "__main__":
    sample_input = {
        "input_round": "middle",
        "input_question": "2021_03_06 交易链路超时，请继续分析根因组件。",
        "input_executed_tasks": "已执行任务：指标分析确认 Tomcat01、Tomcat03、Mysql01 在同一时间窗出现连续异常。尚未分析 trace。",
    }
    planning_chain = planning_prompt | get_model(
        os.environ.get("PLANNING_AGENT_MODEL", "deepseek-v3.2")
    ) | StrOutputParser()
    print(
        planning_chain.invoke(
            {"planning_user_message": format_planning_user_message(sample_input)},
            config={"callbacks": [langfuse_handler]},
        )
    )
