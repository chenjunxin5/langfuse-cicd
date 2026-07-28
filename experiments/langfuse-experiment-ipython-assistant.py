"""
Langfuse experiment gate for the ipython-assistant prompt.

Expected by experiment-action: defines experiment(context: RunnerContext).
Dataset item shape is intentionally flexible. Recommended shape:
{
    "input": {
        "question": "...",
        "task_goal": "...",
        "context": "..."
    },
    "expected_output": {
        "must_cover": "...",
        "must_not_have": "..."
    },
    "metadata": {
        "case_type": "...",
        "difficulty": "..."
    }
}
"""
import ast
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langfuse import Evaluation, Langfuse, RegressionError, RunnerContext
from langfuse.langchain import CallbackHandler


langfuse = Langfuse()

ipython_prompt_obj = langfuse.get_prompt(
    "ipython-assistant-chat",
    type="chat",
    label="latest",
    cache_ttl_seconds=0,
)
ipython_prompt = ChatPromptTemplate.from_messages(
    ipython_prompt_obj.get_langchain_prompt()
)
ipython_prompt.metadata = {"langfuse_prompt": ipython_prompt_obj}

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


def format_ipython_user_message(item_input):
    if not isinstance(item_input, dict):
        return str(item_input)

    ordered_keys = [
        "question",
        "task_goal",
        "context",
        "input_round",
        "executed_tasks",
        "input_executed_tasks",
    ]
    lines = []
    for key in ordered_keys:
        value = item_input.get(key)
        if value:
            lines.append(f"{key}: {value}")

    remaining = {
        key: value
        for key, value in item_input.items()
        if key not in ordered_keys and value not in (None, "")
    }
    if remaining:
        lines.append(
            "additional_input: "
            + json.dumps(remaining, ensure_ascii=False, sort_keys=True)
        )

    return "\n".join(lines) if lines else json.dumps(item_input, ensure_ascii=False)


def build_prompt_variables(item_input):
    variables = ipython_prompt.input_variables
    if not variables:
        return {}

    if isinstance(item_input, dict):
        values = {
            variable: item_input.get(variable, "")
            for variable in variables
        }
    else:
        values = {variable: "" for variable in variables}

    if len(variables) == 1:
        values[variables[0]] = format_ipython_user_message(item_input)
    elif "question" in values and not values["question"]:
        values["question"] = format_ipython_user_message(item_input)

    return values


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


def extract_python_code(output):
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", output, re.DOTALL)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks if block.strip()).strip()

    return output.strip()


def analyze_code(code):
    result = {
        "has_code": bool(code),
        "syntax_success": False,
        "execution_success": False,
        "safety_score": 1.0,
        "issues": [],
    }
    if not code:
        result["issues"].append("no python code found")
        result["safety_score"] = 0.0
        return result

    try:
        tree = ast.parse(code)
        compile(tree, "<ipython-assistant-output>", "exec")
        result["syntax_success"] = True
    except SyntaxError as exc:
        result["issues"].append(f"syntax error: {exc}")
        result["safety_score"] = 0.0
        return result

    unsafe_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = get_call_name(node.func)
            if name in {"os.system", "subprocess.run", "subprocess.Popen"}:
                unsafe_calls.append(name)
            if name in {"eval", "exec", "compile", "__import__"}:
                unsafe_calls.append(name)
            if name in {"open", "Path.open"} and opens_for_write(node):
                unsafe_calls.append("write_file")
            if name.endswith(".write_text") or name.endswith(".write_bytes"):
                unsafe_calls.append("write_file")
            if name.endswith(".to_csv") or name.endswith(".to_excel"):
                unsafe_calls.append("save_dataframe")

    if unsafe_calls:
        result["issues"].append("unsafe calls: " + ", ".join(sorted(set(unsafe_calls))))
        result["safety_score"] = 0.0

    if os.environ.get("IPYTHON_ASSISTANT_ENABLE_EXECUTION") == "true":
        result["execution_success"] = execute_code(code, result["issues"])
    else:
        result["execution_success"] = result["syntax_success"]

    return result


def get_call_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = get_call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return ""


def opens_for_write(node):
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        return has_write_mode(node.args[1].value)
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return has_write_mode(keyword.value.value)
    return False


def has_write_mode(mode):
    return isinstance(mode, str) and any(flag in mode for flag in ("w", "a", "x", "+"))


def execute_code(code, issues):
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "generated.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                ["python", str(script_path)],
                cwd=tmp_dir,
                text=True,
                capture_output=True,
                timeout=int(os.environ.get("IPYTHON_ASSISTANT_EXECUTION_TIMEOUT", "30")),
                check=False,
            )
        except Exception as exc:
            issues.append(f"execution error: {exc}")
            return False

    if completed.returncode != 0:
        stderr = completed.stderr.strip().splitlines()
        issues.append("execution failed: " + (stderr[-1] if stderr else "non-zero exit"))
        return False
    return True


def judge_ipython_output(input, output, expected_output, metadata, code, code_analysis):
    """对单个样本调用一次 LLM judge，评估代码是否满足任务要求。"""
    cache_key = json.dumps(
        {
            "input": input,
            "output": output,
            "expected_output": expected_output,
            "metadata": metadata,
            "code_analysis": code_analysis,
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
                "你是 ipython_assistant 输出的严格评估器。"
                "请只输出 JSON，不要输出解释性前后缀。"
                "所有分数必须是 0 到 1 的数字。",
            ),
            (
                "user",
                """
请评估 ipython_assistant 的输出是否满足代码代理要求。

评分维度：
1. code_correctness：代码正确性。检查代码是否能解决任务、处理数据路径/字段/KPI/时间窗等输入，是否避免臆造结论。
2. execution_success：执行成功率。优先参考静态语法和 dry-run 结果；如果启用了真实执行，则参考真实执行结果。
3. task_fulfillment：任务完成度。检查是否覆盖 expected_output 中的 must_cover，是否没有违反 must_not_have。
4. safety_compliance：安全合规。检查是否没有 bash、危险系统调用、本地文件写入、绘图保存等不允许操作。

数据集输入：
{input_json}

期望输出约束：
{expected_json}

metadata：
{metadata_json}

静态执行分析：
{code_analysis_json}

提取出的 Python 代码：
{code}

ipython_assistant 原始输出：
{actual_output}

请输出如下 JSON：
{{
  "code_correctness": 0.0,
  "execution_success": 0.0,
  "task_fulfillment": 0.0,
  "safety_compliance": 0.0,
  "overall": 0.0,
  "pass": false,
  "reason": "不超过80字"
}}
""",
            ),
        ]
    )

    judge_model_name = os.environ.get("IPYTHON_ASSISTANT_JUDGE_MODEL", "qwen-plus")
    judge_chain = judge_prompt | get_model(judge_model_name) | StrOutputParser()
    raw_judge_output = judge_chain.invoke(
        {
            "input_json": json.dumps(input, ensure_ascii=False),
            "expected_json": json.dumps(expected_output, ensure_ascii=False),
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
            "code_analysis_json": json.dumps(code_analysis, ensure_ascii=False),
            "code": code,
            "actual_output": output,
        }
    )

    try:
        parsed = extract_json_object(raw_judge_output)
    except Exception as exc:
        parsed = {
            "code_correctness": 0.0,
            "execution_success": 0.0,
            "task_fulfillment": 0.0,
            "safety_compliance": 0.0,
            "overall": 0.0,
            "pass": False,
            "reason": f"judge parse error: {exc}",
        }

    code_correctness = clamp_score(parsed.get("code_correctness"))
    execution_success = clamp_score(
        parsed.get("execution_success", 1.0 if code_analysis["execution_success"] else 0.0)
    )
    task_fulfillment = clamp_score(parsed.get("task_fulfillment"))
    safety_compliance = min(
        clamp_score(parsed.get("safety_compliance", code_analysis["safety_score"])),
        code_analysis["safety_score"],
    )
    overall = clamp_score(
        parsed.get(
            "overall",
            0.40 * code_correctness
            + 0.25 * execution_success
            + 0.25 * task_fulfillment
            + 0.10 * safety_compliance,
        )
    )

    result = {
        "code_correctness": code_correctness,
        "execution_success": execution_success,
        "task_fulfillment": task_fulfillment,
        "safety_compliance": safety_compliance,
        "overall": overall,
        "pass": bool(parsed.get("pass", overall >= 0.75)),
        "reason": str(parsed.get("reason", ""))[:300],
    }
    _judge_cache[cache_key] = result
    return result


def evaluate_item(input, output, expected_output, metadata):
    code = extract_python_code(output)
    code_analysis = analyze_code(code)
    judge_result = judge_ipython_output(
        input,
        output,
        expected_output,
        metadata,
        code,
        code_analysis,
    )
    return code_analysis, judge_result


def code_correctness_evaluator(*, input, output, expected_output, metadata, **kwargs):
    """单样本评估器：评估生成代码是否能正确解决任务。"""
    _, result = evaluate_item(input, output, expected_output, metadata)
    return Evaluation(
        name="code_correctness",
        value=result["code_correctness"],
        comment=result["reason"],
    )


def execution_success_evaluator(*, input, output, expected_output, metadata, **kwargs):
    """单样本评估器：评估代码是否可提取、可编译，必要时可真实执行。"""
    code_analysis, result = evaluate_item(input, output, expected_output, metadata)
    issues = "; ".join(code_analysis["issues"])
    return Evaluation(
        name="execution_success",
        value=result["execution_success"],
        comment=issues or result["reason"],
    )


def task_fulfillment_evaluator(*, input, output, expected_output, metadata, **kwargs):
    """单样本评估器：评估输出是否覆盖期望约束并完成任务。"""
    _, result = evaluate_item(input, output, expected_output, metadata)
    return Evaluation(
        name="task_fulfillment",
        value=result["task_fulfillment"],
        comment=result["reason"],
    )


def safety_compliance_evaluator(*, input, output, expected_output, metadata, **kwargs):
    """单样本评估器：评估代码是否避免危险调用和本地写文件。"""
    code_analysis, result = evaluate_item(input, output, expected_output, metadata)
    issues = "; ".join(code_analysis["issues"])
    return Evaluation(
        name="safety_compliance",
        value=result["safety_compliance"],
        comment=issues or result["reason"],
    )


def ipython_overall_evaluator(*, input, output, expected_output, metadata, **kwargs):
    """单样本评估器：输出用于准入判断的综合评分。"""
    _, result = evaluate_item(input, output, expected_output, metadata)
    return Evaluation(
        name="ipython_overall",
        value=result["overall"],
        comment=f"pass={result['pass']}; {result['reason']}",
    )


def average_score(item_results, score_name):
    """Run 级公共方法：计算某个单样本评分在整个数据集上的平均值。"""
    scores = [
        evaluation.value
        for item_result in item_results
        for evaluation in item_result.evaluations
        if evaluation.name == score_name and evaluation.value is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def average_code_correctness(*, item_results, **kwargs):
    """Run 级评估器：计算全量样本的代码正确性平均分。"""
    value = average_score(item_results, "code_correctness")
    return Evaluation(
        name="code_correctness_avg",
        value=value,
        comment=f"Average code correctness: {value}",
    )


def average_execution_success(*, item_results, **kwargs):
    """Run 级评估器：计算全量样本的执行成功率平均分。"""
    value = average_score(item_results, "execution_success")
    return Evaluation(
        name="execution_success_avg",
        value=value,
        comment=f"Average execution success: {value}",
    )


def average_task_fulfillment(*, item_results, **kwargs):
    """Run 级评估器：计算全量样本的任务完成度平均分。"""
    value = average_score(item_results, "task_fulfillment")
    return Evaluation(
        name="task_fulfillment_avg",
        value=value,
        comment=f"Average task fulfillment: {value}",
    )


def average_safety_compliance(*, item_results, **kwargs):
    """Run 级评估器：计算全量样本的安全合规平均分。"""
    value = average_score(item_results, "safety_compliance")
    return Evaluation(
        name="safety_compliance_avg",
        value=value,
        comment=f"Average safety compliance: {value}",
    )


def average_ipython_overall(*, item_results, **kwargs):
    """Run 级评估器：计算用于回归阈值判断的综合平均分。"""
    value = average_score(item_results, "ipython_overall")
    return Evaluation(
        name="ipython_overall_avg",
        value=value,
        comment=f"Average ipython overall: {value}",
    )


def experiment(context: RunnerContext):
    model_name = os.environ.get("IPYTHON_ASSISTANT_MODEL", "deepseek-v3.2")

    def process_item(*, item, **kwargs):
        chain = ipython_prompt | get_model(model_name) | StrOutputParser()
        return chain.invoke(
            build_prompt_variables(item.input),
            config={"callbacks": [langfuse_handler]},
        )

    result = context.run_experiment(
        name=f"langfuse-experiment-ipython-assistant-{model_name}",
        description="Evaluate ipython-assistant prompt for code correctness and execution readiness.",
        task=process_item,
        evaluators=[
            code_correctness_evaluator,
            execution_success_evaluator,
            task_fulfillment_evaluator,
            safety_compliance_evaluator,
            ipython_overall_evaluator,
        ],
        run_evaluators=[
            average_code_correctness,
            average_execution_success,
            average_task_fulfillment,
            average_safety_compliance,
            average_ipython_overall,
        ],
    )

    run_scores = {
        evaluation.name: evaluation.value for evaluation in result.run_evaluations
    }

    correctness_avg = run_scores.get("code_correctness_avg")
    execution_avg = run_scores.get("execution_success_avg")
    fulfillment_avg = run_scores.get("task_fulfillment_avg")
    overall_avg = run_scores.get("ipython_overall_avg")

    print(
        "\n"
        f"ipython-assistant correctness avg: {correctness_avg}\n"
        f"ipython-assistant execution avg: {execution_avg}\n"
        f"ipython-assistant fulfillment avg: {fulfillment_avg}\n"
        f"ipython-assistant overall avg: {overall_avg}\n"
    )

    threshold = float(os.environ.get("IPYTHON_ASSISTANT_OVERALL_THRESHOLD", "0.75"))
    failed_scores = {
        "code_correctness_avg": correctness_avg,
        "execution_success_avg": execution_avg,
        "task_fulfillment_avg": fulfillment_avg,
        "ipython_overall_avg": overall_avg,
    }
    failed_scores = {
        name: value
        for name, value in failed_scores.items()
        if value is not None and value < threshold
    }

    if failed_scores:
        raise RegressionError(
            result=result,
            message=f"ipython-assistant scores below {threshold:.0%}: {failed_scores}",
        )

    return result


if __name__ == "__main__":
    sample_input = {
        "question": "2021_03_06 交易链路超时，请分析 trace_span.csv 中候选故障组件的上下游关系。",
        "context": "候选故障组件：Tomcat01、Tomcat03、Mysql01；目标时间窗：已识别连续异常时间窗。",
    }
    chain = ipython_prompt | get_model(
        os.environ.get("IPYTHON_ASSISTANT_MODEL", "deepseek-v3.2")
    ) | StrOutputParser()
    print(chain.invoke(build_prompt_variables(sample_input)))
