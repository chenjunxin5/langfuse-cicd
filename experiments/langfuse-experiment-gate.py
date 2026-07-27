"""
Langfuse experiment gate — runs evaluation experiments on pull requests.
Expected by experiment-action: defines experiment(context: RunnerContext).
"""
import os
from langfuse import RegressionError, RunnerContext
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Evalutors ---

def accuracy_evaluator(*, input, output, expected_output, metadata, **kwargs):
    value = calc_text_similarity(output, expected_output)
    return {"name": "accuracy", "value": value, "comment": "cosine similarity"}


def average_accuracy(*, item_results, **kwargs):
    accuracies = [
        evaluation.value
        for result in item_results
        for evaluation in result.evaluations
        if evaluation.name == "accuracy"
    ]
    if not accuracies:
        return {"name": "avg_accuracy", "value": None}
    avg = sum(accuracies) / len(accuracies)
    return {"name": "avg_accuracy", "value": avg, "comment": f"Average: {avg:.2%}"}


# --- LLM chain ---

prompt = PromptTemplate.from_template("""*********
你是一位Mr Know All先生，世界万物的知识你无所不知。
问个问题:{input}
*********""")

model = ChatOpenAI(
    model=os.environ.get("LLM_MODEL", "qwen-turbo"),
    api_key=os.environ.get("DASHSCOPE_API_KEY"),       # set via env secret in CI
    base_url=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0,
    seed=42,
)

parser = StrOutputParser()

llm_application = prompt | model | parser


def calc_text_similarity(text_a, text_b):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text_a, text_b])
    return cosine_similarity(vectors)[0][1]


# --- Experiment entry point (required by experiment-action) ---

def experiment(context: RunnerContext):
    """
    Receives RunnerContext from langfuse/experiment-action.
    Uses context.dataset_name (from workflow `dataset_name` input) automatically.
    """
    def process_item(*, item, **kwargs):
        return llm_application.invoke(item.input)

    result = context.run_experiment(
        name="langfuse-experiment-gate",
        description="Langfuse experiment evaluation gate",
        task=process_item,
        evaluators=[accuracy_evaluator],
        run_evaluators=[average_accuracy],
    )

    avg_accuracy = next(
        (e.value for e in result.run_evaluations if e.name == "avg_accuracy"),
        None,
    )

    # Threshold gate — fail CI if accuracy drops below 0.8
    if avg_accuracy is not None and avg_accuracy < 0.8:
        raise RegressionError(
            result=result,
            message=f"avg_accuracy {avg_accuracy:.2%} < 0.8 threshold",
        )

    return result

