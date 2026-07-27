import pytest
from langfuse import Langfuse
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def accuracy_evaluator(*, input, output, expected_output, metadata, **kwargs):
    value = calc_text_similarity(output, expected_output)
    return {
        "name": "accuracy",
        "value": value,
        "comment": "similarity value",
    }

def average_accuracy(*, item_results, **kwargs):
    """Calculate average accuracy across all items"""
    accuracies = [
        evaluation.value for result in item_results
        for evaluation in result.evaluations
        if evaluation.name == "accuracy"
    ]
    if not accuracies:
        return {
            "name": "avg_accuracy",
            "value": None,
        }
    avg = sum(accuracies) / len(accuracies)
    return {
        "name": "avg_accuracy",
        "value": avg,
        "comment": f"Average accuracy: {avg:.2%}",
    }


def run_experiment(langfuse_client, chain, dataset_name):
    """自定义过程：运行llm并评估结果"""
    # 1. 从LangFuse获取指定的数据集
    dataset = langfuse_client.get_dataset(dataset_name)

    # 2.定义一个处理器，用来处理数据集的每一项，处理逻辑是调用calc_text_similarity来计算精度
    def process_item(*, item, **kwargs):
        output = chain.invoke(item.input)
        return output

    # 单线程顺序处理
    result = dataset.run_experiment(
        name="Charles-Demo1",
        description="Charles-Demo1",
        task=process_item,  # see above for the task definition
        evaluators=[accuracy_evaluator],
        run_evaluators=[average_accuracy],
    )
    # Use format method to display results
    print(result.format())
    # process_item(item)

      # Access the run evaluator result directly
    avg_accuracy_value = next(
        (
            evaluation.value
            for evaluation in result.run_evaluations
            if evaluation.name == "average_accuracy"
        ),
        None,
    )
    return avg_accuracy_value



# 基于提示词模板构建提示词
prompt = PromptTemplate.from_template("""
*********
你是一位Mr Know All先生，世界万物的知识你无所不知。
问个问题:{input}
*********""")

# 待测试的模型
model = ChatOpenAI(
    model="qwen-turbo",
    api_key="sk-ws-H.EMHMELD.LKzU.MEUCIEgTw-viM44ZPCbWXa54wZKzIRijTPumuAhFlZQtTTgQAiEAgkY3x_K42UzLqFVdX7kxI9CC9tcxMs9c8Hqo0p-Sumk",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    seed=42
)

# 给出输出解析器，这里是StrOutputParser，它会把AIMessage转为str
parser = StrOutputParser()

# 创建一个langchain链，它会提示词+模型+解析器  结合起来
llm_application = (
    prompt
    | model
    | parser
)

def calc_text_similarity(text_a, text_b):
    """给定2个文本text_a, text_b,计算余弦相似度"""
    # 将文本转为向量
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text_a, text_b])
    # 计算余弦相似度
    similarity = cosine_similarity(vectors)[0][1]
    return similarity


@pytest.fixture
def langfuse_client():
    """Initialize Langfuse client for testing"""
    langfuse = Langfuse()
    return langfuse


def test_accuracy_fails(langfuse_client):
    dataset_name = "Charles-Demo1"
    avg_accuracy_value = run_experiment(langfuse_client, llm_application, dataset_name)
    # This test will fail because the task gives wrong answers
    with pytest.raises(AssertionError):
        assert isinstance(avg_accuracy_value, (int, float)) and avg_accuracy_value >= 0.8, (
            f"Expected test to fail with low accuracy: {avg_accuracy_value}"
        )
