from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# llm = ChatOpenAI(
#     base_url="http://sionic.chat:8001/v1",
#     api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
#     model="xionic-ko-llama-3-70b",
# )
# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
# llm = ChatOllama(model="llama3:70b-instruct")#Llama-3-Open-Ko-8B-Instruct-preview-Q8_0:latest
# import transformers
# import torch

# model_id = "meta-llama/Meta-Llama-3-8B"

# pipeline = transformers.pipeline(
#     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# )
# print(pipeline("Hey how are you doing today?"))

llm = ChatOllama(model="llama3", stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token", "assistant"])
# Prompt 설정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, smart, kind, and efficient AI assistant named '인덕이'. Translate the answer into Korean. You must answer in Korean. You always fulfill the user's requests to the best of your ability. You must generate an answer in Korean.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm | StrOutputParser()
