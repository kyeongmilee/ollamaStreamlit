#import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
# Load model directly
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.retrievers import BM25Retriever, EnsembleRetriever # pip install rank_bm25
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
import torch
import os
from langchain.retrievers.multi_query import MultiQueryRetriever
import time
from langchain.storage import InMemoryByteStore
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import math
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
import time


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    llm = HuggingFacePipeline.from_model_id(
        model_id="megastudy/M-SOLAR-10.7B-v1.3", 
        device=0,               # -1: CPU(default), GPU존재하는 경우는 0번이상(CUDA Device #번호)
        task="text-generation", # 텍스트 생성
        model_kwargs={"temperature": 0.1, # 생성된 텍스트의 다양성 조정
                    "max_length": 128},# 생성할 최대 토큰 수   
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#"cocoirun/AIFT-42dot-PLM-1.3B-ao-instruct-all-v0.4-ff-e1"
#"sue3489/test0_kullm-polyglot-5.8b-v2-koalpaca-v1.1b"#너무 느림
#"mu0gum/AIFT-42dot_LLM-PLM-1.3B-ao-instruct-all-v1.3"#답변이 별로야
#"sue3489/test0_kullm-polyglot-5.8b-v2-koalpaca-v1.1b" #추론시 토큰권한 문제 발생
#"Xwin-LM/Xwin-LM-7B-V0.2" #토크나이저 에러
#"julleong/illuni-llama-2-ko-7b-test" #1분 이상 소요
#"AIFT/AIFT-instruct-SFT-1.3B-refine-v3"#24초 소요 답변 길어 
#"mu0gum/AIFT-42dot_LLM-PLM-1.3B-ao-instruct-all-v1.3" #현재까지 젤 나이스
#"jungyuko/DAVinCI-42dot_LLM-PLM-1.3B-v1.5.3"# 23초 엉뚱답
#"Changgil/K2S3-SOLAR-11b-v2.0" #응답이 안옴
#"cocoirun/AIFT-42dot-PLM-1.3B-ao-instruct-all-v0.4-ff-e1"#베리굿
#"Changgil/K2S3-Llama2-13b-v1.0" #메모리 아웃 
#"momo/polyglot-ko-12.8b-Chat-QLoRA-Merge" # 메모리 아웃 47.53
#"gwonny/llama-2-koen-13b-QLoRA-NEFTune-kolon-v0.1"메모리 47.53.#"mu0gum/AIFT-polyglot-ko-1.3b-ao-instruct-v0.91"#"nlpai-lab/kullm-polyglot-5.8b-v2"#"beomi/open-llama-2-ko-7b"#"Changgil/K2S3-Llama2-13b-v1.0"#"KRAFTON/KORani-v2-13B"#"beomi/open-llama-2-ko-7b"#"yanolja/KoSOLAR-10.7B-v0.2"#"mu0gum/AIFT-polyglot-ko-1.3b-ao-instruct-v0.91"
model_path = "./"+model_id
tokenizer_path = "./"+model_id+"-tokenizer"

def settingModel() :
    global model
    global tokenizer
    global device

    ##캐쉬 비우기
    torch.cuda.empty_cache()

    ## 모델 구동 환경 설정
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    
    # 모델이 로컬에 있는지 확인
    if not os.path.exists(os.path.join(model_path, "config.json")):
        # 로컬에 없으면 다운로드
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": 0}, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        # 로컬에 저장
        print('모델 저장 시작')
        model.save_pretrained(model_path)
    else:
        # 로컬에 있으면 해당 경로에서 로드
        print('모델 로드 시작')
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": 0}, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    model.to(device)
    print('모델 로드 완료')
    
    # 토크나이저가 로컬에 있는지 확인
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
        # 로컬에 없으면 다운로드
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 로컬에 저장
        tokenizer.save_pretrained(tokenizer_path)
    else:
        # 로컬에 있으면 해당 경로에서 로드
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print("Model and Tokenizer loaded")
    # 모델을 추론에만 사용할 것이므로, 'eval'모드로 변경
    model.eval()
    
    # 정보
    #class_index = json.load(open('../path/to/'))

    model.config.use_cache = True
    tokenizer.pad_token = tokenizer.eos_token
    print("ready to infer")
    setting()

#파일 로드/ 임베딩/ 모델 로드
def setting() :
    load_dotenv() # 토큰 정보로드

    ''' #pdf version
    # PDF 파일 로드 ① 데이터 로드
    loader = PyPDFLoader("240208코그플레이리플렛국문-저용량.pdf") #("EYAS_간이_사용설명서.pdf")
    document = loader.load()
    print("내용 일부 출력",document[1].page_content[:200])# 내용 추출

    docs = []
    docs.extend(loader.load_and_split())
    # ② 데이터 분할
    text_splitter = CharacterTextSplitter(
        separator="\n", #청크를 구분하는 데 사용되는 문자열
        chunk_size=1000, # 청크의 최대 길이
        chunk_overlap=200, #인접한 청크 간에 겹치는 문자의 수
        length_function=len,#청크의 길이를 계산하는 데 사용되는 함수
    )
    texts= text_splitter.split_documents(document)
    
    # ③ 저장 및 검색
    # 임베딩
    model_name = "hkunlp/instructor-xl"
    model_kwargs={'device':'cpu'}
    encode_kwargs={'normalize_embeddings':True}
    embeddings= HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs= model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    print("Found HuggingFaceEmbeddings!!♡🤗")
    model_id="mu0gum/AIFT-polyglot-ko-1.3b-ao-instruct-v0.91"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=200)
    
    #모델(LLM) 을 생성
    llama_llm = HuggingFacePipeline(pipeline=gen)

    # HuggingFacePipeline 객체 생성
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id, 
        device=0,               # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
        task="text-generation", # 텍스트 생성
        model_kwargs={"temperature": 0.1, 
                    "max_length": 64},
    )
    print("Found LLM model!!")
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 2

    # 인덱싱: 분할된 텍스트를 검색 가능한 형태로 만드는 단계// DB 에 저장
    chroma_vector = Chroma.from_documents(texts, embeddings)
    chroma_retriever = chroma_vector.as_retriever(search_kwargs={'k':2})
    '''


    ''' #txt version'''
    # txt 파일 로드 ① 데이터 로드
    loader = TextLoader('eyas_docs3.txt')
    data = loader.load()

    # ② 데이터 분할
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 500,
        chunk_overlap  = 100,
        length_function = len,
    )

    texts = text_splitter.split_text(data[0].page_content)
        
    # ③ 저장 및 검색
    # 임베딩
    model_name = "hkunlp/instructor-xl"
    model_kwargs={'device':'cpu'}
    encode_kwargs={'normalize_embeddings':True}
    # 허깅페이스 임베딩을 사용하여 기본 임베딩 설정
    embeddings= HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs= model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # #로컬 파일 저장소 설정
    # store = LocalFileStore("./cache/")

    # # 캐시를 지원하는 임베딩 생성
    # cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    #     embeddings,
    #     store,
    #     namespace=embeddings.model,  # 기본 임베딩과 저장소를 사용하여 캐시 지원 임베딩을 생성
    # )

    # # store에서 키들을 순차적으로 가져옵니다.
    # list(store.yield_keys())




    def cache_embed_wrapper(embedding_model, local_store_path=None):
        if local_store_path is not None:
            store = LocalFileStore(local_store_path)
        else:
            store = InMemoryByteStore()

        cache_embed_wrapper = CacheBackedEmbeddings.from_bytes_store(embedding_model,
                                                                    document_embedding_cache=store,
                                                                    namespace=embedding_model.model_name)
        return cache_embed_wrapper

    print("Found HuggingFaceEmbeddings!!♡🤗")

    def search(query: str, k: int = 3 ):
        """a function that embeds a new query and returns the most probable results"""
        embedded_query = ST.encode(query) # embed new query
        scores, retrieved_examples = data.get_nearest_examples( # retrieve results
            "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
            k=k # get only top k results
        )
        return scores, retrieved_examples


    #model_id="momo/polyglot-ko-12.8b-Chat-QLoRA-Merge"#"gwonny/llama-2-koen-13b-QLoRA-NEFTune-kolon-v0.1"
    #"mu0gum/AIFT-polyglot-ko-1.3b-ao-instruct-v0.91"#"nlpai-lab/kullm-polyglot-5.8b-v2"#"Changgil/K2S3-Llama2-13b-v1.0"#"beomi/open-llama-2-ko-7b"#"KRAFTON/KORani-v2-13B"
    #"beomi/open-llama-2-ko-7b"#"beomi/OPEN-SOLAR-KO-10.7B"#"cocoirun/AIFT-42dot-PLM-1.3B-ao-instruct-all-v0.4-ff-e1"#"mu0gum/AIFT-polyglot-ko-1.3b-ao-instruct-v0.91"

    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)

    # use quantization to lower GPU usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        resume_download = True
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    SYS_PROMPT = """You are an assistant for answering questions.
    You are given the extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say "I do not know." Don't make up an answer."""

    def format_prompt(prompt,retrieved_documents,k):
        """using the retrieved documents we will prompt the model to generate our responses"""
        PROMPT = f"Question:{prompt}\nContext:"
        for idx in range(k) :
            PROMPT+= f"{retrieved_documents['text'][idx]}\n"
        return PROMPT

    def generate(formatted_prompt):
        formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
        messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
        # tell the model to generate
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)

    def rag_chatbot(prompt:str,k:int=2):
        scores , retrieved_documents = search(prompt, k)
        formatted_prompt = format_prompt(prompt,retrieved_documents,k)
        return generate(formatted_prompt)

    rag_chatbot("what's anarchy ?", k = 2)
    print("🐫🐫🐫")
    print(rag_chatbot("what's anarchy ?", k = 2))
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=200)

    #print("크기 확인용👧🏻 ", model.num_parameters())
    #모델(LLM) 을 생성 - 모델/파이프라인을 수동으로 전달
    global llama_llm
    llama_llm = HuggingFacePipeline(pipeline=gen)

    #global gpu_llm
    # HuggingFacePipeline 객체 생성 - from_model_id를 사용하여 모델 매개변수를 지정함으로써 로드할 수 있음
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=model_id, 
    #     device=0,               # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
    #     task="text-generation", # 텍스트 생성
    #     model_kwargs={"temperature": 0.1, 
    #                 "max_length": 64},
    # )
    # gpu_llm = HuggingFacePipeline.from_model_id(
    #     model_id=model_id,  # 사용할 모델의 ID를 지정합니다.
    #     task="text-generation",  # 수행할 작업을 설정합니다. 여기서는 텍스트 생성입니다.
    #     # 사용할 GPU 디바이스 번호를 지정합니다. "auto"로 설정하면 accelerate 라이브러리를 사용합니다.
    #     device=0,
    #     # 파이프라인에 전달할 추가 인자를 설정합니다. 여기서는 생성할 최대 토큰 수를 10으로 제한합니다.
    #     pipeline_kwargs={"max_new_tokens": 64},
    # )
    # gpu_llm = HuggingFacePipeline.from_model_id(
    #     model_id=model_id,  # 사용할 모델의 ID를 지정합니다.
    #     task="text-generation",  # 수행할 작업을 설정합니다.
    #     device=0,  # GPU 디바이스 번호를 지정합니다. -1은 CPU를 의미합니다.
    #     batch_size=2,  # 배치 크기s를 조정합니다. GPU 메모리와 모델 크기에 따라 적절히 설정합니다.
    #     model_kwargs={
    #         "temperature": 0,
    #         "max_length": 64,
    #     },  # 모델에 전달할 추가 인자를 설정합니다.
    # )


    template = """Answer the following question in Korean.
    #Question: 
    {question}

    #Answer: """  # 질문과 답변 형식을 정의하는 템플릿
    prompt = PromptTemplate.from_template(template)  # 템플릿을 사용하여 프롬프트 객체 생성
    
    # # 프롬프트와 언어 모델을 연결하여 체인을 생성합니다.
    # gpu_chain = prompt | llama_llm.bind(stop=["\n\n"])

    # questions = []
    # for i in range(4):
    #     # 질문 리스트를 생성합니다.
    #     questions.append({"question": f"숫자 {i} 이 한글로 뭐에요?"})

    # answers = gpu_chain.batch(questions)  # 질문 리스트를 배치 처리하여 답변을 생성합니다.
    # for answer in answers:
    #     print(answer)  # 생성된 답변을 출력합니다.

    print("Found LLM model!!")

    # llm_chain = LLMChain(prompt=prompt, llm=llama_llm)
    # question = "어깨 아파. 어떡해?"
    # print(question)
    # print(llm_chain.run(question=question))
    

    global bm25_retriever
    #유사도 검색
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 2

    # 인덱싱: 분할된 텍스트를 검색 가능한 형태로 만드는 단계// DB 에 저장
    #키워드 검색
    global chroma_retriever
    #원본
    # chroma_vector = Chroma.from_texts(texts, embeddings)
    # chroma_retriever = chroma_vector.as_retriever(search_kwargs={'k':2})
    #원본

    start = time.time()
    chroma_vector = Chroma.from_texts(texts, cache_embed_wrapper(embeddings))
    chroma_retriever = chroma_vector.as_retriever(search_kwargs={'k':2})
    math.factorial(100000)
    end = time.time()

    print(f"{end - start:.5f} sec")
    #출처: https://blockdmask.tistory.com/549 [개발자 지망생:티스토리]

    # 코드 실행 시간을 측정합니다.
    
    global ensemble_retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers =[bm25_retriever, chroma_retriever], weights = [0.7, 0.3]
    )

    global memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    global conversation_chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llama_llm,
        retriever=ensemble_retriever,
        memory=memory
    )
    # 주석 2024.04.15
    # ensemble_docs = ensemble_retriever.get_relevant_documents("what is some investing advice?")
    # len(ensemble_docs)

    # prompt_template = """마지막 질문에 답하려면 다음 문맥을 사용하세요.
    # 답을 모르면 모른다고 말하고 답을 만들어내려고 하지 마세요.

    # {context}

    # Question: {question}
    # """
    # PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )

    # question = "달려라 다람이는 뭐야?"

    # def format_docs(docs):
    #     # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    #     return "\n\n".join(doc.page_content for doc in docs)
    # chain =(
    #     {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough(),}
    #     | PROMPT
    #     | llama_llm
    # )
    # result = chain.invoke("달려라 다람이는 뭐야?")
    # print(result)
    # 주석 2024.04.15

    # print(llama_llm.predict(text=PROMPT.format_prompt(
    #     context=ensemble_docs,
    #     question=question
    # ).text))

    # prompt_template = """마지막 질문에 답하려면 다음 문맥을 사용하세요.
    # 답을 모르면 모른다고 말하고 답을 만들어내려고 하지 마세요.

    # {context}

    # Question: {question}
    # Answer:"""
    # PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )

    # question = "달려라 다람이는 뭐야?"

    # print(llama_llm.predict(text=PROMPT.format_prompt(
    #     context=ensemble_docs,
    #     question=question
    # ).text))


    # 다중 쿼리 검색기
    # vectordb = FAISS.from_texts(texts=texts, embedding=embeddings)
    # global retriever_from_llm
    # retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(),
    #                                                 llm=llama_llm)

def handle_userinput(user_question):
    # response = st.session_state.conversation({'question': user_question})
    # st.session_state.chat_history = response['chat_history']
    global rp, chat_history, helpfulAnswer_n
    rp = conversation_chain({'question': user_question})
    chat_history = rp['chat_history']

    global text
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            #print(i," wonderful")
            print(i,"//user: " + message.content)
        else:
            print(i,"//bot: " + message.content)
            #text = message.content
            helpfulAnswer = message.content
            # s = user_question+"\nHelpful Answer:"
            # if(s in helpfulAnswer):
            #     print("검색 문자열이 메인 문자열에 포함되어 있음😛")
            helpfulAnswer_n = helpfulAnswer.split(user_question+"\nHelpful Answer:")
          
            

def sendQuestion(question) :
    def format_docs(docs):
        # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in docs)

    # 프롬프트를 생성합니다.
    prompt = hub.pull("rlm/rag-prompt")

    #SQLite Cache 20240416
    set_llm_cache(SQLiteCache(database_path="my_llm_cache.db"))


    handle_userinput(question)

    print("테스트" + time.strftime('%Y.%m.%d - %H:%M:%S'))

    # 지금 테스트 하는 거는 여기 rag_chain 안 맞다
    """
    #체인 생성(Create Chain)
    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llama_llm
        | StrOutputParser()
    )


    print("\n현재시간" + time.strftime('%Y.%m.%d - %H:%M:%S'))

    # now = time.localtime()
    # print ("체인 생성 시간:","%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    global answer
    response = rag_chain.invoke(question+" 두문장으로 말해줘. 인덱스 빼고 답해줘")
    #print("Answer:" in response) True
    #print("Answer:\n" in response)
    if "Answer:" in response:
        answer = response.split("Answer:", 1)[1]
        #answer.replace("\n", " ")
        if "답변:\n" in response:
            answer = response.split("답변:", 1)[1]
        print("이것이 answer다 : ", answer)
    # now = time.localtime()
    # print ("답변 생성시간:","%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print(f"🤖[AI]\n{response}", end='') #, end='' 줄바꿈없는 print
    #text = gen(input("질문 내용 입력: "), model=model, tokenizer=tokenizer, device=device)
    #text = LLM_infer(input("질문 내용 입력: "))
    # text에서 </s>를 삭제
    textArr = response.split(".")
    #print(f"배열 길이 {len(textArr)}")
    if(len(textArr) > 1):
        for i in range(0, len(textArr)):
            for j in range(i+1, len(textArr)):
                if(j < len(textArr)):
                    if(textArr[i] == textArr[j]):
                        del textArr[j]
    print(f"\n텍스트 임당 {response}")
    #text = ".".join(textArr)
    if(len(textArr) > 2):
        text = textArr[0] +"."+ textArr[1] +"."
    else:
        text = ".".join(textArr)
    print(f"==========================중복 제거 확인=================\n {text}")
    text2 = text.replace('\n', ' ')
    text = text2.replace('"answer": "', "")
    text = answer
    """

    '''
    # 앙상블 결과 보기
    def pretty_print(docs):
        for i, doc in enumerate(docs):
            print(f"[{i+1}] {doc.page_content}")

    sample_query = question
    print(f"[Query]\n{sample_query}\n")
    relevant_docs = bm25_retriever.get_relevant_documents(sample_query)
    print("[BM25 Retriever]")
    pretty_print(relevant_docs)
    print("===" * 20)
    relevant_docs = chroma_retriever.get_relevant_documents(sample_query)
    print("[Chroma Retriever]")
    pretty_print(relevant_docs)
    print("===" * 20)
    relevant_docs = ensemble_retriever.get_relevant_documents(sample_query)
    print("[Ensemble Retriever]")
    pretty_print(relevant_docs)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llama_llm,
        retriever=ensemble_retriever,#retriever=vectorstore.as_retriever(),#retriever=ensemble_retriever,# retriever 가져옴
        memory=memory
    )

    chat_history = []
    #result = {}
    query = "자존감은 뭐야?"#"EYAS를 얼마나 해야 인지능력이 좋아지나요?"

    result = conversation_chain.invoke({"question": query, "chat_history": chat_history}, return_only_outputs=True)
    print(query)
    print(result["answer"])

    llm_chain = LLMChain(prompt=prompt, llm=llama_llm)
    question = "자신감이 뭐야?"
    print(question)
    print(llm_chain.run(question=question))
    '''
    # now = time.localtime()
    # print ("중복제거 후 시간:","%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

    #체인 생성(Create Chain)
    # rag_chain2 = (
    #     {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llama_llm
    #     | StrOutputParser()
    # )
    # response = rag_chain2.invoke(question)
    # print(f"[다중 응답 리트리버 답변]\n{response}")

    finalAns = helpfulAnswer_n[-1]
    print("finalAns ==> ",finalAns)
    if("\n" in finalAns):
        #finalAns = ''.join(helpfulAnswer_n[-1].splitlines()).split() #문장의 공백도 끊음 # 리스트에서 빈 문자열인 원소 제거
        #finalAns= list(filter(None, helpfulAnswer_n[-1].splitlines())) 배열로 만들어져서 string 후처리 좀
        print("잘 정리됐는지 확인 " ,finalAns )
        
        #finalAns = finalAns[0] #U
        finalAns=findNicdAnswer(finalAns, question)
        print("잘 정리됐는지 확인 " ,finalAns )
        #finalAns = helpfulAnswer_n[-1].splitlines()[0]
        # ansList = helpfulAnswer_n[-1].splitlines()
        # for i in ansList:
        #     if(ansList[i]==""):
        #         continue
        #     else
        #         finalAns = ansList[i]
        #         break

        # if(len(finalAns) > 10):
        #     finalAns = ' '.join(finalAns)
        #     print(len(helpfulAnswer_n[-1].splitlines()))
        #     finalAns = findNicdAnswer(finalAns, question)
        #     #finalAns = ansList[3]
        #     #helpfulAnswer_2 = finalAns.split("Helpful Answer:")[-1]
        #     print("확인 🙆🏻‍♀️", helpfulAnswer_2[-1])
        print("😝들어왔으\n"+str(helpfulAnswer_n[-1].splitlines()))
    #print("확인 🙆🏻‍♀️", helpfulAnswer_n[-1])
    return finalAns

def findNicdAnswer(sentence, user_question):
    s = user_question+"\nHelpful Answer:"
    print(s in sentence)
    if(s in sentence):
    #     print("검색 문자열이 메인 문자열에 포함되어 있음😛")
        niceAnswer = sentence.split(user_question+"\nHelpful Answer:\n")[-1]
    else:
        niceAnswer = sentence
    print("🥨niceAnswer: ",niceAnswer)
    ansList = niceAnswer.splitlines()
    # for ans in ansList:
    #     if(ans==""):
    #         continue
    #     else:
    #         finalAns = ans
    #         break
    finalAns = ansList[-1]
    return finalAns
          
def main():
    settingModel()
	
    


if __name__ == "__main__":
    main()