import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

#LangChain 支持多种不同的语言模型
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_core.messages import HumanMessage

model.invoke([HumanMessage(content="Hi! I'm Bob")])


model.invoke([HumanMessage(content="What's my name?")])