from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-5-5")
print(llm.invoke("Confirming setup works in 5 words.").content)
