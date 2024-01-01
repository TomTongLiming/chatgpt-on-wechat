"""
Ollama Bot
"""
# encoding:utf-8

from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage

from bot.baidu.baidu_wenxin_session import BaiduWenxinSession
from bot.bot import Bot
from bot.session_manager import SessionManager
from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf

"""
{
  "model": "mistral",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}
"""


# localhost:11434 Ollama API (可用)
class OllamaBot(Bot):

    def __init__(self):
        super().__init__()
        # self.api_key = conf().get("gemini_api_key")
        # 复用文心的token计算方式
        self.sessions = SessionManager(BaiduWenxinSession, model=conf().get("model") or "llama2")

    def reply(self, query, context: Context = None) -> Reply:
        try:
            if context.type != ContextType.TEXT:
                logger.warn(f"[Ollama] Unsupported message type, type={context.type}")
                return Reply(ReplyType.TEXT, None)
            logger.info(f"[Ollama] query={query}")
            session_id = context["session_id"]
            session = self.sessions.session_query(query, session_id)
            model = conf().get("model", "llama2")
            chat_model = ChatOllama(
                model=model
            )
            messages_without_role = [HumanMessage(content=message['content']) for message in session.messages]
            response = chat_model(messages_without_role)
            reply_text = response.content
            self.sessions.session_reply(reply_text, session_id)
            logger.info(f"[Ollama] reply={reply_text}")
            return Reply(ReplyType.TEXT, reply_text)
        except Exception as e:
            logger.error("[Ollama] fetch reply error, may contain unsafe content")
            logger.error(e)
