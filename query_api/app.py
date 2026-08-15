"""面向终端用户的独立 FastAPI 问答应用。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from base.config import Settings, settings
from base.logger import get_logger
from model_trian_classify.query_router import QueryRouter
from mysql_qa.mysql_client import MysqlQaClient
from mysql_qa.redis_client import RedisQuestionCache
from mysql_qa.service import MysqlQaService
from query_api.schemas import QueryCitationResponse, QueryRequest, QueryResponse
from query_api.service import QueryAnswerService
from question_rewrite.langchain_client import LangChainRewriteModel
from question_rewrite.service import QuestionRewriteService
from rag_qa.answer_client import LangChainAnswerModel
from rag_qa.reranker import BgeParentReranker
from rag_qa.retriever import MilvusHybridRetriever
from rag_qa.service import RagQaService

_LOGGER = get_logger("query_api")


@dataclass
class QueryApiServices:
    """用户问答应用的依赖容器，测试时可整体替换。"""

    mysql_client: MysqlQaClient
    redis_cache: RedisQuestionCache
    answer_service: QueryAnswerService

    def start(self) -> None:
        """初始化 FAQ 表和缓存；失败时保留 RAG 服务能力。"""
        try:
            self.mysql_client.initialize_schema()
            self.redis_cache.warmup(self.mysql_client)
        except Exception:
            _LOGGER.exception("FAQ initialization failed; query API will continue with RAG fallback")

    def stop(self) -> None:
        """关闭已创建的 MySQL 连接。"""
        self.mysql_client.close()


def create_default_services(app_settings: Settings = settings) -> QueryApiServices:
    """构建惰性加载的生产依赖。"""
    mysql_client = MysqlQaClient(app_settings)
    redis_cache = RedisQuestionCache(app_settings)
    faq_service = MysqlQaService(app_settings, mysql_client, redis_cache)
    query_router = QueryRouter()
    rewrite_service = QuestionRewriteService(app_settings, LangChainRewriteModel(app_settings))
    rag_service = RagQaService(
        app_settings,
        MilvusHybridRetriever(app_settings),
        BgeParentReranker(),
        LangChainAnswerModel(app_settings),
    )
    return QueryApiServices(
        mysql_client=mysql_client,
        redis_cache=redis_cache,
        answer_service=QueryAnswerService(
            app_settings,
            faq_service,
            query_router,
            rewrite_service,
            rag_service,
        ),
    )


def create_app(
    *,
    app_settings: Settings = settings,
    services: QueryApiServices | None = None,
) -> FastAPI:
    """创建用户问答 API；部署时由外层网关负责认证与限流。"""
    active_services = services or create_default_services(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        active_services.start()
        try:
            yield
        finally:
            active_services.stop()

    app = FastAPI(title="RAG 用户问答 API", version="0.1.0", lifespan=lifespan)
    app.state.services = active_services

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        result = active_services.answer_service.answer(payload.question)
        return QueryResponse(
            answer=result.answer,
            source=result.source,
            classification=result.classification,
            citations=[
                QueryCitationResponse(
                    chunk_id=citation.chunk_id,
                    source=citation.source,
                    title_path=citation.title_path,
                    text=citation.text,
                )
                for citation in result.citations
            ],
            faq_confidence=result.faq_confidence,
            classification_confidence=result.classification_confidence,
            fallback_reason=result.fallback_reason,
        )

    return app
