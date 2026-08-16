"""面向终端用户的独立 FastAPI 问答应用。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from base.config import Settings, settings
from base.logger import get_logger
from conversation_memory.models import ConversationSession, ConversationTurn
from conversation_memory.repository import MysqlConversationStore
from conversation_memory.service import ConversationMemoryService
from model_trian_classify.query_router import QueryRouter
from mysql_qa.mysql_client import MysqlQaClient
from mysql_qa.redis_client import RedisQuestionCache
from mysql_qa.service import MysqlQaService
from query_api.feature_flags import FeatureFlagService, MysqlFeatureFlagStore
from query_api.schemas import (
    ConversationTurnResponse,
    FeatureFlagsResponse,
    FeatureFlagsUpdateRequest,
    QueryCitationResponse,
    QueryRequest,
    QueryResponse,
    SessionCreateRequest,
    SessionResponse,
    SessionUpdateRequest,
    WebCitationResponse,
)
from query_api.service import QueryAnswerService
from question_rewrite.langchain_client import LangChainRewriteModel
from question_rewrite.service import QuestionRewriteService
from rag_qa.answer_client import LangChainAnswerModel
from rag_qa.reranker import BgeParentReranker
from rag_qa.retriever import MilvusHybridRetriever
from rag_qa.service import RagQaService
from web_search.duckduckgo import DuckDuckGoWebSearcher

_LOGGER = get_logger("query_api")


@dataclass
class QueryApiServices:
    """用户问答应用的依赖容器，测试时可整体替换。"""

    mysql_client: MysqlQaClient
    redis_cache: RedisQuestionCache
    feature_flags: FeatureFlagService
    conversations: ConversationMemoryService
    answer_service: QueryAnswerService

    def start(self) -> None:
        """初始化 FAQ 表和缓存；失败时保留 RAG 服务能力。"""
        self.feature_flags.start()
        self.conversations.start()
        try:
            self.mysql_client.initialize_schema()
            self.redis_cache.warmup(self.mysql_client)
        except Exception:
            _LOGGER.exception("FAQ initialization failed; query API will continue with RAG fallback")

    def stop(self) -> None:
        """关闭已创建的 MySQL 连接。"""
        self.mysql_client.close()
        self.feature_flags.close()
        self.conversations.close()


def create_default_services(app_settings: Settings = settings) -> QueryApiServices:
    """构建惰性加载的生产依赖。"""
    if app_settings.configure_langsmith_tracing():
        _LOGGER.info("LangSmith tracing enabled: project=%s", app_settings.langsmith_project)
    else:
        _LOGGER.info("LangSmith tracing disabled: no API key or tracing switch is off")
    mysql_client = MysqlQaClient(app_settings)
    redis_cache = RedisQuestionCache(app_settings)
    faq_service = MysqlQaService(app_settings, mysql_client, redis_cache)
    feature_flags = FeatureFlagService(MysqlFeatureFlagStore(app_settings))
    conversations = ConversationMemoryService(MysqlConversationStore(app_settings))
    query_router = QueryRouter()
    rewrite_service = QuestionRewriteService(app_settings, LangChainRewriteModel(app_settings))
    rag_service = RagQaService(
        app_settings,
        MilvusHybridRetriever(app_settings),
        BgeParentReranker(),
        LangChainAnswerModel(app_settings, web_searcher=DuckDuckGoWebSearcher(app_settings)),
    )
    return QueryApiServices(
        mysql_client=mysql_client,
        redis_cache=redis_cache,
        feature_flags=feature_flags,
        conversations=conversations,
        answer_service=QueryAnswerService(
            app_settings,
            faq_service,
            query_router,
            rewrite_service,
            rag_service,
            feature_flags,
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.state.services = active_services

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        try:
            session = active_services.conversations.resolve(payload.session_id)
            memory_context = active_services.conversations.select_context(session.session_id, payload.question)
            result = active_services.answer_service.answer(
                payload.question,
                active_services.conversations.to_messages(memory_context),
            )
            turn_number = active_services.conversations.append(
                session.session_id,
                payload.question,
                result.answer,
            )
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在") from error
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
            web_citations=[
                WebCitationResponse(title=item.title, url=item.url, snippet=item.snippet)
                for item in result.web_citations
            ],
            web_search_used=result.web_search_used,
            session_id=session.session_id,
            turn_number=turn_number,
            selected_memory_turns=memory_context.selected_turn_count,
        )

    @app.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
    def create_session(payload: SessionCreateRequest) -> SessionResponse:
        return _session_response(
            active_services.conversations.create(
                title=payload.title,
                memory_turn_limit=payload.memory_turn_limit,
            )
        )

    @app.get("/sessions", response_model=list[SessionResponse])
    def list_sessions(
        limit: int = Query(default=50, ge=1, le=256),
        offset: int = Query(default=0, ge=0),
    ) -> list[SessionResponse]:
        return [_session_response(item) for item in active_services.conversations.list_sessions(limit=limit, offset=offset)]

    @app.patch("/sessions/{session_id}", response_model=SessionResponse)
    def update_session(session_id: str, payload: SessionUpdateRequest) -> SessionResponse:
        try:
            return _session_response(
                active_services.conversations.update(
                    session_id,
                    title=payload.title,
                    memory_turn_limit=payload.memory_turn_limit,
                )
            )
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在") from error

    @app.get("/sessions/{session_id}/turns", response_model=list[ConversationTurnResponse])
    def list_session_turns(
        session_id: str,
        limit: int = Query(default=256, ge=1, le=256),
        offset: int = Query(default=0, ge=0),
    ) -> list[ConversationTurnResponse]:
        try:
            return [
                _turn_response(item)
                for item in active_services.conversations.get_turns(
                    session_id,
                    limit=limit,
                    offset=offset,
                )
            ]
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在") from error

    @app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str) -> None:
        try:
            active_services.conversations.delete(session_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在") from error

    @app.get("/runtime/features", response_model=FeatureFlagsResponse)
    def get_feature_flags() -> FeatureFlagsResponse:
        flags = active_services.feature_flags.current()
        return FeatureFlagsResponse(
            faq_enabled=flags.faq_enabled,
            classifier_enabled=flags.classifier_enabled,
        )

    @app.patch("/runtime/features", response_model=FeatureFlagsResponse)
    def update_feature_flags(payload: FeatureFlagsUpdateRequest) -> FeatureFlagsResponse:
        flags = active_services.feature_flags.update(
            faq_enabled=payload.faq_enabled,
            classifier_enabled=payload.classifier_enabled,
        )
        return FeatureFlagsResponse(
            faq_enabled=flags.faq_enabled,
            classifier_enabled=flags.classifier_enabled,
        )

    return app


def _session_response(session: ConversationSession) -> SessionResponse:
    return SessionResponse(
        session_id=session.session_id,
        title=session.title,
        memory_turn_limit=session.memory_turn_limit,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _turn_response(turn: ConversationTurn) -> ConversationTurnResponse:
    return ConversationTurnResponse(
        turn_number=turn.turn_number,
        user_question=turn.user_question,
        assistant_answer=turn.assistant_answer,
        created_at=turn.created_at,
    )
