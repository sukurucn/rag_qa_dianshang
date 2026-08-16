from model_trian_classify.query_router import QueryRouter, RouteLabel


class FakePredictor:
    def __init__(self, label_id: int, confidence: float) -> None:
        self._label_id = label_id
        self._confidence = confidence

    def predict(self, question: str) -> tuple[int, float]:
        return self._label_id, self._confidence


def testRoutesConfidentGeneralKnowledgeToWebAndRag() -> None:
    router = QueryRouter(FakePredictor(label_id=0, confidence=0.92))  # type: ignore[arg-type]

    decision = router.route("What is Python?")

    assert decision.label is RouteLabel.GENERAL_KNOWLEDGE
    assert decision.target_route == "web_rag"


def testRoutesLowConfidenceAndProfessionalQuestionsToRagQa() -> None:
    low_confidence = QueryRouter(FakePredictor(label_id=0, confidence=0.51))  # type: ignore[arg-type]
    professional = QueryRouter(FakePredictor(label_id=1, confidence=0.98))  # type: ignore[arg-type]

    assert low_confidence.route("ambiguous").target_route == "rag_qa"
    assert professional.route("course fee").target_route == "rag_qa"
