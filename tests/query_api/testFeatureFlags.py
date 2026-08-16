from __future__ import annotations

from query_api.feature_flags import FeatureFlags, FeatureFlagService


class FakeStore:
    def __init__(self, flags: FeatureFlags | None = None) -> None:
        self.flags = flags or FeatureFlags()
        self.initialized = False
        self.closed = False

    def initialize(self) -> None:
        self.initialized = True

    def load(self) -> FeatureFlags:
        return self.flags

    def save(self, flags: FeatureFlags) -> None:
        self.flags = flags

    def close(self) -> None:
        self.closed = True


def testFlagsDefaultToEnabledAndPersistUpdates() -> None:
    store = FakeStore()
    service = FeatureFlagService(store)  # type: ignore[arg-type]

    service.start()
    updated = service.update(faq_enabled=False, classifier_enabled=None)
    service.close()

    assert store.initialized is True
    assert updated == FeatureFlags(faq_enabled=False, classifier_enabled=True)
    assert store.flags == updated
    assert store.closed is True
