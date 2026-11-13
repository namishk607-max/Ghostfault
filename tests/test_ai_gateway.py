import json
from types import SimpleNamespace

import pytest

from src.ai_gateway import (
    DEFAULT_BASE_URL,
    GatewayAPIError,
    GatewayAuthenticationError,
    GatewayConfig,
    GatewayConnectionError,
    GatewayDependencyError,
    GatewayNotConfiguredError,
    GatewayResponseError,
    GatewayScopeError,
    GatewayTimeoutError,
    load_gateway_config,
    request_commentary,
)
from src.models import (
    AnalysisStatus,
    FaultAnalysisResult,
    InvestigativeHypothesis,
)


def _incident(dataset_id="DATASET-A"):
    return {
        "incident_id": "INC-1",
        "dataset_id": dataset_id,
        "symptom": "Elevated bearing temperature",
        "maintenance_action": "Lubricated the drive-end bearing",
        "private_note": "Must not leave the application",
    }


def _result():
    return FaultAnalysisResult(
        incident_id="INC-1",
        incident_facts={"secret": "not sent"},
        retrieved_evidence=[],
        hypotheses=[
            InvestigativeHypothesis(
                title="Possible lubrication issue",
                description="The symptom may be consistent with increased friction.",
                qualification="Tentative; requires qualified review.",
                supporting_evidence=[],
                contradictory_evidence=[],
                suggested_checks=["Review applicable OEM procedure."],
            )
        ],
        missing_information=[],
        limitations=[],
        status=AnalysisStatus.COMPLETED,
        analyzed_at="2026-01-01T00:00:00+00:00",
    )


def _config(**kwargs):
    values = {
        "base_url": DEFAULT_BASE_URL,
        "api_key": "test-key-not-for-display",
        "model": "test-model",
    }
    values.update(kwargs)
    return GatewayConfig(**values)


class _FakeClient:
    def __init__(self, content="Tentative commentary."):
        self.content = content
        self.call = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def _client_factory(client):
    def create_client(config):
        assert config.model
        return client

    return create_client


def test_gateway_is_disabled_without_explicit_enable():
    assert load_gateway_config(False, environ={}, secrets={}) is None


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"OMNIROUTE_API_KEY": "key"},
        {"OMNIROUTE_MODEL": "model"},
    ],
)
def test_missing_configuration_is_rejected_only_when_enabled(values):
    with pytest.raises(GatewayNotConfiguredError):
        load_gateway_config(True, environ=values, secrets={})


def test_configuration_reads_environment_and_streamlit_secrets():
    config = load_gateway_config(
        True,
        environ={"OMNIROUTE_API_KEY": "env-key", "OMNIROUTE_MODEL": "env-model"},
        secrets={"OMNIROUTE_API_KEY": "secret-key", "OMNIROUTE_MODEL": "secret-model"},
    )
    assert config is not None
    assert config.api_key == "env-key"
    assert config.model == "env-model"
    assert config.base_url == DEFAULT_BASE_URL
    assert "env-key" not in repr(config)

    from_secrets = load_gateway_config(
        True,
        environ={},
        secrets={"OMNIROUTE_API_KEY": "secret-key", "OMNIROUTE_MODEL": "secret-model"},
    )
    assert from_secrets is not None
    assert from_secrets.api_key == "secret-key"


def test_invalid_gateway_url_is_rejected():
    for base_url in (
        "https://user:password@example.test/v1",
        "http://remote.example/v1",
        "http://localhost:bad/v1",
        "http://[invalid/v1",
    ):
        with pytest.raises(GatewayNotConfiguredError):
            load_gateway_config(
                True,
                environ={
                    "OMNIROUTE_API_KEY": "key",
                    "OMNIROUTE_MODEL": "model",
                    "OMNIROUTE_BASE_URL": base_url,
                },
            )


def test_remote_https_gateway_url_is_allowed():
    config = load_gateway_config(
        True,
        environ={
            "OMNIROUTE_API_KEY": "key",
            "OMNIROUTE_MODEL": "model",
            "OMNIROUTE_BASE_URL": "https://gateway.example/v1",
        },
    )
    assert config is not None
    assert config.base_url == "https://gateway.example/v1"


def test_successful_response_is_bounded_and_only_sends_minimal_current_dataset_data():
    client = _FakeClient()
    result = _result()
    before = result.to_dict()

    commentary = request_commentary(
        _config(),
        _incident(),
        result,
        "DATASET-A",
        client_factory=_client_factory(client),
    )

    assert commentary == "Tentative commentary."
    assert result.to_dict() == before
    assert client.call["model"] == "test-model"
    assert client.call["max_tokens"] > 0
    user_payload = json.loads(client.call["messages"][1]["content"])
    assert user_payload["incident"] == {
        "symptom": "Elevated bearing temperature",
        "maintenance_action": "Lubricated the drive-end bearing",
    }
    serialized_request = json.dumps(client.call["messages"])
    for private_value in ("DATASET-A", "INC-1", "private_note", "not sent"):
        assert private_value not in serialized_request


def test_response_cannot_echo_api_key_to_the_ui():
    client = _FakeClient(content="Unexpected key: test-key-not-for-display")
    commentary = request_commentary(
        _config(),
        _incident(),
        _result(),
        "DATASET-A",
        client_factory=_client_factory(client),
    )
    assert "test-key-not-for-display" not in commentary
    assert "[REDACTED]" in commentary


def test_dataset_or_incident_mismatch_fails_before_client_creation():
    factory_called = False

    def client_factory(config):
        nonlocal factory_called
        assert config.model
        factory_called = True
        return _FakeClient()

    with pytest.raises(GatewayScopeError):
        request_commentary(
            _config(),
            _incident(dataset_id="DATASET-B"),
            _result(),
            "DATASET-A",
            client_factory=client_factory,
        )
    assert not factory_called


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("secret-timeout-diagnostic-8421"), GatewayTimeoutError),
        (ConnectionError("secret-connection-detail-8421"), GatewayConnectionError),
        (
            type("AuthenticationError", (Exception,), {})("secret-auth-detail-8421"),
            GatewayAuthenticationError,
        ),
        (RuntimeError("secret-provider-detail-8421"), GatewayAPIError),
    ],
)
def test_gateway_failures_are_safe_and_do_not_leak_api_key(error, expected):
    def raise_error(**kwargs):
        assert "messages" in kwargs
        raise error

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=raise_error)))
    with pytest.raises(expected) as raised:
        request_commentary(
            _config(),
            _incident(),
            _result(),
            "DATASET-A",
            client_factory=_client_factory(client),
        )
    assert "test-key-not-for-display" not in str(raised.value)
    assert str(error) not in str(raised.value)


def test_missing_optional_openai_dependency_is_reported(monkeypatch):
    import src.ai_gateway as gateway

    def missing_module(name):
        assert name == "openai"
        raise ModuleNotFoundError("openai")

    monkeypatch.setattr(gateway.importlib, "import_module", missing_module)
    with pytest.raises(GatewayDependencyError):
        request_commentary(_config(), _incident(), _result(), "DATASET-A")


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "x" * 4001,
    ],
)
def test_malformed_empty_and_oversized_responses_are_rejected(content):
    with pytest.raises(GatewayResponseError):
        request_commentary(
            _config(),
            _incident(),
            _result(),
            "DATASET-A",
            client_factory=_client_factory(_FakeClient(content=content)),
        )


def test_malformed_response_shape_is_rejected():
    def invalid_response(**kwargs):
        assert kwargs["model"]
        return {}

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=invalid_response),
        )
    )
    with pytest.raises(GatewayResponseError):
        request_commentary(
            _config(),
            _incident(),
            _result(),
            "DATASET-A",
            client_factory=_client_factory(client),
        )
