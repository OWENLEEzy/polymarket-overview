from app.config import Settings


def test_settings_reads_prefixed_deepseek_environment(monkeypatch) -> None:
    monkeypatch.setenv("POLYMARKET_OVERVIEW_DEEPSEEK_API_KEY", "key-123")
    monkeypatch.setenv("POLYMARKET_OVERVIEW_DEEPSEEK_MODEL", "deepseek-test")

    settings = Settings(_env_file=None)

    assert settings.deepseek_api_key == "key-123"
    assert settings.deepseek_model == "deepseek-test"
