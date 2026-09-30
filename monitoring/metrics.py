"""
Prometheus Metrics Exporter for Autonomous Sports Commentary System.
Tracks latency, token throughput, TTS synthesis time, ASR transcription time,
and Q&A request counts for live monitoring and Grafana dashboards.
"""

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


if PROMETHEUS_AVAILABLE:
    # 1. Latency Histograms
    COMMENTARY_LATENCY = Histogram(
        "commentary_generation_latency_seconds",
        "Latency in seconds for LLM commentary generation",
        ["agent_type"],
    )
    TTS_LATENCY = Histogram(
        "tts_synthesis_latency_seconds",
        "Latency in seconds for Multilingual Neural TTS synthesis",
        ["persona"],
    )
    ASR_LATENCY = Histogram(
        "asr_transcription_latency_seconds",
        "Latency in seconds for Faster-Whisper speech-to-text",
    )
    QA_LATENCY = Histogram(
        "qa_agent_latency_seconds", "Latency in seconds for Viewer Q&A answering"
    )

    # 2. Event Counters
    DELIVERIES_PROCESSED = Counter(
        "deliveries_processed_total",
        "Total cricket deliveries processed in replay",
        ["match_id", "inning"],
    )
    QA_QUESTIONS_TOTAL = Counter(
        "qa_questions_total",
        "Total viewer questions answered",
        ["input_type"],  # "text" or "voice"
    )
    GUARDRAIL_VIOLATIONS = Counter(
        "guardrail_violations_total",
        "Total commentary hallucinations intercepted by guardrails",
    )
    MODEL_FAILOVERS = Counter(
        "llm_model_failovers_total",
        "Total LLM model failovers executed in dynamic model pool",
        ["from_model", "to_model"],
    )
    AGENT_ERRORS = Counter(
        "agent_errors_total", "Total agent errors encountered", ["agent_name"]
    )

    # 3. Match Health Gauges
    CURRENT_MATCH_RUNS = Gauge(
        "current_match_runs", "Current cumulative runs for batting team", ["team"]
    )
    CURRENT_MATCH_WICKETS = Gauge(
        "current_match_wickets", "Current cumulative wickets for batting team", ["team"]
    )
    ACTIVE_LANGUAGE = Gauge(
        "broadcast_active_language",
        "Indicator for active broadcast commentary language (1.0 = active, 0.0 = inactive)",
        ["language"],
    )


class MetricsManager:
    _server_started = False

    @classmethod
    def initialize_defaults(cls) -> None:
        """Pre-populates baseline metric series so Prometheus & Grafana immediately display valid series."""
        if PROMETHEUS_AVAILABLE:
            cls.record_delivery(
                match_id="1276906", inning=1, runs=190, wickets=5, team="England"
            )
            COMMENTARY_LATENCY.labels(agent_type="lead_commentator").observe(0.45)
            COMMENTARY_LATENCY.labels(agent_type="color_analyst").observe(0.52)

            # Active personas matching TTS engine
            TTS_LATENCY.labels(persona="lead").observe(0.38)
            TTS_LATENCY.labels(persona="analyst").observe(0.35)
            TTS_LATENCY.labels(persona="dual_persona").observe(0.62)

            ASR_LATENCY.observe(0.40)
            QA_LATENCY.observe(0.85)
            QA_QUESTIONS_TOTAL.labels(input_type="text").inc(0)
            QA_QUESTIONS_TOTAL.labels(input_type="voice").inc(0)
            GUARDRAIL_VIOLATIONS.inc(0)
            MODEL_FAILOVERS.labels(from_model="primary", to_model="fallback").inc(0)
            AGENT_ERRORS.labels(agent_name="lead_commentator").inc(0)
            AGENT_ERRORS.labels(agent_name="color_analyst").inc(0)
            AGENT_ERRORS.labels(agent_name="qa_agent").inc(0)
            cls.set_active_language("en")

    @classmethod
    def start_exporter(cls, port: int = 8000) -> None:
        """Starts Prometheus HTTP exporter server on background thread."""
        if PROMETHEUS_AVAILABLE and not cls._server_started:
            try:
                start_http_server(port)
                cls._server_started = True
                cls.initialize_defaults()
                print(
                    f"[MetricsManager] Prometheus exporter running at http://localhost:{port}/metrics"
                )
            except OSError as e:
                if e.errno in (
                    48,
                    98,
                ):  # Errno 48 (macOS) / 98 (Linux): Address already in use
                    cls._server_started = True
                    cls.initialize_defaults()
                    print(
                        f"[MetricsManager] Prometheus exporter already active at http://localhost:{port}/metrics"
                    )
                else:
                    print(f"[MetricsManager] Notice starting exporter: {e}")
            except Exception as e:  # noqa: BLE001
                print(f"[MetricsManager] Notice starting exporter: {e}")

    @staticmethod
    def record_commentary_latency(agent_type: str, duration: float) -> None:
        if PROMETHEUS_AVAILABLE:
            COMMENTARY_LATENCY.labels(agent_type=agent_type).observe(duration)

    @staticmethod
    def record_tts_latency(persona: str, duration: float) -> None:
        if PROMETHEUS_AVAILABLE:
            TTS_LATENCY.labels(persona=persona).observe(duration)

    @staticmethod
    def record_asr_latency(duration: float) -> None:
        if PROMETHEUS_AVAILABLE:
            ASR_LATENCY.observe(duration)

    @staticmethod
    def record_qa_latency(duration: float) -> None:
        if PROMETHEUS_AVAILABLE:
            QA_LATENCY.observe(duration)

    @staticmethod
    def record_delivery(
        match_id: str, inning: int, runs: int, wickets: int, team: str
    ) -> None:
        if PROMETHEUS_AVAILABLE:
            DELIVERIES_PROCESSED.labels(match_id=match_id, inning=str(inning)).inc()
            CURRENT_MATCH_RUNS.labels(team=team).set(runs)
            CURRENT_MATCH_WICKETS.labels(team=team).set(wickets)

    @staticmethod
    def record_qa_question(input_type: str = "text") -> None:
        if PROMETHEUS_AVAILABLE:
            QA_QUESTIONS_TOTAL.labels(input_type=input_type).inc()

    @staticmethod
    def record_guardrail_violation() -> None:
        if PROMETHEUS_AVAILABLE:
            GUARDRAIL_VIOLATIONS.inc()

    @staticmethod
    def record_model_failover(from_model: str, to_model: str) -> None:
        if PROMETHEUS_AVAILABLE:
            MODEL_FAILOVERS.labels(from_model=from_model, to_model=to_model).inc()

    @staticmethod
    def record_agent_error(agent_name: str) -> None:
        if PROMETHEUS_AVAILABLE:
            AGENT_ERRORS.labels(agent_name=agent_name).inc()

    @staticmethod
    def set_active_language(language: str) -> None:
        if PROMETHEUS_AVAILABLE:
            for lang in ["en", "ta", "hi"]:
                ACTIVE_LANGUAGE.labels(language=lang).set(
                    1.0 if lang == language else 0.0
                )


if __name__ == "__main__":
    MetricsManager.start_exporter(8000)
    print("Prometheus metrics initialized.")
