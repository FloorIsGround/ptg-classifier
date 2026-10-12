class PipelineError(Exception):
    """Base exception for pipeline errors."""


class ConfigError(PipelineError):
    pass


class ValidationError(PipelineError):
    pass
