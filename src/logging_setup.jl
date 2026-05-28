# ==============================================================================
# src/logging_setup.jl
#
# Logging configuration helpers.
# ==============================================================================

function setup_logger!(; level::LogLevel=Logging.Info)
    global_logger(ConsoleLogger(stderr, level))
    return nothing
end
