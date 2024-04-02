import argparse
import tracing

trace_events = [
    tracing.TraceEvent.VECTOR_INST,
    tracing.TraceEvent.CORE_EVENT_0,
    tracing.TraceEvent.CORE_EVENT_1,
    tracing.TraceEvent.CORE_PORT_RUNNING_0,
    tracing.TraceEvent.CORE_PORT_RUNNING_1,
    tracing.TraceEvent.LOCK_STALL,
    tracing.TraceEvent.LOCK_ACQ_REQ,
    tracing.TraceEvent.LOCK_REL_REQ,
]


def get_default_argparser(M_default, K_default, N_default):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-M", type=int, default=M_default)
    argparser.add_argument("-K", type=int, default=K_default)
    argparser.add_argument("-N", type=int, default=N_default)
    argparser.add_argument(
        "--trace",
        type=int,
        default=0,
        help="Trace Size (0 disables trace functionality)",
    )
    return argparser
