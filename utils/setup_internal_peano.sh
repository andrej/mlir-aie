export RYZENAI_SP=/proj/aiebuilds/ryzen-ai/ryzen-ai-TA/main/ryzenai_main_daily_latest/lnx64-internal/lib/python3.12/site-packages
export PEANO_INSTALL_DIR=$RYZENAI_SP/lnx64.o/tools/peano
# Picked up by Makefiles that thread it into PEANOWRAP*_FLAGS as the first -I,
# so <aie_api/aie.hpp> resolves to the ryzen-ai copy and stays in sync with
# the peano-bundled aie_api_compat.h under $PEANO_INSTALL_DIR.
export AIE_API_INCLUDE_DIR=$RYZENAI_SP/include
