#pragma once

// Compile only the PULSE-S channel pairs. Keeping this registry separate from
// the 5k decoder avoids increasing its binary or instruction-cache footprint.
#define DENSE_PLAIN_PAIRS   \
    DISPATCH_PAIR(32, 32)   \
    DISPATCH_PAIR(80, 256)  \
    DISPATCH_PAIR(96, 96)   \
    DISPATCH_PAIR(192, 96)  \
    DISPATCH_PAIR(256, 96)  \
    DISPATCH_PAIR(256, 128)

#define DENSE_ACTIVATION_PAIRS \
    DISPATCH_PAIR(32, 32)      \
    DISPATCH_PAIR(96, 192)

#define DENSE_UPSAMPLE2_REGISTRY \
    DISPATCH_PAIR(80, 320)

#define DENSE_UPSAMPLE4_REGISTRY \
    DISPATCH_PAIR(32, 48)        \
    DISPATCH_PAIR(96, 512)       \
    DISPATCH_PAIR(256, 256)

// These paths use dedicated fused PULSE-S kernels at runtime. Retain the
// registrations as a correctness fallback for isolated kernel tests.
#define DENSE_CHUNKS2_REGISTRY \
    DISPATCH_PAIR(128, 512)

#define DENSE_CHUNKS3_REGISTRY \
    DISPATCH_PAIR(32, 96)
