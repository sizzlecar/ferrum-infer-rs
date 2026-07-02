# Removed FA2 Source Bridge Inputs

The old source-linked FA2 bridge used this directory for vendored
FlashAttention and CUTLASS inputs. Those bulk third-party C++/CUDA sources are
no longer stored in the main repository.

This directory now only keeps Ferrum-owned compatibility stubs for historical
paths. The `fa2-source` Cargo feature is an obsolete alias and must not compile
native operator source from `crates/`.

FA2 support must be provided through a Ferrum native operator artifact with a
validated manifest and matching binary hash.
