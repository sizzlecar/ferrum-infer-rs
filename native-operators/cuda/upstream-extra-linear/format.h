// Independent production extra-format ABI; the original operator stays closed.
#pragma once
static constexpr bool ferrum_linear_format(unsigned f) { return f==11 || f==20 || f==21; }
static constexpr unsigned ferrum_linear_qk(unsigned f) { return f==20 ? 32 : 256; }
static constexpr unsigned ferrum_linear_bytes(unsigned f) { return f==20 ? 18 : 110; }
static constexpr bool ferrum_linear_d4(unsigned) { return true; }
