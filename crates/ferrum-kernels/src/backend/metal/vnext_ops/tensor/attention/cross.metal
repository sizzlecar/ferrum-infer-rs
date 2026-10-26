// Compile the shared tiled attention body as rectangular noncausal attention.
// Q and KV have independent batch strides; visibility still comes exclusively
// from the validated per-batch key runs. No second arithmetic implementation.
#define FERRUM_ATTENTION_CROSS 1
#define ferrum_tensor_attention_bf16 ferrum_tensor_cross_attention_bf16
