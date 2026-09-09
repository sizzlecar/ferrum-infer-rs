# 下一版本 GGUF 输入冻结

2026-09-09 冻结下列三个输入，供数值策略解耦与跨后端 GGUF 开发复用。
来源 revision 和摘要用于复现，不作为机器、路径或提交号正确性门槛。
目前只有来源及描述符盘点证据，尚未完成新版本模型运行或发布验收。

## 权重与语义来源

| 输入 | 权重仓库与固定 revision | 文件 | 完整文件字节数 |
| --- | --- | --- | ---: |
| N1：4B | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523) | `Qwen3.5-4B-Q4_K_M.gguf` | 2740937888 |
| N2：9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5) | `Qwen3.5-9B-Q4_K_M.gguf` | 5680522464 |
| N3：27B | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) | `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16464440224 |

权重 SHA-256（HF 固定 revision 的 LFS 元数据）：

- N1：`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
- N2：`03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`
- N3：`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`

N1 本地完整文件重新计算的 SHA-256 与上值一致。N2、N3 本次仅下载了包含完整描述符表的文件前缀；HTTP Content-Range 与固定来源的完整长度一致，未验证完整权重摘要。进入产品运行验收前仍需取得并校验完整文件。

三个输入均为单文件，声明 `general.quantization_version = 2`，描述符未声明 `split.*`。本轮未要求通过重分片或换包规避 N3 的实际编码。

独立语义源采用模型作者仓库的 `config.json`、`tokenizer.json`、`tokenizer_config.json` 和 `chat_template.jinja`；存在 `generation_config.json` 时一并记录。禁止混用下载缓存中的其他 revision 模板。

| 输入 | semantic/tokenizer/template 固定来源 | tokenizer.json SHA-256 | chat_template.jinja Git blob ID |
| --- | --- | --- | --- |
| N1 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` | `a585dec894e63da457d9440ec6aa7caa16d20860` |
| N2 | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` | `a585dec894e63da457d9440ec6aa7caa16d20860` |
| N3 | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` | `c0c686f9c38d70d179fb7b5f5aa7530bc913dda3` |

Git blob ID 包含 Git 对象前缀，与文件内容的普通 SHA-1／SHA-256 不可互换。N1、N2 的 tokenizer 和模板在上述来源中相同，模型配置仍各自独立。

## 完整描述符盘点

使用 [Rust GGUF inventory 工具](release-regression.md#gguf-source-inventory) 读取全部描述符，不物化张量。工具输出每个张量的外部名称、row-major shape、原始 GGML type、block ABI、偏移及实际存储字节；以下为类型汇总，`Q3K` 等沿用工具中 Candle 的名称。

| 实际 tensor dtype | N1：4B | N2：9B | N3：27B |
| --- | ---: | ---: | ---: |
| F32 | 177 | 177 | 360 |
| Q3K | 0 | 0 | 7 |
| Q4K | 131 | 132 | 104 |
| Q5K | 48 | 48 | 131 |
| Q6K | 22 | 22 | 30 |
| Q8_0 | 48 | 48 | 106 |
| IQ3_S | 0 | 0 | 4 |
| IQ4_NL | 0 | 0 | 7 |
| IQ4_XS | 0 | 0 | 117 |
| 合计 | 426 | 427 | 866 |
| tensor payload bytes（不含 header／间隙） | 2729969664 | 5669554176 | 16453443584 |

N1、N2 的 `Q4_K_M` 已混合五种 dtype，不能把文件名视为每个张量都是 Q4_K。N3 的 UD 配方混合九种 dtype，其中 128 个 IQ 张量超出现有 Candle runtime reader 的识别范围。独立 inventory reader 能报告这些描述符，不代表已具备解码、上传或 GPU 运算支持。

N3 的 IQ 张量涉及 FFN gate/up/down、attention q/output，以及 GDN 的 `attn_gate`、`attn_qkv`、`ssm_out` 等外部张量。后续须按 family 的真实角色映射、shape 与选定 numerical profile 完成所需 reader/source/schema/provider 支持和数值验收。不能把 IQ4_XS 通用扩展列为后续工作而跳过本输入实际需要的部分，也不能换成较容易的量化包完成 N3。

本轮完成的是 M0 的来源冻结与完整描述符盘点。真实角色绑定、语义文件在运行进程中的身份验证、既有路径的同机数值和性能基线、运行前登记的误差及资源预算仍需完成；本表不替代这些证据。完整 inventory、下载前缀、API 原始响应及后续测量日志均保存在仓库外。
