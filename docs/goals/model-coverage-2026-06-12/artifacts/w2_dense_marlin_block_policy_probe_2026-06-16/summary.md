# W2 dense Marlin block-policy native probe

- Status: PASS, diagnostic only, not release-grade evidence.
- Probe rc: 0; VERDICT line present.
- Finding: block-policy override is not a useful product optimization for the current W2 gap.
- m16 gate_up auto weight-cycle: default 133.956us, blocks_n_tiles 134.284us, blocks_2sms 134.647us.
- m16 down auto weight-cycle: default 68.689us, blocks_n_tiles 74.203us, blocks_2sms 74.354us.
- Next lever: leave dense Marlin block policy alone; move to decode integration / launch-count / prefill TTFT work.
