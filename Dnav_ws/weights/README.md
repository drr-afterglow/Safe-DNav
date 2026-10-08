# Safe-DNav deployment weights

| File | SHA-256 | Purpose |
|---|---|---|
| `policy_best.pt` | `834affd0ecaebbe966254ef64ef322cea4ce5c241672a9dfe4de650c763fb8a0` | Final five-channel PPO policy |
| `perception_best.ts.pt` | `8c164084323a4077ef7f700c24cb7ef2bd50178d5b2ac1f7687bca4871a23399` | Current portable SA-UMNet TorchScript |
| `perception_best_legacy_cpu.ts.pt` | `08aed4849714f92e184903a41ed8625825004c0ba52e0b313a3ccb0e9f40d8a9` | Legacy perception export retained for regression |

The policy loader checks the final policy hash, all 39 source keys, tensor
shapes and finite values before mapping the inference subset. The deployment
actor uses the Beta distribution mean and produces a body-frame velocity in
`[-3, 3] m/s`, followed by full-quaternion rotation into the world frame.
