# perception_training 文档目录

本目录集中存放 SA-UMNet 感知模块的设计、评估、训练策略和任务进度文档。

- [`REPRESENTATION.md`](REPRESENTATION.md)：输入、坐标系、速度、不确定性及 CBF 物理接口契约。
- [`EVALUATION.md`](EVALUATION.md)：完整感知评估指标、统计原则和 checkpoint 选择规则。
- [`STRATEGY.md`](STRATEGY.md)：监督数据采集与感知网络训练策略。
- [`DATASET_PLAN.md`](DATASET_PLAN.md)：正式数据配比、覆盖下限、场景划分与验收流程。
- [`TEST_RESULTS_12K_FP32.md`](TEST_RESULTS_12K_FP32.md)：12K FP32 最佳模型的冻结测试集结果与结论。
- [`AMP_COMPARISON_12K.md`](AMP_COMPARISON_12K.md)：12K FP32/AMP 精度、速度与显存对照。
- [`DANGER_POOL_ABLATION.md`](DANGER_POOL_ABLATION.md)：危险池化指标修正及冻结策略扫描。
- [`任务进度.md`](任务进度.md)：当前代码进度、已知限制和下一步计划。

模块入口与使用方法保留在上一级 [`README.md`](../README.md)，数据采集子模块说明保留在 [`data_collection/README.md`](../data_collection/README.md)。
