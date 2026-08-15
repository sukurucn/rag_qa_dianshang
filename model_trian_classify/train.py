"""Query-router 教师训练与学生蒸馏的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_trian_classify.distillation_trainer import DistillationConfig, train_student
from model_trian_classify.teacher_trainer import TeacherTrainingConfig, train_teacher


def main() -> None:
    """解析执行阶段与可选数据集路径。"""
    parser = argparse.ArgumentParser(description="Train and distill the query-router classifier.")
    parser.add_argument("stage", choices=("teacher", "student", "all"))
    parser.add_argument("--data-path", type=Path, default=None)
    arguments = parser.parse_args()

    results: dict[str, object] = {}
    if arguments.stage in {"teacher", "all"}:
        teacher_config = TeacherTrainingConfig(
            data_path=arguments.data_path or TeacherTrainingConfig().data_path
        )
        results["teacher"] = train_teacher(teacher_config)
    if arguments.stage in {"student", "all"}:
        student_config = DistillationConfig(
            data_path=arguments.data_path or DistillationConfig().data_path
        )
        results["student"] = train_student(student_config)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
