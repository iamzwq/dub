from __future__ import annotations


class StepError(RuntimeError):
    """可预期的流程错误：打印提示后中断。"""
