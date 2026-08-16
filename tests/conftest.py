"""测试期间禁止向外部 LangSmith 服务发送真实 trace。"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
