---
name: user_request
description: 用户请求模板 - 单条评论分析的用户消息
target: agent_llm
variables:
  - name: comment
    required: true
---

# 用户请求模板

## user

分析此评论：{{comment}}
