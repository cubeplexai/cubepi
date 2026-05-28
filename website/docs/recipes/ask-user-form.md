---
title: Multi-question Form via ask_user
---

# Recipe: Multi-question Form via `ask_user`

```python
from cubepi.hitl import ask_user_tool, InMemoryChannel

channel = InMemoryChannel()
agent = Agent(
    provider=..., model=...,
    tools=[ask_user_tool(channel)],
    channel=channel,
)
```

The model invokes `ask_user` like any other tool. Example parameters the model can pass:

```json
{
  "questions": [
    {"key": "framework", "prompt": "Which framework?",
     "options": [
       {"label": "React", "value": "react"},
       {"label": "Vue", "value": "vue"},
       {"label": "Other", "value": "other", "allow_input": true}
     ]},
    {"key": "features", "prompt": "Which features?",
     "multi_select": true,
     "options": [
       {"label": "Auth", "value": "auth"},
       {"label": "Payments", "value": "payments"}
     ]}
  ]
}
```

Answer shape: `{"framework": "react", "features": ["auth", "payments"]}` — or for `Other` with `allow_input`, the value is the free-text string the user typed.
