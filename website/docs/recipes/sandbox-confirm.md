---
title: Sandbox Confirm with ApprovalPolicyMiddleware
---

# Recipe: Sandbox Confirm with `ApprovalPolicyMiddleware`

Use case: cubebox-style web service where every bash command has a rule engine that classifies it as auto-allow, hard-deny, or human-confirm.

```python
from cubepi.hitl import (
    Approve, ApprovalPolicyMiddleware, AskUser, CheckpointedChannel, Deny,
)

def policy(ctx):
    cmd = ctx.args.get("cmd", "") if isinstance(ctx.args, dict) else ctx.args.cmd
    rule = command_rule_engine.classify(cmd)
    if rule.tier == "allow":   return Approve()
    if rule.tier == "block":   return Deny(reason=rule.reason)
    return AskUser(
        timeout_seconds=180,
        details={"rule": rule.matched_pattern, "impact": rule.impact},
    )

channel = CheckpointedChannel(checkpointer=cp, thread_id=thread_id)
agent = Agent(
    provider=..., model=..., tools=[bash_tool],
    middleware=[ApprovalPolicyMiddleware(channel, policy)],
    channel=channel, checkpointer=cp, thread_id=thread_id,
)
```

`HitlRequest.timeout_seconds` is embedded in the emitted event so the frontend can render a countdown.

On timeout: middleware translates to `BeforeToolCallResult(block=True, deny_reason="approval_timeout")`. The model sees `tool_result.is_error=True` with `details.hitl.decision == "timed_out"` and naturally produces a follow-up turn explaining the timeout.
