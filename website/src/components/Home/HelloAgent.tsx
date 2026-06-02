import React from 'react';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import { useIsZhHans } from '@site/src/hooks/useIsZhHans';
import styles from './HelloAgent.module.css';

const SAMPLE = `import asyncio
from pydantic import BaseModel
from cubepi import Agent, AgentTool, Model
from cubepi.agent.types import AgentToolResult
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.base import TextContent

provider = AnthropicProvider(api_key="sk-...")

class GetWeatherParams(BaseModel):
    city: str

async def get_weather(tool_call_id, params: GetWeatherParams, *, signal=None, on_update=None):
    return AgentToolResult(
        content=[TextContent(text=f"72°F and sunny in {params.city}")]
    )

agent = Agent(
    provider=provider,
    model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
    tools=[AgentTool(
        name="get_weather",
        description="Get current weather for a city",
        parameters=GetWeatherParams,
        execute=get_weather,
    )],
    system_prompt="You are a helpful weather assistant.",
)

def on_event(event, signal=None):
    if event.type == "text_delta":
        print(event.delta, end="", flush=True)

agent.subscribe(on_event)
asyncio.run(agent.prompt("What's the weather in Tokyo?"))
`;

export default function HelloAgent() {
  const zh = useIsZhHans();
  return (
    <section className={styles.section}>
      <div className={styles.left}>
        <h2 className={styles.h2}>{zh ? '你好，agent。' : 'Hello, agent.'}</h2>
        <p className={styles.lede}>
          {zh ? (
            <>一个 async 函数循环。一个 <code>Provider</code>，一个 <code>AgentTool</code>，即可开始流式输出。</>
          ) : (
            <>A single async function loop. One <code>Provider</code>, one <code>AgentTool</code>, and you're streaming.</>
          )}
        </p>
        <Link to="/docs/getting-started/quick-start" className={styles.link}>
          {zh ? '完整快速开始 →' : 'Full quick-start →'}
        </Link>
      </div>
      <div className={styles.right}>
        <CodeBlock language="python">{SAMPLE}</CodeBlock>
      </div>
    </section>
  );
}
