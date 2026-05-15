import React from 'react';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import styles from './HelloAgent.module.css';

const SAMPLE = `import asyncio
from cubepi import Agent, AgentTool, Model
from cubepi.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(api_key="sk-...")

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72°F and sunny in {city}"

agent = Agent(
    model=Model(provider=provider, model="claude-sonnet-4-5-20250929"),
    tools=[AgentTool(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        execute=get_weather,
    )],
    system_prompt="You are a helpful weather assistant.",
)

async def main():
    stream = await agent.prompt("What's the weather in Tokyo?")
    async for event in stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)

asyncio.run(main())
`;

export default function HelloAgent() {
  return (
    <section className={styles.section}>
      <div className={styles.left}>
        <h2 className={styles.h2}>Hello, agent.</h2>
        <p className={styles.lede}>
          A single async function loop. One <code>Provider</code>, one <code>AgentTool</code>, and you're streaming.
        </p>
        <Link to="/docs/getting-started/quick-start" className={styles.link}>
          Full quick-start →
        </Link>
      </div>
      <div className={styles.right}>
        <CodeBlock language="python">{SAMPLE}</CodeBlock>
      </div>
    </section>
  );
}
