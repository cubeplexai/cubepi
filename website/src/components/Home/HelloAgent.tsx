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
    provider=provider,
    model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
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

def on_event(event, signal=None):
    if event.type == "text_delta":
        print(event.delta, end="", flush=True)

agent.subscribe(on_event)
asyncio.run(agent.prompt("What's the weather in Tokyo?"))
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
