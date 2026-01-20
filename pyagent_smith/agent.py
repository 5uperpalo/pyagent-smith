from typing import Any, List, Optional, Dict

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from pyagent_smith.llm import create_ollama_chat_llm
from pyagent_smith.tools import calculator, web_crawler, web_search_tool, web_search_tool_with_self_consistency
from pyagent_smith.langsmith_utils import setup_langsmith_tracing

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class VerboseToolCallbackHandler(BaseCallbackHandler):
    """Callback handler for verbose tool output."""

    def __init__(self, n_chars: int = 10000):
        super().__init__()
        self.n_chars = n_chars

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "unknown")
        print(f"[Tool] {tool_name} started")
        print(f"[Tool] Input: {input_str[:self.n_chars]}{'...' if len(input_str) > self.n_chars else ''}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool finishes running."""
        output_str = str(output)
        print(f"[Tool] Output: {output_str[:self.n_chars]}{'...' if len(output_str) > self.n_chars else ''}")
        print(f"[Tool] Output length: {len(output_str)} characters")

    def on_tool_error(
        self, error: BaseException, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        """Called when a tool encounters an error."""
        print(f"[Tool] Error: {error}")


class ChatAgent:
    """Wrapper around CompiledStateGraph that can optionally maintain chat history."""

    def __init__(
        self,
        agent_graph: Any,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        retain_history: bool = True,
    ):
        self.agent_graph = agent_graph
        self.chat_history: List[Any] = []
        self.callbacks = callbacks
        self.retain_history = retain_history

    def invoke(
        self,
        user_input: str,
        run_name: Optional[str] = None,
    ) -> str:
        """Invoke the agent with user input and automatically update chat history.

        Args:
            user_input: The user's input/question
            run_name: Optional name for this run (used in LangSmith tracing)
            tags: Optional list of tags to attach to the trace
            metadata: Optional dictionary of metadata to attach to the trace

        Returns:
            The agent's response as a string, or an error message if something goes wrong
        """
        try:
            # Use chat history only if retention is enabled
            if self.retain_history:
                messages = self.chat_history + [HumanMessage(content=user_input)]
            else:
                messages = [HumanMessage(content=user_input)]
            config = {"callbacks": self.callbacks} if self.callbacks else {}

            result = self.agent_graph.invoke({"messages": messages}, config=config)  # type: ignore[arg-type]

            # Extract the output from the result
            # LangChain v1's create_agent returns a state dict with "messages" key
            # containing all messages including the new AI response
            if isinstance(result, dict) and "messages" in result:
                messages_result = result["messages"]
                # Find the last AIMessage in the result (the agent's response)
                # We need to find messages that weren't in our input
                input_message_count = len(messages)
                if len(messages_result) > input_message_count:
                    # Get messages that were added by the agent
                    new_messages = messages_result[input_message_count:]
                    # Find the last AIMessage
                    for msg in reversed(new_messages):
                        if isinstance(msg, AIMessage):
                            output = str(msg.content)
                            break
                    else:
                        # Fallback: if no AIMessage found, get the last message content
                        output = str(new_messages[-1].content) if new_messages else ""
                else:
                    # If no new messages, try to get the last AIMessage from all messages
                    for msg in reversed(messages_result):
                        if isinstance(msg, AIMessage):
                            output = str(msg.content)  # type: ignore[assignment]
                            break
                    else:
                        output = str(result.get("output", ""))
            else:
                # Fallback: try to get "output" key or convert to string
                output = str(result.get("output", result))

            # Update chat history only if retention is enabled
            if self.retain_history:
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=output))

            return output
        except Exception as e:
            return f"Agent error: {e}"

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []


def build_chat_agent(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    *,
    verbose: bool = False,
    verbose_n_chars: int = 10000,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    retain_chat_history: bool = False,
    use_self_consistency_search: bool = False,
    enable_langsmith: bool = False,
    langsmith_project: Optional[str] = None,
) -> Optional[Any]:
    """Build a chat agent that can be continuously prompted.

    Args:
        system_prompt: The system prompt to use for the agent
        verbose: Whether to enable verbose output (currently not fully utilized)
        callbacks: Optional list of callback handlers for monitoring agent execution
        retain_chat_history: If True, returns a ChatAgent wrapper that maintains chat history
                            between calls. If False, returns a ChatAgent that doesn't retain history
                            (each call is independent, but still accepts string inputs).
        use_self_consistency_search: If True, enables self-consistency approach with majority voting
                                    for web search using multiple search engines (default: False)
        enable_langsmith: If True, automatically sets up LangSmith tracing if configured (default: True)
        langsmith_project: Optional project name for LangSmith (overrides config/env)

    Returns:
        A ChatAgent instance if Ollama is configured, None otherwise.
        ChatAgent always provides a string-based interface (accepts strings, returns strings).
        If retain_chat_history=True, the agent maintains conversation history between calls.
        If retain_chat_history=False, each call is independent (no history retention).
    """
    # Set up LangSmith tracing if enabled
    if enable_langsmith:
        setup_langsmith_tracing(project=langsmith_project)

    llm = create_ollama_chat_llm()
    if llm is None:
        return None

    if use_self_consistency_search:
        search_tool = web_search_tool_with_self_consistency
    else:
        search_tool = web_search_tool

    tools = [
        calculator,
        web_crawler,
        search_tool,
    ]

    # Create agent using LangChain v1's create_agent
    # It takes model, tools, and system_prompt (as a string or SystemMessage)
    agent_graph: Any = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    # Add verbose callback if verbose is enabled
    final_callbacks = list(callbacks) if callbacks else []
    if verbose:
        final_callbacks.append(VerboseToolCallbackHandler(n_chars=verbose_n_chars))

    if retain_chat_history:
        return ChatAgent(agent_graph, callbacks=final_callbacks, retain_history=True)
    # Return ChatAgent without history retention for simpler string-based interface
    return ChatAgent(agent_graph, callbacks=final_callbacks, retain_history=False)
