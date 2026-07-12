"""Bounded, request-local access to provider tools for specialist agents."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from ..models.multi_agent import AgentRole, CallBudgetLedger, CandidateRegistry, RegistryEntity


class ToolGatewayError(RuntimeError):
    def __init__(self, code: str, message: str, *, transient: bool = False):
        super().__init__(message)
        self.code = code
        self.transient = transient


class ToolGateway:
    """Enforce role allowlists, budgets, deduplication, and registry ownership."""

    DEFAULT_ALLOWLIST = {
        "experience": {"attraction_search", "rag_search", "place_detail"},
        "logistics": {"hotel_search", "meal_search"},
        "composer": set(),
    }

    def __init__(
        self,
        *,
        registry: CandidateRegistry,
        tools: Optional[Mapping[str, Callable[..., Any]]] = None,
        budgets: Optional[Mapping[str, int]] = None,
        allowlist: Optional[Mapping[AgentRole, Iterable[str]]] = None,
        ledger: Optional[CallBudgetLedger] = None,
    ):
        self.registry = registry
        self.tools = dict(tools or {})
        self.budgets = dict(budgets or {})
        configured = allowlist or self.DEFAULT_ALLOWLIST
        self.allowlist = {role: set(names) for role, names in configured.items()}
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.ledger = ledger
        self.early_stop_reasons: Dict[str, str] = {}
        self._query_cache: Dict[tuple[str, str, str], Any] = {}

    def call(self, role: AgentRole, tool_name: str, *, query_key: str = "", **kwargs: Any) -> Any:
        if tool_name not in self.allowlist.get(role, set()):
            raise ToolGatewayError("tool_not_allowed", f"{role} cannot call {tool_name}")
        cache_key = (role, tool_name, query_key.strip().lower())
        if query_key and cache_key in self._query_cache:
            return self._query_cache[cache_key]
        limit = self.budgets.get(tool_name)
        if limit is not None and self.call_counts[tool_name] >= limit:
            raise ToolGatewayError("tool_budget_exhausted", f"budget exhausted for {tool_name}")
        tool = self.tools.get(tool_name)
        if tool is None:
            raise ToolGatewayError("tool_unavailable", f"tool unavailable: {tool_name}")
        resource = "rag" if tool_name == "rag_search" else "maps"
        if self.ledger is not None:
            try:
                self.ledger.consume(role, resource, tool_name)
            except ValueError as exc:
                raise ToolGatewayError("tool_budget_exhausted", str(exc)) from exc
        self.call_counts[tool_name] += 1
        try:
            result = tool(**kwargs)
        except (TimeoutError, ConnectionError) as exc:
            raise ToolGatewayError("provider_transient", str(exc), transient=True) from exc
        except ToolGatewayError:
            raise
        except Exception as exc:
            raise ToolGatewayError("provider_error", str(exc)) from exc
        if query_key:
            self._query_cache[cache_key] = result
        return result

    def register(self, role: AgentRole, entities: Iterable[RegistryEntity]) -> None:
        for entity in entities:
            self.registry.add(entity, actor=role)


class BudgetedRunnable:
    def __init__(self, runnable: Any, role: AgentRole, ledger: CallBudgetLedger):
        self.runnable = runnable
        self.role = role
        self.ledger = ledger

    def invoke(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        self.ledger.consume(self.role, "llm", "structured_output")
        return self.runnable.invoke(value, *args, **kwargs)


class BudgetedLLM:
    def __init__(self, llm: Any, role: AgentRole, ledger: CallBudgetLedger):
        self.llm = llm
        self.role = role
        self.ledger = ledger

    def invoke(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        self.ledger.consume(self.role, "llm", "invoke")
        return self.llm.invoke(value, *args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> BudgetedRunnable:
        if hasattr(self.llm, "with_structured_output"):
            return BudgetedRunnable(self.llm.with_structured_output(*args, **kwargs), self.role, self.ledger)
        schema = args[0]

        class FallbackRunnable:
            def __init__(self, llm: Any):
                self.llm = llm

            def invoke(self, value: Any, *invoke_args: Any, **invoke_kwargs: Any) -> Any:
                response = self.llm.invoke(value, *invoke_args, **invoke_kwargs)
                content = getattr(response, "content", response)
                return schema.model_validate(content if isinstance(content, dict) else json.loads(str(content)))

        return BudgetedRunnable(FallbackRunnable(self.llm), self.role, self.ledger)
