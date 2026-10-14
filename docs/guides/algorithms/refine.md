# Generate-test-repair and critic

Both loops **generate, check, feed the failure back, and regenerate**, bounded by `max_iterations`. They differ in what
does the checking.

| | `GenerateTestRepair` | `GeneratorCritic` |
|---|---|---|
| Checker | **your code** (a validator function) | **another LLM call** |
| Reliability | as good as your check | probabilistic |
| Cost per attempt | 1 LLM call | 2 LLM calls |
| Use when | you can write the check | you only have judgement |

## GenerateTestRepair

```python
class GenerateTestRepair:
    def __init__(
        self,
        validator="default",
        *,
        system="",
        model=None,
        max_iterations=3,
        params=None,
        extract_code=True,
    ): ...
```

```python
--8<-- "examples/generate_test_repair.py"
```

```text
success ["apple", "pear", "fig"] after 3 tries
```

1. The model produces a candidate.
2. The named **validator** judges it. A validator is any function `(candidate: str, context: dict) -> Verdict | bool |
   (bool, str)`, sync or async: a unit test run, a linter, a schema check, a business rule.
3. If accepted: success `{"candidate", "iterations"}`.
4. If rejected: the *feedback* is appended to the same conversation ("Your answer was rejected. Feedback: ...") and the
   model tries again, seeing all earlier attempts.
5. After `max_iterations` rejected attempts: `no_solution`, with the best candidate in `value` and the last feedback in `reason`.

Register validators by name on a `ValidationHandler`:

```python
from aaa.validate import ValidationHandler, Verdict, json_validator

ValidationHandler(
    {
        "default": json_validator(list[str]),  # candidate must be a JSON list of strings
        "tests": run_unit_tests,  # your function
    }
).attach(engine)

GenerateTestRepair(validator="tests")
```

`json_validator(type)` checks that the candidate parses as JSON and validates against any Pydantic-compatible type,
returning field-level feedback. `context` carries `{"task": <run input>, "iteration": n}`.

`extract_code=True` strips a surrounding markdown code fence before validation, because models like to wrap output.

!!! warning "Running generated code"
    AAA ships no code-execution sandbox, deliberately. If your validator executes model-generated code, run it in a
    sandbox you trust (container, gVisor, a restricted subprocess). The validator is *your* function and a crashing one
    is reported as a failed validation rather than killing the run.

## GeneratorCritic

```python
GeneratorCritic(critic_system=..., critic_model=None, **same_as_above)
```

After each candidate a second call, with a JSON schema `{"ok": bool, "feedback": str}`, judges it against the task. If it
rejects, `feedback` goes back to the generator. An invalid critic reply counts as a rejection (so it cannot silently
accept). Using a different `critic_model` than the generator reduces self-agreement bias.

```python
--8<-- "examples/ensembles.py"
```

```text
vote: {'answer': '42', 'votes': 3, 'samples': 5, 'agreement': 0.6}
critic loop: {'candidate': 'draft v2', 'iterations': 2}
```

## Both are bounded

There is no way for either loop to run forever: `max_iterations` caps attempts, and the engine's `Budget` caps spend.
The conversation is stored as deltas, so a long repair loop stays cheap in the log.
