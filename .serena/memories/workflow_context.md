# Workflow Context Management

## Current Workflow System
The project uses a Kiro-style workflow with agents for planning, execution, and review.

## Key Files
- **docs/context.md**: Current state and progress tracking (UTF-8)
- **docs/plan.md**: Implementation plan and steps (UTF-8)
- **.kiro/steering/**: Project-wide rules and context
- **.kiro/specs/**: Feature specifications

## Agent Roles
1. **@agent-planner**: Creates and updates implementation plans
2. **@agent-executor**: Implements one step at a time
3. **@agent-reviewer**: Reviews implementation for quality

## Workflow Loop
```
1. Planner → Create/update plan
2. Executor → Implement 1 step only
3. Reviewer → Review implementation
4. If issues → Fix → Review again
5. If OK → Next step → Back to 1
6. Continue until all steps complete
```

## Current Task Status
- **Task**: Task 2 - Common data models and interfaces
- **Branch**: task_1
- **Progress**: Step 4/8 completed
- **Coverage**: 86.29% (exceeds 80% target)

## Completed Steps
1. ✅ Tick model with Float32 constraints
2. ✅ OHLC model with TimeFrame enum
3. ✅ Prediction/Alert models
4. ✅ Base interfaces (DataFetcher, DataProcessor, StorageHandler, Predictor)

## Next Steps
5. ⏳ Config management system (src/common/config.py)
6. ⏳ Unit tests for models
7. ⏳ Unit tests for interfaces
8. ⏳ Unit tests for config

## Important Rules
- **One step at a time**: Each executor run implements exactly one step
- **Always review**: Every implementation must be reviewed
- **Update tracking**: Keep context.md and plan.md current
- **Test coverage**: Maintain >= 80% coverage
- **Quality first**: Fix review issues before proceeding