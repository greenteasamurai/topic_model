# Spec-Loop Method

This project uses the **Spec-Loop Method** for AI-assisted development.

## Overview

- **Spec Phase**: Spec-driven planning that prevents context rot
- **Loop Phase**: Autonomous iteration loops for execution

## Spec Enforcement Configuration

<!-- Change the mode below to configure enforcement behavior -->
spec-loop-enforcement: ask

**Enforcement modes:**
- `ask` (default) - Block but allow bypass via `.spec-loop-skip` file
- `block` - Hard block, no bypass allowed
- `warn` - Show warning but allow code changes
- `off` - Disable enforcement entirely

### How It Works

A hook checks for `PROJECT.md` with `STATUS: FINALIZED` before allowing code changes.

**When blocked (ask mode):** Create a `.spec-loop-skip` file in the project root to bypass. Delete it when ready to enforce again.

**When blocked (block mode):** User must create the spec first, no bypass.

**When warning (warn mode):** Show warning but continue with the work.

## Core Principles

1. **Spec before code** - Never write code without a finalized specification
2. **Fresh context per task** - Break work into atomic tasks (2-3 max per phase)
3. **Iteration over perfection** - Let loops refine the work
4. **Measurable completion** - Define clear success criteria for every task

---

## Document Structure

Maintain these documents throughout the project:

| Document | Purpose |
|----------|---------|
| `PROJECT.md` | Vision, goals, requirements, constraints |
| `ROADMAP.md` | Phases and milestones |
| `STATE.md` | Current progress and status |
| `PLAN.md` | Active atomic tasks (max 3) |
| `ISSUES.md` | Discovered problems and blockers |

---

## Spec Workflow

### Phase 1: Define
Before any implementation, create `PROJECT.md` with:
- Project vision and goals
- Core requirements (must-have vs nice-to-have)
- Technical constraints
- Success criteria

Mark as `STATUS: FINALIZED` before proceeding.

### Phase 2: Plan
Create `ROADMAP.md` with phases:
```markdown
## Phase 1: [Name]
- [ ] Task 1
- [ ] Task 2

## Phase 2: [Name]
...
```

### Phase 3: Execute
For each phase, create `PLAN.md` with atomic tasks:
```xml
<task>
  <name>Implement user authentication</name>
  <files>src/auth.ts, src/middleware/auth.ts</files>
  <actions>
    1. Create User interface with email, passwordHash fields
    2. Add hashPassword and verifyPassword functions
    3. Create authMiddleware that validates JWT
  </actions>
  <verification>npm test -- --grep "auth"</verification>
  <success>All auth tests pass</success>
</task>
```

### Phase 4: Iterate
After completing a phase:
1. Update `STATE.md` with progress
2. Review and update `ISSUES.md`
3. Plan next phase or add new phases as needed

---

## Loop Execution

For autonomous execution of well-defined tasks:

### When to Use
- Tasks with automated verification (tests, builds, lints)
- Clear, measurable completion criteria
- Iterative refinement work

### When NOT to Use
- Ambiguous requirements
- Tasks requiring human judgment
- One-shot operations

### Loop Structure
```
"[Task description]

After each change:
1. Run [verification command]
2. If it fails, analyze and fix
3. Repeat until success

Say DONE when [success criteria]."
```

### Safety Rules
- Always set iteration limits
- Define explicit completion signals
- Use sandboxed environments for risky operations

---

## Task Format

Use this XML structure for atomic tasks:

```xml
<task>
  <name>Brief task name</name>
  <files>file1.ts, file2.ts</files>
  <actions>
    1. First specific action
    2. Second specific action
    3. Third specific action
  </actions>
  <verification>Command to verify success</verification>
  <success>Measurable success criteria</success>
</task>
```

---

## Commands Reference

### Spec Commands
- `/spec` - Show workflow status and next steps
- `/spec plan` - Create or update PLAN.md for current phase
- `/spec execute` - Execute current plan tasks
- `/spec status` - Show project status

### Loop Commands
- `/loop "[task]"` - Start an iteration loop
- `/loop stop` - Stop current loop

---

## Quick Start

1. **New Project**
   ```
   Create PROJECT.md with vision and requirements
   Mark STATUS: FINALIZED when ready
   Create ROADMAP.md with phases
   /spec plan (for phase 1)
   /spec execute
   ```

2. **Existing Project**
   ```
   Analyze codebase structure
   Create PROJECT.md documenting current state
   Create ROADMAP.md for planned changes
   Continue with standard workflow
   ```

---

## Best Practices

### Planning (Spec)
- Keep PROJECT.md under 500 lines
- Maximum 3 tasks per PLAN.md
- Each task should be completable in one focused session
- Write success criteria that a machine can verify

### Execution (Loop)
- Run verification after every change
- Commit working states frequently
- Update STATE.md after completing each task
- Log blockers immediately to ISSUES.md

### Iteration
- Trust the loop - don't micromanage
- Set reasonable iteration limits (10-30 typically)
- Review outputs periodically on long loops
- Learn from failures - they inform better prompts
