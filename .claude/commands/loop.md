# Loop Phase - Spec-Loop Method

You are operating in the Loop phase of the Spec-Loop Method - an autonomous iteration loop that keeps trying until success.

## Task: $ARGUMENTS

## Instructions

Execute the task above using the Loop methodology:

### Loop Behavior
1. Attempt to complete the task
2. Run verification (tests, build, lint, or specified command)
3. If verification fails:
   - Analyze the error output
   - Identify the root cause
   - Apply a fix
   - Return to step 2
4. If verification passes:
   - Confirm all success criteria are met
   - Say "DONE" to signal completion

### Iteration Rules
- Each iteration should make measurable progress
- Don't repeat the same failed approach more than twice
- If stuck after 3 similar attempts, try a fundamentally different approach
- Log each attempt's outcome for learning

### Verification Checklist
Before declaring DONE, confirm:
- [ ] Primary task objective is achieved
- [ ] All tests pass (if applicable)
- [ ] Build succeeds (if applicable)
- [ ] No new linting errors introduced
- [ ] Code follows project conventions

### Safety Limits
- If you've made 20+ iterations without progress, stop and report blockers
- If you're about to make a destructive change, pause and confirm
- If requirements are ambiguous, ask for clarification instead of guessing

### Output Format
After each iteration:
```
[Iteration N]
Action: What I did
Result: What happened
Next: What I'll try next (or DONE)
```

## Begin
Start working on the task. Remember: iteration over perfection. Keep trying until success.
