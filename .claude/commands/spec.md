# Spec Phase - Spec-Loop Method

You are operating in the Spec phase of the Spec-Loop Method. Follow these phases strictly.

## Current Request: $ARGUMENTS

## Instructions

Based on the argument provided, execute the appropriate action:

### If no argument or "status":
1. Check for existing PROJECT.md, ROADMAP.md, STATE.md, PLAN.md
2. Report current project status
3. Suggest next action based on workflow state

### If "init" or "new":
1. Ask targeted questions about the project:
   - What is the core problem being solved?
   - What are the must-have requirements?
   - What are the technical constraints?
   - What does success look like?
2. Create PROJECT.md with answers
3. Mark STATUS: DRAFT until user confirms

### If "plan" or "plan [phase-number]":
1. Read ROADMAP.md to identify current/specified phase
2. Break the phase into 2-3 atomic tasks maximum
3. Write PLAN.md using the XML task format:
```xml
<task>
  <name>Task name</name>
  <files>affected files</files>
  <actions>numbered steps</actions>
  <verification>test command</verification>
  <success>criteria</success>
</task>
```

### If "execute":
1. Read PLAN.md
2. Execute each task sequentially
3. Run verification after each task
4. Update STATE.md with progress
5. Report completion or blockers

### If "roadmap":
1. Read PROJECT.md (must be FINALIZED)
2. Create ROADMAP.md with logical phases
3. Each phase should be a coherent milestone

### If "finalize":
1. Review PROJECT.md with user
2. Confirm all requirements are captured
3. Change STATUS: DRAFT to STATUS: FINALIZED

## Rules
- Never write code without FINALIZED PROJECT.md
- Maximum 3 tasks per PLAN.md
- Always verify after implementation
- Update STATE.md after every completed task
